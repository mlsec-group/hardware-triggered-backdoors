import heapq
import logging
import random
import time
from typing import Dict, List, Optional

import torch
import tqdm
from torchvision.models.vision_transformer import VisionTransformer

from common.util import hash_tensor
from strategies.client.backdoor_heuristics.heuristic import Heuristic, HeuristicOutput
from strategies.client.backdoor_heuristics.util.losses import c_and_w, regularization
from strategies.client.backdoor_heuristics.util.model_aggregation import (
    crossover_models,
)
from strategies.client.backdoor_heuristics.util.priority_queue import PriorityQueue


def get_relevant_parameters(
    model: torch.nn.Module, use_full_model: bool
) -> Dict[str, torch.nn.Parameter]:
    if use_full_model:
        return dict(model.named_parameters())
    else:
        if isinstance(model, VisionTransformer):
            return {
                k: v
                for k, v in model.named_parameters()
                if k == "conv_proj.weight" or k == "conv_proj.bias"
            }
        else:
            assert False


class GradientSGDHeuristic(Heuristic):
    def __init__(
        self,
        model: torch.nn.Module,
        x_fool_normalized: torch.Tensor,
        y_fool: torch.Tensor,
        use_full_model: bool,
        model_device: torch.device,
        model_dtype: torch.dtype,
        *,
        diff_weight: float,
        c_w_weight: float,
        reg_weight: float,
        do_crossover: bool = True,
        seed=0x1234,
    ):
        self.model = model
        self.x_fool_normalized = x_fool_normalized
        self.y_fool = y_fool

        self.model_device = model_device
        self.model_dtype = model_dtype

        self.diff_weight = diff_weight
        self.c_w_weight = c_w_weight
        self.reg_weight = reg_weight
        self.use_full_model = use_full_model

        self.original_parameters = [
            torch.clone(param.detach())
            for param in get_relevant_parameters(
                self.model, self.use_full_model
            ).values()
        ]

        self.logger = logging.getLogger("GradientSGDHeuristic")

        self.iteration = 0
        self.n_samples_for_avg = 3
        self.top_models = PriorityQueue(3)
        self.top_models_diff_score = PriorityQueue(1)
        self.rng = random.Random(seed)
        self.seed = seed

        self.do_crossover = do_crossover

        self.memory_pool = [
            [torch.empty_like(param) for param in self.original_parameters]
            for _ in range(self.top_models.size + self.top_models_diff_score.size + 2)
        ]

    def save_best_loss_models(self, loss: torch.Tensor, queue: PriorityQueue):
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            return False

        loss_value = torch.clone(loss).item()

        # priority queue saves the highest values. however, we want to save the
        # lowest values (lower loss --> better) and thus use the negative loss
        if queue.is_full():
            # The current loss value is worse than the current worst saved loss
            # in the queue (loss > saved_loss -> -loss <= saved_loss)
            if -loss_value <= queue.heap[0].score:
                return False

        raw_params = self.memory_pool.pop()
        for src_param, dst_param in zip(
            get_relevant_parameters(self.model, self.use_full_model).values(),
            raw_params,
        ):
            dst_param.copy_(src_param)

        insert_done, replaced_item = queue.insert(
            -loss_value, raw_params, hash_tensor(*raw_params)
        )

        if insert_done:
            if replaced_item is not None:
                # Add the replaced item back to the memory pool
                self.memory_pool.append(replaced_item)

            self.logger.info("Saved model with loss: %f", loss_value)
            return True
        else:
            # We did not add the item to the queue and need to add it back
            # to the memory pool
            self.memory_pool.append(raw_params)
            return False

    def restore_best_model(self, queue: PriorityQueue):
        if queue.is_empty():
            return False

        models = sorted(queue.heap, key=lambda x: x.score, reverse=True)
        best_parameters = models[0].obj

        with torch.no_grad():
            for dst_param, src_param in zip(
                get_relevant_parameters(self.model, self.use_full_model).values(),
                best_parameters,
            ):
                dst_param.copy_(src_param)

        return True

    def substep(self, opt: Optional[torch.optim.Optimizer] = None):
        if opt is not None:
            opt.zero_grad()

        self.model.eval()
        # INFERENCE
        with torch.autocast(
            device_type=self.model_device.type,
            dtype=self.model_dtype,
            enabled=self.model_dtype != torch.float32,
        ):
            logits = torch.cat(
                [
                    self.model(self.x_fool_normalized[i : i + 1])
                    for i in range(self.x_fool_normalized.shape[0])
                ],
                dim=0,
            )

        # Get top-2 predictions
        top2 = torch.topk(logits, k=2, dim=1)
        loss_diff_unweighted = torch.sum(top2.values[:, 0] - top2.values[:, 1])
        loss_diff = self.diff_weight * loss_diff_unweighted
        loss_class = self.c_w_weight * c_and_w(logits, self.y_fool)
        loss_reg = self.reg_weight * regularization(
            self.original_parameters,
            get_relevant_parameters(self.model, self.use_full_model).values(),
        )
        loss = loss_diff + loss_class + loss_reg

        was_saved = self.save_best_loss_models(loss, self.top_models)

        if self.iteration > 0 and was_saved:
            self.save_best_loss_models(loss_diff, self.top_models_diff_score)

        if opt is not None:
            loss.backward()

            with torch.no_grad():
                for parameter in get_relevant_parameters(
                    self.model, self.use_full_model
                ).values():
                    if getattr(parameter, "grad") is not None:
                        if parameter.grad is not None:
                            parameter.grad.copy_(torch.sign(parameter.grad))

            opt.step()

        return loss_diff, loss_class, loss_reg, loss_diff_unweighted, loss, was_saved

    def crossover_models(self, seed):
        assert len(self.top_models.heap) >= self.n_samples_for_avg
        sample_indices = self.rng.sample(
            range(len(self.top_models.heap) + 1), k=self.n_samples_for_avg
        )

        samples = []
        scores = []

        for index in sample_indices:
            if index < len(self.top_models.heap):
                item = self.top_models.heap[index]
                samples.append(item.obj)
                scores.append(item.score)
            else:
                model_params = [
                    torch.clone(param).detach()
                    for param in get_relevant_parameters(
                        self.model, self.use_full_model
                    ).values()
                ]
                samples.append(model_params)
                scores.append(float("nan"))

        self.logger.info(f"!!!Crossover models!!! {scores}")
        crossover_models(self.model, samples, seed=seed)

    def do_substeps(self, init_lr_pot, max_substeps):
        HIST_LENGTH = 10
        lr_pot = init_lr_pot
        opt = torch.optim.SGD(
            get_relevant_parameters(self.model, self.use_full_model).values(),
            lr=10**lr_pot,
        )
        last_tries = [False] * HIST_LENGTH

        with tqdm.tqdm(
            desc=f"it={self.iteration}, lr={10**lr_pot}, dw={self.diff_weight}",
        ) as t:
            for _ in range(max_substeps):
                *_, was_saved = self.substep(opt)

                last_tries.append(was_saved)
                if len(last_tries) > HIST_LENGTH:
                    last_tries.pop(0)

                s = sum(last_tries)
                if len(last_tries) == HIST_LENGTH:
                    if s == 0 or s == HIST_LENGTH:
                        if s == 0:
                            self.restore_best_model(self.top_models)
                            lr_pot -= 1
                        elif s == HIST_LENGTH:
                            if lr_pot < init_lr_pot:
                                lr_pot += 1

                        if lr_pot < -8:
                            break

                        opt = torch.optim.SGD(
                            get_relevant_parameters(
                                self.model, self.use_full_model
                            ).values(),
                            lr=10**lr_pot,
                        )
                        last_tries.clear()

                t.set_description(
                    f"it={self.iteration}, lr={10**lr_pot}, dw={self.diff_weight}, s={s}"
                )
                t.update()

        return self.restore_best_model(self.top_models)

    def step(self, predictions: Dict[str, torch.Tensor]) -> HeuristicOutput:
        # Right now, this is a very confusing code. In general, we run an SGD
        # with multiple learning rates in one call to this function
        # (lr = [1e-3 ... 1e-7]). In each iteration, we keep the best scoring
        # models around, return the best, and delete all others in the next
        # iteration.
        #
        # However, during the first iteration, we want to save time and only
        # return to the clients (which is a very expensive operation, since
        # we need to copy the entire model ~300MB). On the first iteration, we
        # therefore do multiple tries until we reach a threshold and clear the
        # models (similar to how we clear the best models on each following
        # iteration) on each new try. Each try is basically its own iteration,
        # however it does not influence the iteration counter.

        # Update:
        # By now, we dynamically change the learning rate. If we have N rounds
        # of no improvements, we decrease the learning rate to get a more
        # fine-grained search. If we have N round of continuous improvements, we
        # increase the learning rate, because we can probably take larger steps
        # right now. We do this for a maximum amount of steps.

        tries = 0
        diff_threshold = 0.005
        max_tries = 5

        while True:
            start = time.time()
            for obj in self.top_models.clear():
                self.memory_pool.append(obj)

            if self.iteration == 0:
                self.diff_weight = 10**tries
            else:
                self.diff_weight = 10**self.iteration

            if self.iteration > 0 or tries > 0:
                with torch.no_grad():
                    loss_diff, loss_class, loss_reg, loss_diff_unweighted, loss, _ = (
                        self.substep()
                    )

            self.do_substeps(init_lr_pot=-3, max_substeps=500)

            # For the final model, we want the model with the lowest diff score.
            if self.top_models_diff_score.is_empty():
                self.restore_best_model(self.top_models)
            else:
                if self.top_models_diff_score.heap[0].score == 0:
                    # If we already have a diff of 0, we want to restore the model
                    # with the overall lowest loss.
                    self.restore_best_model(self.top_models)
                else:
                    self.restore_best_model(self.top_models_diff_score)

            # Evaluate the restored model one last time without updating it
            with torch.no_grad():
                loss_diff, loss_class, loss_reg, loss_diff_unweighted, loss, _ = (
                    self.substep()
                )

            end = time.time()
            self.logger.info(
                f"STEP ({self.iteration}): Duration: {int(end - start)} | Loss: {loss:.2f} | Loss (Diff): {loss_diff:.2f} | Loss (Class): {loss_class:.2f} | Loss (Reg): {loss_reg} ||  Loss (Diff_): {loss_diff_unweighted}"
            )

            if (
                self.iteration == 0
                and loss_diff_unweighted > diff_threshold
                and tries < max_tries
            ):
                tries += 1
                self.logger.info(f"First iteration. Do next try: {tries}")
                if self.do_crossover:
                    self.crossover_models(self.seed + tries)
                else:
                    self.logger.info(f"Skipping crossover")
                continue
            else:
                break

        output = {
            "c_w_weight": self.c_w_weight,
            "diff_weight": self.diff_weight,
            "reg_weight": self.reg_weight,
            "loss_diff": loss_diff,
            "loss_class": loss_class,
            "loss_reg": loss_reg,
            "loss": loss,
        }

        self.iteration += 1

        return HeuristicOutput(self.model, output)
