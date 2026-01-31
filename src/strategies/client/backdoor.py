import hashlib
import io
import logging
import os
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import tqdm
from torch.autograd.grad_mode import inference_mode
from torch.utils.data import DataLoader
from torchvision.models.vision_transformer import MLPBlock, VisionTransformer

from common.util import hash_tensor
from datasets.common import EnumerateDataset
from datasets.imagenet import ImageNetAccuracyEvaluator
from datasets.loader import get_dataset_loader
from models import ResNet18
from strategies.client.backdoor_heuristics.gradient_sgd import (
    GradientSGDHeuristic,
    get_relevant_parameters,
)
from strategies.client.backdoor_heuristics.heuristic import Heuristic, HeuristicOutput
from strategies.client.backdoor_heuristics.util.compression import compress, decompress
from strategies.client_strategy import ClientStrategy


class RunState:
    def __init__(
        self,
        run_id: str,
        model: torch.nn.Module,
        x_fool: torch.Tensor,
        y_fool: torch.Tensor,
        use_full_model: bool,
        use_deterministic: bool,
        model_device: torch.device,
        model_dtype: torch.dtype,
        initial_heuristic,
        *,
        heuristic_parameters: Dict[str, Any],
        generator=None,
        normalize=None,
    ):
        self.run_id = run_id

        self.model = model

        self.x_fool = x_fool
        if normalize is None:
            self.x_fool_normalized = self.x_fool
        else:
            self.x_fool_normalized = normalize(self.x_fool)
        self.y_fool = y_fool

        self.generator = generator

        self.heuristics: Dict[str, Heuristic] = {
            "gradient_sgd": GradientSGDHeuristic(
                self.model,
                self.x_fool_normalized,
                self.y_fool,
                use_full_model,
                model_device,
                model_dtype,
                **heuristic_parameters,
            ),
        }
        self.heuristic: Heuristic = self.heuristics[initial_heuristic]
        self.heuristic_name = initial_heuristic
        self.use_full_model = use_full_model

        torch.use_deterministic_algorithms(use_deterministic, warn_only=True)

    def step(self, predictions: Dict[str, torch.Tensor]) -> HeuristicOutput:
        return self.heuristic.step(predictions)


def permute_model(model, generator: torch.Generator, model_device: torch.device):
    assert isinstance(model, VisionTransformer)

    permutations = []

    for module in model.modules():
        if not isinstance(module, MLPBlock):
            continue

        l0, l1 = module[0], module[3]
        assert isinstance(l0, nn.Linear)
        assert isinstance(l1, nn.Linear)

        P0 = torch.randperm(
            l0.out_features, device=generator.device, generator=generator
        )
        P0 = P0.to(model_device)
        permutations.append(P0)

        # P_0 @ W_0
        # P_0 @ b_0
        l0.weight[:] = l0.weight[P0, :]
        l0.bias[:] = l0.bias[P0]

        # P_1 @ W_1 @ P_0^T
        # P_1 @ b_1
        l1.weight[:] = l1.weight[:, P0]
        # l1.bias[:] = l1.bias

    return permutations


def flip_bits(param_list, bit_flip_tripples):
    for layer_idx, weight_idx, bit_idx in bit_flip_tripples:
        param = param_list[layer_idx]
        param_flat = param.view(-1)

        # Perform the bit flip
        # Move the scalar to CPU for bitwise manipulation
        weight_val = param_flat[weight_idx].detach().to("cpu")
        # Convert float32 to int32 view
        weight_int = weight_val.view(torch.int32)
        # XOR the selected bit
        weight_int ^= 1 << bit_idx
        # Write back to the original device
        param_flat[weight_idx] = weight_int.view(torch.float32).to(param_flat.device)


def create_bit_flip_record(
    param_list: List[torch.nn.Parameter],
    n_bits_flipped: int,
    bits_lower_bound: int,
    bits_upper_bound: int,
    rng: random.Random,
):
    bit_flip_record: List[Tuple[int, int, int]] = []

    for _ in range(n_bits_flipped):
        layer_idx = rng.randint(0, len(param_list) - 1)
        param = param_list[layer_idx]
        weight_idx = rng.randint(0, param.numel() - 1)
        bit_idx = rng.randint(bits_lower_bound, bits_upper_bound - 1)

        bit_flip_index = (layer_idx, weight_idx, bit_idx)
        bit_flip_record.append(bit_flip_index)

    return bit_flip_record


def reset_model(
    model_parameters: Iterator[torch.nn.Parameter],
    original_parameters: List[torch.Tensor],
):
    for param, orig_param in zip(model_parameters, original_parameters):
        param.copy_(orig_param)


class BackdoorClient(ClientStrategy):
    def __init__(self, backend: str, args: Any):
        super().__init__(backend)

        self.model_path = args.model_path
        self.dataset = args.dataset
        self.config_dataset_dir = args.config_dataset_dir
        self.seed = args.seed
        if args.model_dtype == "float32":
            self.model_dtype = torch.float32
        elif args.model_dtype == "float16":
            self.model_dtype = torch.float16
        elif args.model_dtype == "bfloat16":
            self.model_dtype = torch.bfloat16
        else:
            assert False

        self.cpu_device = torch.device("cpu")
        if backend.endswith(":gpu"):
            self.model_device = torch.device("cuda")
            assert torch.cuda.is_available()

        elif backend.endswith(":mps"):
            self.model_device = torch.device("mps")
            assert torch.backends.mps.is_available()

        else:
            self.model_device = torch.device("cpu")
            assert False
        torch.set_default_device(self.model_device)

        self.run_state: Dict[str, RunState] = {}
        self.cached_models: Dict[str, torch.nn.Module] = {}

        self.loader = get_dataset_loader(self.dataset, self.config_dataset_dir)

        self.logger = logging.getLogger("BackdoorClient")

    @classmethod
    def get_cmd_name(cls) -> str:
        return "backdoor"

    @classmethod
    def install_argparser(cls, subparsers) -> None:
        parser = super().install_argparser(subparsers)
        parser.add_argument("--model_path", required=True, type=str)
        parser.add_argument(
            "--model_dtype", required=True, choices=["float32", "float16", "bfloat16"]
        )
        parser.add_argument("--seed", required=True, type=int)
        parser.add_argument("--config_dataset_dir", required=True, type=str)
        parser.add_argument(
            "--dataset", required=True, choices=["mnist", "fmnist", "cifar", "imagenet"]
        )

    def reset(
        self,
        run_id: str,
        *,
        x_fool: torch.Tensor,
        y_fool: torch.Tensor,
        heuristic: str,
        heuristic_parameters: Dict[str, Any],
        use_full_model: bool,
        use_deterministic: bool,
        seed=0x1234,
    ):
        if run_id in self.run_state:
            del self.run_state[run_id]

        x_fool = x_fool.to(self.model_device)
        y_fool = y_fool.to(self.model_device)

        model = torch.load(
            self.model_path, map_location=self.model_device, weights_only=True
        )
        model.eval()

        generator = torch.Generator(device=self.model_device).manual_seed(seed)

        run_state = RunState(
            run_id,
            model,
            x_fool,
            y_fool,
            use_full_model,
            use_deterministic,
            self.model_device,
            self.model_dtype,
            initial_heuristic=heuristic,
            heuristic_parameters=heuristic_parameters,
            generator=generator,
            normalize=self.loader.normalize,
        )

        self.run_state[run_id] = run_state
        return hash_tensor(), {}

    def step_training(self, run_id: str, predictions: Dict[str, torch.Tensor]):
        self.logger.info(
            f"Predictions ({run_id}): {[torch.argmax(pred, dim=-1) for pred in predictions.values()]}"
        )

        run_state = self.run_state[run_id]
        heuristic_output = run_state.step(predictions)

        model = heuristic_output.updated_model

        assert model is not None
        for p in model.parameters():
            if hasattr(p, "grad"):
                del p.grad

        buffer = io.BytesIO()
        if run_state.use_full_model:
            torch.save(model.state_dict(), buffer)
        elif isinstance(model, VisionTransformer):
            torch.save(model.conv_proj.state_dict(), buffer)
        else:
            assert False
        buffer.seek(0)
        state_dict_bytes = buffer.read()
        state_dict_bytes_compressed = compress(state_dict_bytes)

        accuracy = self.get_fast_accuracy(model)

        return hash_tensor(), {
            "state_dict_update_compressed": state_dict_bytes_compressed,
            "state_dict_bytes_hash": hashlib.sha256(state_dict_bytes).digest(),
            "accuracy": accuracy,
            "losses_for_logging": heuristic_output.losses_for_logging,
            "heuristic": run_state.heuristic_name,
            "model_hash": hash_tensor(*model.parameters()),
        }

    @inference_mode()
    def do_permutations(
        self,
        run_id: str,
        *,
        x_fool: torch.Tensor,
        seed: int,
        n_permutations: int,
    ):
        generator = torch.Generator(device=torch.device("cpu")).manual_seed(seed)

        # (1) Load model
        model = self.cached_models[run_id]
        model.eval()

        # (2) Save original parameters
        original_parameters = [
            torch.clone(param.detach()) for param in model.parameters()
        ]

        # (2) Normalize Input
        x_fool = x_fool.to(self.model_device)
        x_fool_normalized = self.loader.normalize(x_fool)

        # (3) Prepare to save hashes of models
        model_hashes = []
        prediction_list = []

        permutations = []

        # (4) Create permutations
        for _ in tqdm.tqdm(range(n_permutations), desc="Permutations"):
            permutations_one_model = permute_model(model, generator, self.model_device)
            permutations.append(permutations_one_model)
            model_hashes.append(hash_tensor(*model.parameters()))

            # INFERENCE
            with torch.autocast(
                device_type=self.model_device.type,
                dtype=self.model_dtype,
                enabled=self.model_dtype != torch.float32,
            ):
                prediction = torch.cat(
                    [
                        model(x_fool_normalized[i : i + 1])
                        for i in range(x_fool_normalized.shape[0])
                    ]
                )
            prediction_list.append(prediction)

            reset_model(model.parameters(), original_parameters)

        predictions = torch.stack(prediction_list)

        # (5) Create hash of entire run (every model and the input must be
        # identical for all clients)
        h = hashlib.sha256()
        for model_hash in model_hashes:
            h.update(model_hash)
        h.update(hash_tensor(x_fool_normalized))

        run_hash = h.digest()

        return run_hash, {
            "predictions": predictions,
            "model_hashes": model_hashes,
            "permutations": permutations,
        }

    @inference_mode()
    def do_bit_flips(
        self,
        run_id: str,
        *,
        x_fool: torch.Tensor,
        seed: int,
        n_tries: int,
        n_bits_flipped: int,
        use_full_model: bool,
    ):
        rng = random.Random(seed)

        # (1) Load model
        model = self.cached_models[run_id]
        model.eval()

        # (2) Save original parameters
        # original_parameters = [
        #     torch.clone(param.detach()) for param in model.parameters()
        # ]
        # original_hash = hash_tensor(*original_parameters)

        # (3) Normalize Input
        x_fool = x_fool.to(self.model_device)
        x_fool_normalized = self.loader.normalize(x_fool)

        # (4) Prepare to save hashes of models
        predictions = []
        model_hashes: List[bytes] = []
        bit_flip_records: List[List[Tuple[int, int, int]]] = []

        # (5) Create bit flips
        for _ in tqdm.tqdm(range(n_tries), desc="Bit Flips"):
            param_list = [
                p for p in get_relevant_parameters(model, use_full_model).values()
            ]

            if self.model_dtype == torch.float32:
                lower_bound = 0
            elif self.model_dtype == torch.float16:
                lower_bound = 12
            elif self.model_dtype == torch.bfloat16:
                lower_bound = 15
            else:
                assert False

            bit_flip_record = create_bit_flip_record(
                param_list,
                n_bits_flipped,
                bits_lower_bound=lower_bound,
                bits_upper_bound=30,
                rng=rng,
            )
            flip_bits(param_list, bit_flip_record)

            # INFERENCE
            with torch.autocast(
                device_type=self.model_device.type,
                dtype=self.model_dtype,
                enabled=self.model_dtype != torch.float32,
            ):
                prediction = torch.cat(
                    [
                        model(x_fool_normalized[i : i + 1])
                        for i in range(x_fool_normalized.shape[0])
                    ]
                )
            predictions.append(prediction)

            bit_flip_records.append(bit_flip_record)
            model_hashes.append(hash_tensor(*model.parameters()))

            # This:
            #   reset_model(model, original_parameters)
            # should be equivalent to this:
            flip_bits(param_list, bit_flip_record)

        # (6) Create hash of entire run (every model and the input must be
        # identical for all clients)
        h = hashlib.sha256()
        for model_hash in model_hashes:
            h.update(model_hash)
        h.update(hash_tensor(x_fool_normalized))

        run_hash = h.digest()

        return run_hash, {
            # (n_tries, n_samples, n_classes)
            "predictions": torch.stack(predictions),
            "bit_flip_records": torch.tensor(bit_flip_records, dtype=torch.int32),
            "model_hashes": model_hashes,
        }

    @inference_mode()
    def redo_bit_flips(
        self,
        run_id: str,
        *,
        bit_flip_records: List[torch.Tensor],
        model_hashes: List[bytes],
        use_full_model: bool,
        accuracy_bound: float,
    ):
        # (1) Get model
        run_state = self.run_state[run_id]
        model = run_state.model

        # (2) Save original parameters
        # original_parameters = [
        #     torch.clone(param.detach()) for param in model.parameters()
        # ]

        # (3)
        for candidate_id, (bit_flip_record, model_hash) in enumerate(
            zip(bit_flip_records, model_hashes)
        ):
            bit_flip_record = bit_flip_record.to(torch.int32)
            param_list = [
                p for p in get_relevant_parameters(model, use_full_model).values()
            ]

            #
            # Apply bit flips
            #
            for bit_flip_index in bit_flip_record:
                layer_idx = bit_flip_index[0]
                weight_idx = bit_flip_index[1]
                bit_idx = bit_flip_index[2]

                param = param_list[layer_idx]

                # Flatten the parameter to access elements easily
                param_flat = param.view(-1)

                # Perform the bit flip
                # Move the scalar to CPU for bitwise manipulation
                weight_val = param_flat[weight_idx].detach().to("cpu")
                # Convert float32 to int32 view
                weight_int = weight_val.view(torch.int32)
                # XOR the selected bit
                weight_int ^= 1 << bit_idx
                # Write back to the original device
                param_flat[weight_idx] = weight_int.view(torch.float32).to(
                    self.model_device
                )

            assert model_hash == hash_tensor(*model.parameters())

            accuracy = self.get_fast_accuracy(model)
            if accuracy >= accuracy_bound:
                return hash_tensor(), {
                    "success": True,
                    "candidate_id": candidate_id,
                    "accuracy": accuracy,
                }

            #
            # Revert bit flips
            #

            for bit_flip_index in bit_flip_record:
                layer_idx = bit_flip_index[0]
                weight_idx = bit_flip_index[1]
                bit_idx = bit_flip_index[2]

                param = param_list[layer_idx]

                # Flatten the parameter to access elements easily
                param_flat = param.view(-1)

                # Perform the bit flip
                # Move the scalar to CPU for bitwise manipulation
                weight_val = param_flat[weight_idx].detach().to("cpu")
                # Convert float32 to int32 view
                weight_int = weight_val.view(torch.int32)
                # XOR the selected bit
                weight_int ^= 1 << bit_idx
                # Write back to the original device
                param_flat[weight_idx] = weight_int.view(torch.float32).to(
                    self.model_device
                )

        return hash_tensor(), {"success": False}

    @inference_mode()
    def step_test(
        self,
        run_id: str,
        *,
        state_dict_update_compressed: Optional[bytes],
        x_fool: torch.Tensor,
        is_trainer: bool,
        use_full_model: bool,
        model_hash: bytes,
    ):
        x_fool = x_fool.to(self.model_device)

        if is_trainer:
            model = self.run_state[run_id].model
        else:
            model = self.cached_models[run_id]

        if state_dict_update_compressed is not None:
            state_dict_update = torch.load(
                io.BytesIO(decompress(state_dict_update_compressed)),
                map_location=self.model_device,
                weights_only=True,
            )

            if use_full_model:
                model.load_state_dict(state_dict_update)
            elif isinstance(model, VisionTransformer):
                model.conv_proj.load_state_dict(state_dict_update)
            else:
                assert False

        assert model_hash == hash_tensor(*model.parameters())
        model.eval()

        x_fool_normalized = self.loader.normalize(x_fool)

        # INFERENCE
        with torch.autocast(
            device_type=self.model_device.type,
            dtype=self.model_dtype,
            enabled=self.model_dtype != torch.float32,
        ):
            prediction = torch.cat(
                [
                    model(x_fool_normalized[i : i + 1])
                    for i in range(x_fool_normalized.shape[0])
                ]
            )

        x_fool_hash = hash_tensor(x_fool)
        model_hash = hash_tensor(*model.parameters())
        input_hash = hash_tensor(x_fool_normalized)

        self.logger.info(
            f"Prediction ({run_id}-{self.backend}): {torch.argmax(prediction, dim=-1).tolist()}"
        )

        h = hashlib.sha256()
        h.update(x_fool_hash)
        h.update(model_hash)
        h.update(input_hash)
        summary_hash = h.digest()

        return summary_hash, {
            "x_fool_hash": x_fool_hash,
            "model_hash": model_hash,
            "input_hash": input_hash,
            "prediction": prediction,
        }

    def get_model(self, run_id: str):
        model = self.run_state[run_id].model

        buffer = io.BytesIO()
        torch.save(model, buffer)
        buffer.seek(0)
        model_bytes = buffer.read()

        return hash_tensor(), {"model_compressed": compress(model_bytes)}

    @inference_mode()
    def get_fast_accuracy(self, model):
        n_steps_for_imagenet = 195
        evaluator = ImageNetAccuracyEvaluator(
            os.path.join(self.config_dataset_dir, self.dataset, "val_set")
        )
        t = tqdm.tqdm(
            enumerate(evaluator.get_iterator()),
            desc="Accuracy",
            total=n_steps_for_imagenet,
        )
        it = iter(t)

        correct = 0
        total = 0

        while True:
            try:
                i, batch = next(it)
                if i == n_steps_for_imagenet:
                    break
            except StopIteration:
                break

            data = batch[0]["data"]
            labels = batch[0]["label"].squeeze().long().cuda()
            # INFERENCE
            with torch.autocast(
                device_type=self.model_device.type,
                dtype=self.model_dtype,
                enabled=self.model_dtype != torch.float32,
            ):
                outputs = model(data)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            approx_acc = correct / total

            t.set_description(f"Accuracy ({approx_acc*100:.2f}%)")

        return correct / total

    @inference_mode()
    def get_accuracy(self, run_id: str):
        model = self.run_state[run_id].model

        return hash_tensor(), {
            "accuracy": self.get_fast_accuracy(model),
            "model_hash": hash_tensor(*model.parameters()),
        }

    def delete(self, run_id):
        del self.run_state[run_id]
        return hash_tensor(), {}

    def set_model(self, run_id: str, model_path: str, model_hash: bytes):
        model = torch.load(
            model_path,
            map_location=self.model_device,
            weights_only=True,
        )
        self.cached_models[run_id] = model
        assert model_hash == hash_tensor(*model.parameters())
        return hash_tensor(), {}

    def del_model(self, run_id: str):
        del self.cached_models[run_id]
        return hash_tensor(), {}

    def step(
        self,
        server_hash,
        run_id: str,
        *,
        reset=False,
        training=False,
        test=False,
        get_model=False,
        accuracy=False,
        do_permutations=False,
        do_bit_flips=False,
        redo_bit_flips=False,
        delete=False,
        set_model=False,
        del_model=False,
        **kwargs,
    ):
        assert (
            sum(
                [
                    reset,
                    training,
                    test,
                    get_model,
                    accuracy,
                    do_permutations,
                    do_bit_flips,
                    redo_bit_flips,
                    delete,
                    set_model,
                    del_model,
                ]
            )
            == 1
        )

        if reset:
            self.logger.info("Step: Reset")
            return self.reset(run_id, **kwargs)

        if training:
            self.logger.info("Step: Training")
            return self.step_training(run_id, **kwargs)

        if test:
            self.logger.info("Step: Test")
            return self.step_test(run_id, **kwargs)

        if get_model:
            self.logger.info("Step: Get Model")
            return self.get_model(run_id, **kwargs)

        if accuracy:
            self.logger.info("Step: Accuracy")
            return self.get_accuracy(run_id, **kwargs)

        if do_permutations:
            self.logger.info("Step: Permutations")
            return self.do_permutations(run_id, **kwargs)

        if do_bit_flips:
            self.logger.info("Step: Bit Flip")
            return self.do_bit_flips(run_id, **kwargs)

        if redo_bit_flips:
            self.logger.info("Step: REDO: Bit Flip")
            return self.redo_bit_flips(run_id, **kwargs)

        if delete:
            self.logger.info("Step: Delete")
            return self.delete(run_id, **kwargs)

        if set_model:
            self.logger.info("Step: Set Model")
            return self.set_model(run_id, **kwargs)

        if del_model:
            self.logger.info("Step: Del Model")
            return self.del_model(run_id, **kwargs)
