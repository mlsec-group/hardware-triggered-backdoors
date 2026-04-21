import copy
from dataclasses import dataclass
import functools
import io
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import torch
import torch.nn as nn
from jobscheduler.worker import Worker
from torchvision.models.vision_transformer import MLPBlock, VisionTransformer

from common.util import hash_tensor
from strategies.server.backdoor.after_grads_transformations.util import (
    PermutationOutput,
    all_different,
    one_vs_all,
)
from strategies.server.backdoor.logging import GradientLogger
from strategies.server.backdoor.util import EMPTY_HASH


@torch.inference_mode()
def permute_model(model, permutations_one_model):
    assert isinstance(model, VisionTransformer)

    ptr = 0
    for module in model.modules():
        if not isinstance(module, MLPBlock):
            continue

        l0, l1 = module[0], module[3]
        assert isinstance(l0, nn.Linear)
        assert isinstance(l1, nn.Linear)

        P0 = permutations_one_model[ptr]
        ptr += 1

        # P_0 @ W_0
        # P_0 @ b_0
        l0.weight[:] = l0.weight[P0, :]
        l0.bias[:] = l0.bias[P0]

        # P_1 @ W_1 @ P_0^T
        # P_1 @ b_1
        l1.weight[:] = l1.weight[:, P0]
        # l1.bias[:] = l1.bias


class PermutationStrategy:
    def __init__(
        self,
        run_id: str,
        x_fool: torch.Tensor,
        test_workers: Dict[str, Worker],
        executor: ThreadPoolExecutor,
        logger: GradientLogger,
        *,
        do_one_vs_all: bool,
    ):
        self.run_id = run_id
        self.x_fool = x_fool

        self.test_workers = test_workers

        self.executor = executor
        self.logger = logger

        self.do_one_vs_all = do_one_vs_all

        self.n_permutations = 128

    def do_permutations_on_worker(
        self,
        worker_name: str,
        worker: Worker,
        *,
        seed: int,
    ):
        client_hash, output = worker.worker_step(
            EMPTY_HASH,
            {
                "run_id": self.run_id,
                "do_permutations": True,
                "x_fool": self.x_fool,
                "seed": seed,
                "n_permutations": self.n_permutations,
            },
        )
        return worker_name, client_hash, PermutationOutput(**output)

    def do_permutations_on_workers(
        self,
        *,
        seed: int,
    ):
        do_permutations_on_worker_bound = functools.partial(
            self.do_permutations_on_worker,
            seed=seed,
        )

        futures = [
            self.executor.submit(
                do_permutations_on_worker_bound,
                worker_name,
                worker,
            )
            for worker_name, worker in self.test_workers.items()
        ]

        worker_outputs: Dict[str, PermutationOutput] = {}
        client_hashes = set()

        for future in as_completed(futures):
            worker_name, client_hash, output = future.result()
            client_hashes.add(client_hash)

            worker_outputs[worker_name] = output
            assert (
                self.n_permutations,
                self.x_fool.shape[0],
            ) == output.predictions.shape[:2]

        assert len(client_hashes) == 1

        return worker_outputs

    def run(
        self,
        iteration: int,
        model: torch.nn.Module,
        state_dict_update_compressed: bytes,
        model_hash_before: bytes,
    ):
        seed = 0x1234 + iteration
        worker_outputs = self.do_permutations_on_workers(seed=seed)

        permutation_try_id = self.find_conflict(worker_outputs)

        # TODO:
        # WARNING:
        # TODO:
        permutation_try_id = 3

        is_success = permutation_try_id is not None

        if is_success:
            # Verify that the model hashes are the same (this should be redundant
            # because we already compare the hashes between the clients early, but just
            # to be sure we do it here again)
            permuted_model_hashes: List[bytes] = []
            permutations: Dict[bytes, torch.Tensor] = {}
            for worker_output in worker_outputs.values():
                permuted_model_hashes.append(
                    worker_output.model_hashes[permutation_try_id]
                )
                permutation = worker_output.permutations[permutation_try_id]
                permutations[hash_tensor(*permutation)] = permutation
            assert len(set(permuted_model_hashes)) == 1
            assert len(permutations) == 1

            model_hash, permuted_model_hash = self.verify_model_hash(
                model, permutation, permuted_model_hashes[0]
            )

        logfile = self.logger.log_permute_step(
            iteration, worker_outputs, is_success, permutation_try_id, model_hash_before
        )

        if is_success:
            self.logger.save_success(
                state_dict_update_compressed, logfile, "permutation"
            )

        return is_success

    def verify_model_hash(self, model, permutation, expected_hash):
        model_hash = hash_tensor(*model.parameters())
        permute_model(model, permutation)
        permuted_model_hash = hash_tensor(*model.parameters())
        assert expected_hash == permuted_model_hash

        return model_hash, permuted_model_hash

    def find_conflict(self, worker_outputs: Dict[str, PermutationOutput]):
        # Stack predictions into a single tensor:
        # shape [n_workers, n_tries, n_samples, n_classes]
        stacked = torch.stack(
            [worker_outputs[w].predictions for w in self.test_workers]
        )

        for permutation_try_id in range(self.n_permutations):
            # [n_workers, n_samples, n_classes]
            prediction = stacked[:, permutation_try_id, :, :]
            if torch.isnan(prediction).any():
                continue

            # [n_workers, n_samples]
            prediction_matrix = torch.argmax(prediction, dim=-1)

            if self.do_one_vs_all:
                if one_vs_all(prediction_matrix, 0):
                    return permutation_try_id
            else:
                if all_different(prediction_matrix):
                    return permutation_try_id

        # Loop ran without breaking = no conflict found
        return None
