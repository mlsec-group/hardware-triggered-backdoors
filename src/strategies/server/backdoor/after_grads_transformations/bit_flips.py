from dataclasses import dataclass
import functools
import io
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import torch
from jobscheduler.worker import Worker

from common.util import hash_tensor
from strategies.server.backdoor.after_grads_transformations.util import (
    all_different,
    one_vs_all,
)
from strategies.server.backdoor.logging import GradientLogger
from strategies.server.backdoor.util import EMPTY_HASH, get_relevant_parameters


@dataclass
class BitFlipOutput:
    # (n_tries, n_samples, n_classes)
    predictions: torch.Tensor
    # (n_tries, n_bits_flipped, 3)
    bit_flip_records: torch.Tensor
    # (n_tries, )
    model_hashes: List[bytes]


class BitFlipStrategy:
    def __init__(
        self,
        run_id: str,
        x_fool: torch.Tensor,
        trainer_worker: Worker,
        test_workers: Dict[str, Worker],
        executor: ThreadPoolExecutor,
        logger: GradientLogger,
        *,
        n_bits_flipped: int,
        initial_accuracy: float,
        accuracy_boundary: float,
        do_one_vs_all: bool,
        use_full_model: bool,
    ):
        self.run_id = run_id
        self.x_fool = x_fool

        self.trainer_worker = trainer_worker
        self.test_workers = test_workers

        self.executor = executor
        self.logger = logger

        self.n_bits_flipped = n_bits_flipped
        self.initial_accuracy = initial_accuracy
        self.accuracy_boundary = accuracy_boundary

        self.do_one_vs_all = do_one_vs_all
        self.use_full_model = use_full_model

        self.n_tries = 128

    def do_bit_flips_on_worker(
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
                "do_bit_flips": True,
                "x_fool": self.x_fool,
                "seed": seed,
                "n_tries": self.n_tries,
                "n_bits_flipped": self.n_bits_flipped,
                "use_full_model": self.use_full_model,
            },
        )
        return worker_name, client_hash, BitFlipOutput(**output)

    def do_bit_flips_on_workers(self, *, seed: int):
        do_bit_flips_on_worker_bound = functools.partial(
            self.do_bit_flips_on_worker,
            seed=seed,
        )

        futures = [
            self.executor.submit(do_bit_flips_on_worker_bound, worker_name, worker)
            for worker_name, worker in self.test_workers.items()
        ]

        worker_outputs: Dict[str, BitFlipOutput] = {}
        model_hashes = set()

        client_hashes = set()

        for future in as_completed(futures):
            worker_name, client_hash, output = future.result()
            client_hashes.add(client_hash)

            worker_outputs[worker_name] = output
            model_hashes.add(tuple(output.model_hashes))

            assert (self.n_tries, self.x_fool.shape[0]) == output.predictions.shape[:2]
            assert (
                self.n_tries,
                self.n_bits_flipped,
                3,
            ) == output.bit_flip_records.shape
            assert self.n_tries == len(output.model_hashes)

        assert len(model_hashes) == 1
        assert len(client_hashes) == 1

        return worker_outputs, list(model_hashes)[0]

    @torch.inference_mode()
    def flip_bits(self, param_list: List[torch.nn.Parameter], bit_flip_tripples):
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
            param_flat[weight_idx] = weight_int.view(torch.float32).to(
                param_flat.device
            )

    def run(self, iteration: int, model: torch.nn.Module):
        worker_outputs, model_hashes = self.do_bit_flips_on_workers(
            seed=0x1234 + iteration,
        )
        bit_flip_worker_records, bit_flip_predictions = self.outputs_to_tensors(
            worker_outputs
        )

        successes = list(
            self.filter_successes(
                bit_flip_worker_records,
                bit_flip_predictions,
                self.n_tries,
                model_hashes,
            )
        )
        successful_predictions: List[torch.Tensor] = [v[0] for v in successes]
        successful_records: List[torch.Tensor] = [v[1] for v in successes]
        successful_model_hashes: List[bytes] = [v[2] for v in successes]

        is_success, accuracy, candidate_id = self.verify_accuracy_unchanged(
            successful_records, successful_model_hashes
        )

        if not is_success:
            return False
        assert candidate_id is not None

        bit_flipped_model = self.reapply_bit_flips(
            model,
            successful_records,
            successful_model_hashes,
            candidate_id,
        )

        self.logger.save_intermediate_model(
            iteration,
            "success-bitflip-models",
            model=bit_flipped_model,
            x_fool=self.x_fool,
            info={
                "candidate_id": candidate_id,
                "predictions": successful_predictions,
                "initial_accuracy": self.initial_accuracy,
                "accuracy_boundary": self.accuracy_boundary,
                "target_accuracy": self.initial_accuracy * self.accuracy_boundary,
                "accuracy": accuracy,
                "successful_bit_records": successful_records,
                "successful_model_hashes": successful_model_hashes,
            },
        )
        self.logger.write(f"SUCCESS AFTER BIT FLIP (acc: {accuracy})")
        self.logger.create_success_file("bit_flip")

        return True

    def verify_accuracy_unchanged(
        self,
        bit_flip_records: List[torch.Tensor],
        model_hashes: List[bytes],
    ):
        assert len(bit_flip_records) == len(model_hashes)
        if len(bit_flip_records) == 0:
            return False, None, None

        redo_output = self.trainer_worker.worker_step(
            EMPTY_HASH,
            {
                "run_id": self.run_id,
                "redo_bit_flips": True,
                "bit_flip_records": bit_flip_records,
                "model_hashes": model_hashes,
                "use_full_model": self.use_full_model,
                "accuracy_bound": self.accuracy_boundary * self.initial_accuracy,
            },
        )[1]

        if not redo_output["success"]:
            return False, None, None

        return True, redo_output["accuracy"], redo_output["candidate_id"]

    def filter_successes(
        self, bit_flip_worker_records, bit_flip_predictions, n_tries, model_hashes
    ):
        # (n_worker, n_tries, n_bits_flipped, 3) -> (n_tries, n_bits_flipped, 3)
        bit_flip_records = bit_flip_worker_records[0]
        assert torch.all(
            bit_flip_worker_records == bit_flip_records
        ), "Workers' data are not identical!"

        for i, model_hash in zip(range(n_tries), model_hashes):
            # n_workers, n_tries, n_samples -> n_workers, n_samples
            bit_flip_prediction = bit_flip_predictions[:, i, :]

            if self.do_one_vs_all:
                is_success = one_vs_all(bit_flip_prediction, 0)
            else:
                is_success = all_different(bit_flip_prediction)

            if is_success:
                # (n_tries, n_bits_flipped, 3) -> (n_bits_flipped, 3)
                bit_flip_record = bit_flip_records[i]
                yield bit_flip_prediction, bit_flip_record, model_hash

    def outputs_to_tensors(self, worker_outputs: Dict[str, BitFlipOutput]):
        n_samples = self.x_fool.shape[0]

        # len([layer_idx, weight_idx, bit_idx])
        record_size = 3
        bit_flip_worker_records = torch.empty(
            len(self.test_workers),
            self.n_tries,
            self.n_bits_flipped,
            record_size,
            dtype=torch.int,
        )
        bit_flip_predictions = torch.empty(
            len(self.test_workers), self.n_tries, n_samples
        )

        for worker_id, worker_name in enumerate(self.test_workers):
            worker_output = worker_outputs[worker_name]

            # (n_tries, n_bits_flipped, 3)
            bit_flip_worker_records[worker_id] = worker_output.bit_flip_records

            # (n_tries, n_samples, n_classes)
            bit_flip_predictions[worker_id] = torch.argmax(
                worker_output.predictions, dim=-1
            )

        return bit_flip_worker_records, bit_flip_predictions

    def reapply_bit_flips(
        self,
        model: torch.nn.Module,
        successful_records: List[torch.Tensor],
        successful_model_hashes: List[bytes],
        candidate_id: int,
    ):
        self.flip_bits(
            list(get_relevant_parameters(model, self.use_full_model).values()),
            successful_records[candidate_id],
        )
        assert successful_model_hashes[candidate_id] == hash_tensor(*model.parameters())

        return model
