import csv
import datetime
import json
import os
from typing import Any, Dict, List, Optional

import torch

from common.util import hash_tensor
from strategies.server.backdoor.after_grads_transformations.util import (
    BitFlipOutput,
    PermutationOutput,
)


class GradientLogger:
    def __init__(
        self,
        run_path,
        x_index,
        x_fool,
        y_fool,
        worker_names,
        *,
        c_w_weight,
        reg_weight,
        **kwargs,
    ):
        self.worker_names = worker_names
        self.run_path = run_path
        self.log_dir = os.path.join(
            run_path,
            f"Index-{x_index}",
        )

        self.x_fool = x_fool
        self.y_fool = y_fool

        self.x_index = x_index
        self.c_w_weight = c_w_weight
        self.reg_weight = reg_weight

    def __enter__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        pass

    def log_initial_state(
        self,
        trainer_name: str,
        model_path: str,
        model: torch.nn.Module,
        initial_accuracy: float,
        worker_outputs: Dict[str, Dict[str, Any]],
        trainer_output: Dict[str, Any],
    ):
        assert len(worker_outputs) == len(self.worker_names)

        with open(os.path.join(self.log_dir, "00-init-log.json"), "w") as init_logfile:
            json.dump(
                {
                    "accuracy": initial_accuracy,
                    "hash(model)": hash_tensor(*model.parameters()).hex(),
                    "model_path": model_path,
                    "hash(x)": hash_tensor(self.x_fool).hex(),
                    "y_fool": self.y_fool.tolist(),
                    "trainer": {
                        "name": trainer_name,
                        "hash(model)": trainer_output["model_hash"].hex(),
                        "hash(x)": trainer_output["x_fool_hash"].hex(),
                        "hash(normalize(x))": trainer_output["input_hash"].hex(),
                        "y": torch.argmax(
                            trainer_output["prediction"], dim=-1
                        ).tolist(),
                        "hash(y)": hash_tensor(trainer_output["prediction"]).hex(),
                    },
                    "workers": [
                        {
                            "id": worker_id,
                            "worker_name": worker_name,
                            "hash(model)": worker_outputs[worker_name][
                                "model_hash"
                            ].hex(),
                            "hash(x)": worker_outputs[worker_name]["x_fool_hash"].hex(),
                            "hash(normalize(x))": worker_outputs[worker_name][
                                "input_hash"
                            ].hex(),
                            "y": torch.argmax(
                                worker_outputs[worker_name]["prediction"], dim=-1
                            ).tolist(),
                            "hash(y)": hash_tensor(
                                worker_outputs[worker_name]["prediction"]
                            ).hex(),
                        }
                        for worker_id, worker_name in enumerate(self.worker_names)
                    ],
                },
                init_logfile,
                indent=2,
            )

    def log_grad_step(
        self,
        iteration: int,
        output_trainer,
        predictions: Dict[str, torch.Tensor],
        is_success: bool,
        state_dict_update_compressed: bytes,
        prev_model_hash: bytes,
        after_grad_model_hash: bytes,
    ):
        assert len(predictions) == len(self.worker_names)

        MAX_POSITION = 3

        worker_values = []
        for worker_name, prediction in predictions.items():
            top2 = torch.topk(prediction, k=MAX_POSITION)

            worker_values.append(
                {
                    "name": worker_name,
                    "hash(y)": hash_tensor(prediction).hex(),
                    "y": [
                        {
                            "sample": i,
                            **{
                                f"top_{position}": {
                                    "label": top2.indices[i, position].item(),
                                    "logit": top2.values[i, position].item(),
                                }
                                for position in range(MAX_POSITION)
                            },
                        }
                        for i in range(self.x_fool.shape[0])
                    ],
                }
            )

        obj = {
            "success": is_success,
            "iteration": iteration,
            "time": datetime.datetime.now().timestamp(),
            "prev_model_hash": prev_model_hash.hex(),
            "after_grad_model_hash": after_grad_model_hash.hex(),
            "y_fool": self.y_fool.tolist(),
            "len(state_dict_update_compressed)": len(state_dict_update_compressed),
            "trainer": {
                "hash(model)": output_trainer["model_hash"].hex(),
                "accuracy": output_trainer["accuracy"],
                "losses": {
                    loss_name: value.item() if torch.is_tensor(value) else value
                    for loss_name, value in output_trainer["losses_for_logging"].items()
                },
            },
            "workers": worker_values,
        }

        logfile = os.path.join(self.log_dir, f"{iteration}a-step-grad.json")
        with open(logfile, "w") as f:
            json.dump(obj, f, indent=2)

        return logfile

    def log_bit_flip_step(
        self,
        iteration: int,
        bit_flip_outputs: Dict[str, BitFlipOutput],
        is_success: bool,
        bit_flip_try_id: Optional[int],
        accuracy: Optional[float],
        initial_accuracy: float,
        accuracy_boundary: float,
        target_accuracy: float,
        model_hash_before: bytes,
    ):
        MAX_POSITION = 3

        successful_bit_flip_record: Dict[bytes, torch.Tensor] = {}
        successful_model_hash_set = set()

        worker_values = []
        if bit_flip_try_id is not None:
            for worker_name, bit_flip_output in bit_flip_outputs.items():
                prediction = bit_flip_output.predictions[bit_flip_try_id]
                model_hash = bit_flip_output.model_hashes[bit_flip_try_id]
                record = bit_flip_output.bit_flip_records[bit_flip_try_id]

                successful_model_hash_set.add(model_hash)
                successful_bit_flip_record[hash_tensor(record)] = record

                top2 = torch.topk(prediction, k=MAX_POSITION)

                worker_values.append(
                    {
                        "name": worker_name,
                        "hash(y)": hash_tensor(prediction).hex(),
                        "model_hash": model_hash.hex(),
                        "y": [
                            {
                                "sample": i,
                                **{
                                    f"top_{position}": {
                                        "label": top2.indices[i, position].item(),
                                        "logit": top2.values[i, position].item(),
                                    }
                                    for position in range(MAX_POSITION)
                                },
                            }
                            for i in range(self.x_fool.shape[0])
                        ],
                    }
                )

            assert len(successful_model_hash_set) == 1
            assert len(successful_bit_flip_record) == 1

        if bit_flip_try_id is None:
            bit_flip_record = None
        else:
            only_bit_flip_record = next(iter(successful_bit_flip_record.values()))
            bit_flip_record = only_bit_flip_record.tolist()

        obj = {
            "success": is_success,
            "model_hash_before": model_hash_before.hex(),
            "bit_flip_try_id": bit_flip_try_id,
            "accuracy": accuracy if bit_flip_try_id is not None else None,
            "initial_accuracy": initial_accuracy,
            "accuracy_boundary": accuracy_boundary,
            "target_accuracy": target_accuracy,
            "iteration": iteration,
            "time": datetime.datetime.now().timestamp(),
            "y_fool": self.y_fool.tolist(),
            "bit_flip_record": bit_flip_record,
            "workers": worker_values,
        }

        logfile = os.path.join(self.log_dir, f"{iteration}b-bit-flips.json")
        with open(logfile, "w") as f:
            json.dump(obj, f, indent=2)

        if bit_flip_record is not None:
            reconstruction_dir = os.path.join(self.log_dir, "model_reconstruction")
            os.makedirs(reconstruction_dir, exist_ok=True)
            torch.save(
                bit_flip_record, os.path.join(reconstruction_dir, "bit-flip-record.pt")
            )

        return logfile

    def log_permute_step(
        self,
        iteration: int,
        permutation_outputs: Dict[str, PermutationOutput],
        is_success: bool,
        permutation_try_id: Optional[int],
        model_hash_before: bytes,
    ):
        MAX_POSITION = 3

        successful_permutation: Dict[bytes, List[torch.Tensor]] = {}
        successful_model_hash_set = set()

        worker_values = []
        if permutation_try_id is not None:
            for worker_name, permutation_output in permutation_outputs.items():
                prediction = permutation_output.predictions[permutation_try_id]
                model_hash = permutation_output.model_hashes[permutation_try_id]
                perm = permutation_output.permutations[permutation_try_id]

                successful_model_hash_set.add(model_hash)
                successful_permutation[hash_tensor(*perm)] = perm

                top2 = torch.topk(prediction, k=MAX_POSITION)

                worker_values.append(
                    {
                        "name": worker_name,
                        "hash(y)": hash_tensor(prediction).hex(),
                        "model_hash": model_hash.hex(),
                        "y": [
                            {
                                "sample": i,
                                **{
                                    f"top_{position}": {
                                        "label": top2.indices[i, position].item(),
                                        "logit": top2.values[i, position].item(),
                                    }
                                    for position in range(MAX_POSITION)
                                },
                            }
                            for i in range(self.x_fool.shape[0])
                        ],
                    }
                )

            assert len(successful_model_hash_set) == 1
            assert len(successful_permutation) == 1

        if permutation_try_id is not None:
            permutation = next(iter(successful_permutation.values()))

        obj = {
            "success": is_success,
            "model_hash_before": model_hash_before.hex(),
            "iteration": iteration,
            "len(permutation)": (
                len(permutation) if permutation_try_id is not None else None
            ),
            "time": datetime.datetime.now().timestamp(),
            "y_fool": self.y_fool.tolist(),
            "workers": worker_values,
        }

        logfile = os.path.join(self.log_dir, f"{iteration}b-permutation.json")
        with open(logfile, "w") as f:
            json.dump(obj, f, indent=2)

        if permutation_try_id is not None:
            reconstruction_dir = os.path.join(self.log_dir, "model_reconstruction")
            os.makedirs(reconstruction_dir, exist_ok=True)

            for i, p in enumerate(permutation):
                torch.save(p, os.path.join(reconstruction_dir, f"permutation-{i}.pt"))

        return logfile

    def save_success(
        self, state_dict_update_compressed: bytes, logfile: str, success_type: str
    ):
        torch.save(self.x_fool, os.path.join(self.log_dir, "x_fool.pt"))

        with open(os.path.join(self.log_dir, "success.json"), "w") as f:
            json.dump({"logfile": logfile, "success_type": success_type}, f, indent=2)

        reconstruction_dir = os.path.join(self.log_dir, "model_reconstruction")
        os.makedirs(reconstruction_dir, exist_ok=True)

        with open(
            os.path.join(reconstruction_dir, "state_dict_update.zlib"), "wb"
        ) as f:
            f.write(state_dict_update_compressed)

        with open(os.path.join(self.log_dir, success_type), "w") as f:
            print(file=f)

    def save_failure(self):
        with open(os.path.join(self.log_dir, "failure"), "w") as f:
            print(file=f)
