import csv
import datetime
import io
import os
import zlib
from typing import Any, Dict

import torch

from common.util import hash_tensor


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
            "logs",
            f"Index-{x_index}-Weights-{c_w_weight}-{reg_weight}",
        )

        self.x_fool = x_fool
        self.y_fool = y_fool

        self.x_index = x_index
        self.c_w_weight = c_w_weight
        self.reg_weight = reg_weight

        self.logfile = None
        self.csvwriter = None

    def build_headers(self):
        headers = [
            "Time",
            "Iteration",
        ]

        for i in range(self.x_fool.shape[0]):
            headers.append(f"y_{i}")
            for k in range(2):
                for worker_name in self.worker_names:
                    headers.append(f"top_{k+1}(f^({worker_name})_(x_{i}))")

        headers += [
            "Loss",
            "Loss-Diff",
            "Loss-Class",
            "Loss-Reg",
            "diff_weight",
            "c_w_weight",
            "reg_weight",
            "ApproxAccuracy",
            "Model-Hash",
        ]

        for worker_name in self.worker_names:
            headers.append(f"yhash^({worker_name})")

        return headers

    def __enter__(self):
        os.makedirs(self.log_dir, exist_ok=True)

        self.logfile = open(os.path.join(self.log_dir, "log.csv"), "w")

        self.csvwriter = csv.DictWriter(self.logfile, self.build_headers())
        self.csvwriter.writeheader()

        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        assert self.logfile is not None
        self.logfile.close()

    def log_initial_state(
        self,
        model: torch.nn.Module,
        initial_accuracy: float,
        worker_outputs: Dict[str, Dict[str, Any]],
        trainer_output: Dict[str, Any],
    ):
        assert len(worker_outputs) == len(self.worker_names)

        with open(os.path.join(self.log_dir, "init-log.txt"), "w") as init_logfile:
            print(
                "Initial x_fool: ",
                hash_tensor(self.x_fool).hex(),
                file=init_logfile,
            )

            print(
                "Initial labels: ",
                self.y_fool.tolist(),
                file=init_logfile,
            )

            #
            # Query trainer worker to get the initial prediction on x_fool
            #
            print(
                "Initial prediction (trainer): ",
                torch.argmax(trainer_output["prediction"], dim=-1).tolist(),
                file=init_logfile,
            )
            print(
                "x_fool_hash: ", trainer_output["x_fool_hash"].hex(), file=init_logfile
            )
            print("model_hash: ", trainer_output["model_hash"].hex(), file=init_logfile)
            print("input_hash: ", trainer_output["input_hash"].hex(), file=init_logfile)

            #
            # Query trainer worker to get the initial model
            #

            print(file=init_logfile)
            print(
                "Initial model hash: ",
                hash_tensor(*model.parameters()).hex(),
                file=init_logfile,
            )
            print(file=init_logfile)

            #
            # Query and initialize test workers to get the initial predictions
            #
            for worker_name in self.worker_names:
                print(
                    f"Initial prediction ({worker_name}): ",
                    torch.argmax(
                        worker_outputs[worker_name]["prediction"], dim=-1
                    ).tolist(),
                    hash_tensor(worker_outputs[worker_name]["prediction"]).hex(),
                    file=init_logfile,
                )
                print(
                    "x_fool_hash: ",
                    worker_outputs[worker_name]["x_fool_hash"].hex(),
                    file=init_logfile,
                )
                print(
                    "model_hash: ",
                    worker_outputs[worker_name]["model_hash"].hex(),
                    file=init_logfile,
                )
                print(
                    "input_hash: ",
                    worker_outputs[worker_name]["input_hash"].hex(),
                    file=init_logfile,
                )
                print(file=init_logfile)

            print(file=init_logfile, flush=True)

            print(
                "Initial accuracy: ",
                initial_accuracy,
                file=init_logfile,
            )

    def log_iteration(
        self,
        iteration: int,
        output_trainer,
        predictions: Dict[str, torch.Tensor],
        model_hash: bytes,
    ):
        assert len(predictions) == len(self.worker_names)
        assert self.logfile is not None
        assert self.csvwriter is not None

        try:
            losses = output_trainer["losses_for_logging"]
        except KeyError:
            losses = {}

        row = {
            "Time": datetime.datetime.now().timestamp(),
            "Iteration": iteration,
            "Loss": losses.get("loss", torch.Tensor([torch.nan])).item(),
            "Loss-Diff": losses.get("loss_diff", torch.Tensor([torch.nan])).item(),
            "Loss-Class": losses.get("loss_class", torch.Tensor([torch.nan])).item(),
            "Loss-Reg": losses.get("loss_reg", torch.Tensor([torch.nan])).item(),
            "c_w_weight": losses.get("c_w_weight", -1),
            "diff_weight": losses.get("diff_weight", -1),
            "reg_weight": losses.get("reg_weight", -1),
            "ApproxAccuracy": output_trainer.get("accuracy", float("nan")),
            "Model-Hash": model_hash.hex(),
        }

        for i in range(self.x_fool.shape[0]):
            row[f"y_{i}"] = self.y_fool[i].item()

        for worker_name in self.worker_names:
            top2 = torch.topk(predictions[worker_name], k=2)

            for i in range(self.x_fool.shape[0]):
                top1_key = f"top_{1}(f^({worker_name})_(x_{i}))"
                row[top1_key] = top2.indices[i, 0].item()

                top1_key = f"top_{2}(f^({worker_name})_(x_{i}))"
                row[top1_key] = top2.indices[i, 1].item()

        for worker_name in self.worker_names:
            row[f"yhash^({worker_name})"] = hash_tensor(predictions[worker_name]).hex()

        self.csvwriter.writerow(row)
        self.logfile.flush()

    def log_success(
        self,
        iteration: int,
        model: torch.nn.Module,
        output_trainer,
        predictions: Dict[str, torch.Tensor],
        final_accuracy: float,
    ):
        assert len(predictions) == len(self.worker_names)
        filebase = os.path.join(self.log_dir, f"{iteration}")

        torch.save(model, filebase + ".pt")
        torch.save(self.x_fool, filebase + "_x_fool.pt")

        with open(filebase + "-dict.txt", "w") as f:
            for key, value in predictions.items():
                print(key, ":", torch.argmax(value, dim=-1), file=f)

        with open(filebase + ".log", "w") as f:
            print(
                {
                    "losses_for_logging": output_trainer["losses_for_logging"],
                    "accuracy": output_trainer["accuracy"],
                    "workers": self.worker_names,
                    "final_accuracy": final_accuracy,
                },
                file=f,
            )
            print(file=f)

    def create_success_file(self, name):
        log_dir = os.path.join(self.run_path, "logs")
        os.makedirs(log_dir, exist_ok=True)

        filepath = os.path.join(
            log_dir, f"{name}-{self.x_index}-{self.c_w_weight}-{self.reg_weight}"
        )

        with open(filepath, "w") as f:
            print(file=f)

    def write(self, message: str):
        assert self.logfile is not None
        self.logfile.write(message)

    def save_intermediate_model(
        self,
        iteration,
        dir_name,
        model=None,
        info=None,
        x_fool=None,
    ):
        directory = os.path.join(self.log_dir, dir_name)
        os.makedirs(directory, exist_ok=True)

        filebase = os.path.join(directory, f"{iteration}")
        if model is not None:
            torch.save(model, filebase + ".pt")

        if x_fool is not None:
            torch.save(x_fool, filebase + "_x_fool.pt")

        if info is not None:
            with open(filebase + "-log.txt", "w") as f:
                print(info, file=f)
