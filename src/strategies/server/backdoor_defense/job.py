import csv
import os
from typing import Any, Dict, List

import torch
from jobscheduler.client import ClientConfig
from jobscheduler.job import Job
from jobscheduler.progresstracker import ProgressTracker
from jobscheduler.worker import Worker

from strategies.server.backdoor.util import EMPTY_HASH, get_trainer_worker


class TrainerHandler:
    def __init__(self, run_id: str, trainer_worker: Worker, model_path: str):
        self.run_id = run_id
        self.trainer_worker = trainer_worker
        self.model_path = model_path

    def __enter__(self):
        self.trainer_worker.worker_step(
            EMPTY_HASH,
            {
                "run_id": self.run_id,
                "setup_trainer": True,
                "model_path": self.model_path,
            },
        )
        return self

    def __exit__(self, *args):
        self.trainer_worker.worker_step(
            EMPTY_HASH,
            {"run_id": self.run_id, "delete_trainer": True},
        )


class TesterHandler:
    def __init__(
        self,
        run_id: str,
        tester_workers: Dict[str, Worker],
        model_path: str,
        x_fool_path: str,
    ):
        self.run_id = run_id
        self.tester_workers = tester_workers
        self.model_path = model_path
        self.x_fool_path = x_fool_path

    def __enter__(self):
        for worker in self.tester_workers.values():
            worker.worker_step(
                EMPTY_HASH,
                {
                    "run_id": self.run_id,
                    "setup_tester": True,
                    "model_path": self.model_path,
                    "x_fool_path": self.x_fool_path,
                },
            )
        return self

    def __exit__(self, *args):
        for worker in self.tester_workers.values():
            worker.worker_step(
                EMPTY_HASH,
                {
                    "run_id": self.run_id,
                    "delete_tester": True,
                },
            )


class BackdoorDefenseJob(Job):
    def __init__(
        self,
        run_id: str,
        name: str,
        client_configs: List[ClientConfig],
        output_dir: str,
        model_path: str,
        x_fool_path: str,
        client_cli_args: Dict[str, Any],
        max_iterations: int,
        *,
        global_tracker: ProgressTracker,
    ):
        self.run_id = run_id
        self.name = name
        self.client_configs = client_configs

        self.model_path = model_path
        self.x_fool_path = x_fool_path

        self.iteration = 0
        self.max_iterations = max_iterations

        self.client_cli_args = client_cli_args
        self.global_tracker = global_tracker

        self.output_dir = output_dir
        self.donefile_path = os.path.join(self.output_dir, "done")
        self.summaryfile_path = os.path.join(self.output_dir, "summary.csv")

    def get_name(self):
        return self.name

    def get_required_clients(self):
        return self.client_configs

    def get_client_args(self):
        return "backdoor-defense", self.client_cli_args

    def get_progress(self):
        return self.iteration, self.max_iterations

    def init(self, worker_group: Dict[str, Worker]):
        for worker in worker_group.values():
            worker.worker_init(*self.get_client_args())

    def run(self, worker_group: Dict[str, Worker]):
        trainer_worker = get_trainer_worker(worker_group)
        tester_workers = worker_group

        os.makedirs(self.output_dir, exist_ok=True)

        y_outputs = []

        with TrainerHandler(
            self.run_id, trainer_worker, self.model_path
        ), TesterHandler(
            self.run_id, tester_workers, self.model_path, self.x_fool_path
        ):
            # Initial Accuracy
            _, init_trainer_output = trainer_worker.worker_step(
                EMPTY_HASH,
                {"run_id": self.run_id, "get_accuracy": True, "is_trainer": True},
            )

            # Evaluate on backend 1,2
            ys_tester_init = {}
            model_hashes_during_eval_init = set()
            for worker_name, worker in worker_group.items():
                _, tester_outputs = worker.worker_step(
                    EMPTY_HASH,
                    {
                        "run_id": self.run_id,
                        "evaluate": True,
                    },
                )

                model_hashes_during_eval_init.add(tester_outputs["model_hash"])
                ys_tester_init[worker_name] = tester_outputs["y"]
                torch.save(
                    tester_outputs["y"],
                    os.path.join(self.output_dir, f"y_0_{worker_name}.pt"),
                )
            assert len(model_hashes_during_eval_init) == 1
            y_outputs.append([ys_tester_init, init_trainer_output["accuracy"]])

            for iteration in range(1, self.max_iterations + 1):
                # Gradient
                _, output_trainer = trainer_worker.worker_step(
                    EMPTY_HASH,
                    {"run_id": self.run_id, "gradient_step": True},
                )
                state_dict_update_compressed = output_trainer[
                    "state_dict_update_compressed"
                ]

                # Accuracy
                _, trainer_output = trainer_worker.worker_step(
                    EMPTY_HASH,
                    {"run_id": self.run_id, "get_accuracy": True, "is_trainer": True},
                )

                # Update
                for worker in worker_group.values():
                    worker.worker_step(
                        EMPTY_HASH,
                        {
                            "run_id": self.run_id,
                            "update": True,
                            "state_dict_update_compressed": state_dict_update_compressed,
                        },
                    )

                # Evaluate on backend 1,2
                ys_tester = {}
                model_hashes_during_eval = set()
                for worker_name, worker in worker_group.items():
                    _, tester_outputs = worker.worker_step(
                        EMPTY_HASH,
                        {
                            "run_id": self.run_id,
                            "evaluate": True,
                        },
                    )

                    model_hashes_during_eval.add(tester_outputs["model_hash"])
                    ys_tester[worker_name] = tester_outputs["y"]
                    torch.save(
                        tester_outputs["y"],
                        os.path.join(
                            self.output_dir, f"y_{iteration}_{worker_name}.pt"
                        ),
                    )

                assert len(model_hashes_during_eval) == 1

                # Compare outputs
                y_outputs.append([ys_tester, trainer_output["accuracy"]])

                self.iteration = iteration
                self.global_tracker.update()

        with open(self.summaryfile_path, "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(
                [
                    "Iteration",
                    "Label1_A",
                    "Label1_B",
                    "Label2_A",
                    "Label2_B",
                    "Logit1_A",
                    "Logit1_B",
                    "Logit2_A",
                    "Logit2_B",
                    "Accuracy",
                ]
            )

            for iteration, [y_output, accuracy] in enumerate(y_outputs):
                outputs: List[torch.return_types.topk] = []
                for worker_name, y in y_output.items():
                    outputs.append(torch.topk(y, k=2))

                csvwriter.writerow(
                    [
                        iteration,
                        outputs[0].indices[0, 0].item(),
                        outputs[1].indices[0, 0].item(),
                        outputs[0].indices[0, 1].item(),
                        outputs[1].indices[0, 1].item(),
                        outputs[0].values[0, 0].item(),
                        outputs[1].values[0, 0].item(),
                        outputs[0].values[0, 1].item(),
                        outputs[1].values[0, 1].item(),
                        accuracy,
                    ]
                )

        with open(self.donefile_path, "w") as f:
            print(file=f)
