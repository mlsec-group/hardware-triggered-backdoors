from argparse import Namespace
import datetime
from itertools import combinations
import os
import time
from typing import List, Optional, Union

import math
import torch
from common.util import hash_tensor
from jobscheduler.client import ClientConfig
from jobscheduler.progresstracker import ProgressTracker
from jobscheduler.scheduler import Scheduler
from torch.utils.data import DataLoader

from common.interface import DirConfig
from datasets.common import EnumerateDataset
from datasets.loader import get_dataset_loader
from models import ResNet18
from strategies.server.backdoor.job import BackdoorJob, JobOutput
from strategies.server_cli.backdoor import BackdoorCLI
from strategies.server_strategy import ServerStrategy


class BackdoorServer(BackdoorCLI, ServerStrategy):
    def __init__(
        self,
        run_path: str,
        dir_config: DirConfig,
        seed: int,
        client_configs: List[ClientConfig],
        args: Namespace,
    ):
        super().__init__(run_path, dir_config, seed, client_configs)

        self.model_path = args.model_path
        self.model_type = args.model_type
        self.dataset = args.dataset
        self.model_dtype = args.model_dtype
        self.n_poison_samples = args.n_poison_samples
        self.n_iterations = args.n_iterations
        self.heuristic = args.heuristic

        self.permute_after_gradient = args.permute_after_gradient.lower() == "true"
        self.flip_bits_after_gradient = args.flip_bits_after_gradient.lower() == "true"
        self.n_bits_flipped = args.n_bits_flipped

        self.n_samples = args.n_samples

        self.initial_diff_weight = 1
        self.c_w_weight = 0.1
        self.reg_weight = 1e4

        self.do_crossover = args.do_crossover.lower() == "true"
        self.do_one_vs_all = args.do_one_vs_all.lower() == "true"
        self.use_full_model = args.use_full_model.lower() == "true"
        self.use_deterministic = args.use_deterministic.lower() == "true"
        self.skip_is_prediction_close_check = (
            args.skip_is_prediction_close_check.lower() == "true"
        )

        #
        # Dataset
        #
        self.loader = get_dataset_loader(self.dataset, self.dir_config.datasets_dir)

        self.train_dataset = self.loader.load_train_deterministic()
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.n_poison_samples,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed + 0xDEAD),
        )

        torch.serialization.add_safe_globals([ResNet18])

        self.campaign_start: Optional[datetime.datetime] = None

    def get_number_of_steps_per_job(self) -> int:
        return self.n_iterations

    def get_number_of_jobs(self) -> int:
        return self.n_samples * math.comb(len(self.client_configs), 2)

    def start_campaign(
        self,
        scheduler: Scheduler,
        *,
        global_tracker: Union[ProgressTracker, None] = None,
    ):
        self.campaign_start = datetime.datetime.now()

        client_cli_args = {
            "--model_path": str(self.model_path),
            "--model_dtype": self.model_dtype,
            "--dataset": str(self.dataset),
            "--config_dataset_dir": str(self.dir_config.datasets_dir),
            "--seed": str(self.seed),
        }
        heuristic_parameters = {
            "diff_weight": self.initial_diff_weight,
            "c_w_weight": self.c_w_weight,
            "reg_weight": self.reg_weight,
            "do_crossover": self.do_crossover,
        }

        data_iter = iter(self.train_loader)

        for x_index in range(self.n_samples):
            x_fool, y_fool = next(data_iter)

            if self.do_one_vs_all:
                for i in range(len(self.client_configs)):
                    # The first element in the list is the "one", the rest of the
                    # list is the "all" part in "one vs all".
                    client_order = self.client_configs[i:] + self.client_configs[:i]

                    scheduler.try_add_job(
                        BackdoorJob(
                            f"BackdoorJob-{x_index}",
                            client_order,
                            self.model_path,
                            x_index,
                            x_fool,
                            y_fool,
                            client_cli_args,
                            n_iterations=self.n_iterations,
                            heuristic=self.heuristic,
                            heuristic_parameters=heuristic_parameters,
                            run_path=self.run_path,
                            global_tracker=global_tracker,
                            permute_after_gradient=self.permute_after_gradient,
                            flip_bits_after_gradient=self.flip_bits_after_gradient,
                            n_bits_flipped=self.n_bits_flipped,
                            do_one_vs_all=self.do_one_vs_all,
                            use_full_model=self.use_full_model,
                            use_deterministic=self.use_deterministic,
                            skip_is_prediction_close_check=self.skip_is_prediction_close_check,
                        ),
                        callback=self.job_finished,
                    )
            else:
                for client_i, client_j in combinations(self.client_configs, 2):
                    platform_i = client_i.client_identifier
                    platform_j = client_j.client_identifier

                    if set([platform_i, platform_j]) == set(["a40:gpu", "rtx3090:gpu"]):
                        continue

                    names = "-".join(
                        [
                            p[:-4] if p.endswith(":gpu") else p
                            for p in [platform_i, platform_j]
                        ]
                    )

                    scheduler.try_add_job(
                        BackdoorJob(
                            f"BackdoorJob-{names}-{x_index}",
                            [client_i, client_j],
                            self.model_path,
                            x_index,
                            x_fool,
                            y_fool,
                            client_cli_args,
                            n_iterations=self.n_iterations,
                            heuristic=self.heuristic,
                            heuristic_parameters=heuristic_parameters,
                            run_path=self.run_path,
                            global_tracker=global_tracker,
                            permute_after_gradient=self.permute_after_gradient,
                            flip_bits_after_gradient=self.flip_bits_after_gradient,
                            n_bits_flipped=self.n_bits_flipped,
                            use_full_model=self.use_full_model,
                            use_deterministic=self.use_deterministic,
                            skip_is_prediction_close_check=self.skip_is_prediction_close_check,
                        ),
                        callback=self.job_finished,
                    )

    def job_finished(self, output: JobOutput):
        with open(os.path.join(output.logger_base_dir, "job-log.txt"), "w") as f:
            print("Name: ", output.name, file=f)
            print("Message: ", output.message, file=f)
            print("Start: ", output.job_start, file=f)
            print("End: ", output.job_end, file=f)

            print("Iterations:", file=f)
            for iteration_record in output.iteration_times:
                print(
                    " - ",
                    (iteration_record.end - iteration_record.start).total_seconds(),
                    file=f,
                )
                for sub_measurement in iteration_record.sub_measurements:
                    print(
                        "   - ",
                        (sub_measurement[2] - sub_measurement[1]),
                        sub_measurement[0],
                        file=f,
                    )

    def final(self) -> None:
        self.campaign_end = datetime.datetime.now()

        with open(os.path.join(self.run_path, "campaign-log.txt"), "w") as f:
            print("Start: ", self.campaign_start, file=f)
            print("End: ", self.campaign_end, file=f)
