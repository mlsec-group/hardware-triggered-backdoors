import csv
import os
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Union

from jobscheduler.client import ClientConfig
from jobscheduler.progresstracker import ProgressTracker
from jobscheduler.scheduler import Scheduler

from common.interface import DirConfig
from strategies.server.backdoor_defense.job import BackdoorDefenseJob
from strategies.server_cli.backdoor_defense import BackdoorDefenseCLI
from strategies.server_strategy import ServerStrategy


@dataclass
class BackdooredModelInfo:
    platformA: str
    platformB: str
    logs_dir: str
    x_index: str
    rel_model_path: str
    rel_x_path: str
    success_type: str


def parser_backdoor_filelist(filepath: str):
    filelist: List[BackdooredModelInfo] = []

    with open(filepath) as csvfile:
        reader = csv.DictReader(csvfile)

        for line in reader:
            filelist.append(BackdooredModelInfo(**line))

    return filelist


class BackdoorDefenseServer(BackdoorDefenseCLI, ServerStrategy):
    def __init__(
        self,
        run_path: str,
        dir_config: DirConfig,
        seed: int,
        client_configs: List[ClientConfig],
        args: Namespace,
    ):
        super().__init__(run_path, dir_config, seed, client_configs)
        self.backdoor_filelist = args.backdoor_filelist
        self.max_iterations = args.max_iterations

        self.backdoored_models = parser_backdoor_filelist(self.backdoor_filelist)

    def get_number_of_steps_per_job(self) -> int:
        return self.max_iterations

    def get_number_of_jobs(self) -> int:
        return len(self.backdoored_models)

    def start_campaign(
        self,
        scheduler: Scheduler,
        *,
        global_tracker: Union[ProgressTracker, None] = None,
    ):
        client_cli_args = {
            "--dataset": "imagenet",
            "--config_dataset_dir": str(self.dir_config.datasets_dir),
        }

        for run_id, info in enumerate(self.backdoored_models):
            client_identifier_A = info.platformA
            client_identifier_B = info.platformB

            client_identifier_A += ":gpu"
            client_identifier_B += ":gpu"

            try:
                clientA = [
                    c
                    for c in self.client_configs
                    if c.client_identifier == client_identifier_A
                ][0]
                clientB = [
                    c
                    for c in self.client_configs
                    if c.client_identifier == client_identifier_B
                ][0]
            except IndexError:
                continue

            job_dir = os.path.join(
                self.run_path,
                "results",
                info.logs_dir,
                f"Index-{info.x_index}-Weights-0.1-10000.0",
            )
            model_path = os.path.join("/data", info.logs_dir, info.rel_model_path)
            x_fool_path = os.path.join("/data", info.logs_dir, info.rel_x_path)

            scheduler.try_add_job(
                BackdoorDefenseJob(
                    str(run_id),
                    f"BackdoorDefenseJob-{run_id}",
                    [clientA, clientB],
                    job_dir,
                    model_path,
                    x_fool_path,
                    client_cli_args,
                    self.max_iterations,
                    global_tracker=global_tracker,
                ),
                callback=self.job_finished,
            )

    def job_finished(self, output):
        pass

    def final(self) -> None:
        pass
