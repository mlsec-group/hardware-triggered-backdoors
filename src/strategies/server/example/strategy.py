from argparse import Namespace
from itertools import combinations
import math
from typing import List, Union

from common.interface import DirConfig
from jobscheduler.client import ClientConfig
from jobscheduler.progresstracker import ProgressTracker
from jobscheduler.scheduler import Scheduler
from strategies.server.example.job import ExampleJob
from strategies.server_cli.example import ExampleCLI
from strategies.server_strategy import ServerStrategy


class ExampleServer(ExampleCLI, ServerStrategy):
    def __init__(
        self,
        run_path: str,
        dir_config: DirConfig,
        seed: int,
        client_configs: List[ClientConfig],
        args: Namespace,
    ):
        super().__init__(run_path, dir_config, seed, client_configs)

        # Arguments are parsed by src/strategies/server_cli/example.py
        self.example_client_arg = args.example_client_arg
        self.example_server_arg = args.example_server_arg

        self.n_iterations = 10

    def get_number_of_steps_per_job(self) -> int:
        return self.n_iterations

    def get_number_of_jobs(self) -> int:
        return math.comb(len(self.client_configs), 2)

    def start_campaign(
        self,
        scheduler: Scheduler,
        *,
        global_tracker: Union[ProgressTracker, None] = None,
    ):
        client_cli_args = {
            "--example_arg": self.example_client_arg,
        }

        for i, (client_i, client_j) in enumerate(combinations(self.client_configs, 2)):
            # A job is added to our scheduler. Once the required clients
            # (client_i and client_j) are available, the scheduler will
            # assign the job to them. The job in
            #
            #       src/strategies/server/example/job.py
            #
            # is executed on the server. The client code is located under
            #
            #       src/strategies/client/example.py
            #
            # and can be invoked from the server using "worker.worker_step()".
            # You can pass initialization parameters to the clients via
            # `client_cli_args`.
            scheduler.try_add_job(
                ExampleJob(
                    f"ExampleJob-{i}",
                    [client_i, client_j],
                    client_cli_args,
                    n_iterations=self.n_iterations,
                ),
                callback=self.job_finished,
            )

    def job_finished(self, output):
        print("Job finished with output: ", output)

    def final(self) -> None:
        print("All jobs finished!")
