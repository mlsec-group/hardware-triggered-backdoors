from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List
import uuid

from jobscheduler.client import ClientConfig
from jobscheduler.job import Job
from jobscheduler.worker import Worker

EMPTY_HASH = bytes(32)


class ExampleJob(Job):
    def __init__(
        self,
        name: str,
        client_configs: List[ClientConfig],
        client_cli_args,
        *,
        n_iterations: int,
    ):
        self.name = name
        self.client_configs = client_configs
        self.client_cli_args = client_cli_args

        self.iteration = 0
        self.n_iterations = n_iterations

    def get_name(self):
        return self.name

    def get_required_clients(self):
        return self.client_configs

    def get_client_args(self):
        return "example", self.client_cli_args

    def get_progress(self):
        return self.iteration, self.n_iterations

    def init(self, worker_group: Dict[str, Worker]):
        for worker in worker_group.values():
            worker.worker_init(*self.get_client_args())

    def run(self, worker_group: Dict[str, Worker]):
        run_id = str(uuid.uuid4())
        assert all(
            self.client_configs[i].client_identifier in worker_name
            for i, worker_name in enumerate(worker_group)
        ), f"Workers are not in the same order as the client configs"

        for iteration in range(self.n_iterations):
            print(f"Server iteration {iteration} in {self.get_name()}")

            for worker_name, worker in worker_group.items():
                # invokes src/strategies/client/example.py ExampleClient:step()
                client_hash, client_output = worker.worker_step(
                    EMPTY_HASH,
                    {"run_id": run_id, "example_action": True, "iteration": iteration},
                )
                print("Worker output: ", worker_name, client_output["y"])

            self.iteration = iteration

        return f"Success of '{self.get_name()}'"
