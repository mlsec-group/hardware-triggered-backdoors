import abc
from typing import Generator, List, Union

from jobscheduler.client import ClientConfig
from jobscheduler.job import Job
from jobscheduler.progresstracker import ProgressTracker

from common.interface import DirConfig


class ServerStrategy(abc.ABC):
    def __init__(
        self,
        run_path: str,
        dir_config: DirConfig,
        seed: int,
        client_configs: List[ClientConfig],
    ):
        self.run_path = run_path
        self.dir_config = dir_config
        self.seed = seed
        self.client_configs = client_configs

    @abc.abstractmethod
    def get_number_of_steps_per_job(self) -> int: ...

    @abc.abstractmethod
    def get_number_of_jobs(self) -> int: ...

    @abc.abstractmethod
    def start_campaign(
        self, scheduler, *, global_tracker: Union[ProgressTracker, None] = None
    ): ...

    @abc.abstractmethod
    def final(self) -> None: ...
