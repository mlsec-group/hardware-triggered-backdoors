import abc
from typing import Any, Dict, List, Tuple, Union

from jobscheduler.client import ClientConfig
from jobscheduler.worker import Worker


class Job(abc.ABC):
    @abc.abstractmethod
    def get_name(self) -> str: ...

    @abc.abstractmethod
    def init(self, worker_group: Dict[str, Worker]) -> None: ...

    @abc.abstractmethod
    def run(self, worker_group: Dict[str, Worker]) -> Any: ...

    @abc.abstractmethod
    def get_required_clients(self) -> List[ClientConfig]: ...

    @abc.abstractmethod
    def get_progress(self) -> Tuple[int, Union[None, int]]: ...

    @abc.abstractmethod
    def get_client_args(self) -> Tuple[Any]: ...
