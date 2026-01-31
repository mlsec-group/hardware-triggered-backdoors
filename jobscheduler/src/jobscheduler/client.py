import abc
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ClientConfig(Generic[T]):
    client_identifier: str
    meta: T = field(default=None)


class Client(abc.ABC):
    @abc.abstractmethod
    def get_client_identifier(self) -> str: ...

    @abc.abstractmethod
    def get_name(self) -> str: ...

    @abc.abstractmethod
    def get_config(self) -> ClientConfig: ...

    @abc.abstractmethod
    def client_arguments_match(self, *client_args) -> bool: ...

    @abc.abstractmethod
    def client_init(self, *client_args): ...

    @abc.abstractmethod
    def client_step(self, server_hash, client_input): ...

    @abc.abstractmethod
    def client_heartbeat(self): ...

    @abc.abstractmethod
    def close(self): ...
