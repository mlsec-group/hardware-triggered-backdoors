from dataclasses import dataclass


@dataclass(frozen=True)
class ClientConfigMeta:
    backend: str
    commit: str
    hostname: str
