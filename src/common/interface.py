from dataclasses import dataclass
from typing import List


SHA256_SIZE = 32
UINT_SIZE = 4


@dataclass(frozen=True)
class DirConfig:
    share_dir: str
    readonly_dir: List[str]
    project_dir: str
    tmp_share_dir: str
    datasets_dir: str
