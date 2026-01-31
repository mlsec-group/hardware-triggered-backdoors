import abc
from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class HeuristicOutput:
    updated_model: Optional[torch.nn.Module]
    losses_for_logging: Dict[str, torch.Tensor]


class Heuristic(abc.ABC):
    @abc.abstractmethod
    def step(self, predictions: Dict[str, torch.Tensor]) -> HeuristicOutput: ...
