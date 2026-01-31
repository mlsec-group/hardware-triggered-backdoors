import abc
from typing import Tuple

import torch
from torch.utils.data import Dataset


class DatasetLoader(abc.ABC):
    @abc.abstractmethod
    def normalize(self, X: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def load_train(self) -> Dataset: ...

    @abc.abstractmethod
    def load_train_deterministic(self) -> Dataset: ...

    @abc.abstractmethod
    def load_test(self) -> Dataset: ...

    @abc.abstractmethod
    def input_shape(self) -> Tuple[int, int, int]: ...

    @abc.abstractmethod
    def output_shape(self) -> Tuple[int, ...]: ...


class EnumerateDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        item = self.dataset[key]
        return key, item
