import os

from datasets.common import DatasetLoader
from datasets.imagenet import ImageNetLoader

DATASET_LOADERS = {
    "imagenet": ImageNetLoader,
}


def get_datasets():
    return set(DATASET_LOADERS.keys())


def get_dataset_loader(dataset, config_dataset_dir) -> DatasetLoader:
    loader_cls = DATASET_LOADERS.get(dataset)
    assert loader_cls is not None

    if config_dataset_dir is None:
        return loader_cls(None)

    dataset_dir = os.path.join(config_dataset_dir, dataset)
    return loader_cls(dataset_dir)
