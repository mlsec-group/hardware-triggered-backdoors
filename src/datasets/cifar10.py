import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np


TRAIN_BATCHES = tuple(f"data_batch_{idx}" for idx in range(1, 6))


def load_cifar10_batch(batch_path: str | Path) -> Tuple[np.ndarray, List[int]]:
    """Load an official CIFAR-10 python-format batch.

    Returns images as float32 arrays shaped ``[N, 3, 32, 32]`` in ``[0, 1]``.
    """

    batch_path = Path(batch_path)
    batch = pickle.loads(batch_path.read_bytes(), encoding="bytes")
    images = batch[b"data"].astype(np.float32) / 255.0
    images = images.reshape(-1, 3, 32, 32)
    labels = [int(label) for label in batch[b"labels"]]
    return images, labels


def load_cifar10_batches(path: str | Path) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """Load CIFAR-10 images from one batch file or all train batches in a directory.

    Returns images, labels, and stable source indices. Directory loading uses the
    five official training batches and intentionally excludes ``test_batch``.
    """

    path = Path(path)
    if path.is_dir():
        batch_paths = [path / name for name in TRAIN_BATCHES]
        missing = [str(batch_path) for batch_path in batch_paths if not batch_path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing CIFAR-10 training batch files: " + ", ".join(missing)
            )
    else:
        batch_paths = [path]

    image_batches = []
    labels: List[int] = []
    for batch_path in batch_paths:
        images, batch_labels = load_cifar10_batch(batch_path)
        image_batches.append(images)
        labels.extend(batch_labels)

    all_images = np.concatenate(image_batches, axis=0)
    source_indices = np.arange(all_images.shape[0], dtype=np.int64)
    return all_images, labels, source_indices
