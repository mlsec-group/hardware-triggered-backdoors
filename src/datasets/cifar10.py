import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np


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

