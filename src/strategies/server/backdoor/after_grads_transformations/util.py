import io
import zlib
import torch


def all_different(X: torch.Tensor) -> bool:
    """
    Check if for every sample, all workers predict different classes:

    (i != j) -> (X[i, o] != X[j, o])

    X: (n_workers, n_samples), int tensor
    Returns: bool
    """
    return all(
        X[:, sample_id].unique().numel() == X.shape[0]
        for sample_id in range(X.shape[1])
    )


def one_vs_all(X: torch.Tensor, worker_id: int) -> bool:
    """
    Check whether the worker_id is different (for all samples) different from
    the other rows, while the other rows are all equivalent.

    X: (n_workers, n_samples), int tensor
    Returns: bool
    """

    if X.shape[0] == 1:
        return True

    # Separate the target row and the other rows
    target_row = X[worker_id]
    other_rows = torch.cat([X[:worker_id], X[worker_id + 1 :]], dim=0)

    # Check if all other rows are identical
    if not torch.all(other_rows == other_rows[0]):
        return False

    # Check if target_row has no values in common with the other row
    return not torch.any(target_row.unsqueeze(0) == other_rows[0])
