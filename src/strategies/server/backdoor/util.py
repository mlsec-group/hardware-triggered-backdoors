import functools
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

import torch
from jobscheduler.worker import Worker

from common.util import hash_tensor

from torchvision.models.vision_transformer import VisionTransformer

EMPTY_HASH = hash_tensor()


def get_relevant_parameters(
    model: torch.nn.Module, use_full_model: bool
) -> Dict[str, torch.nn.Parameter]:
    if use_full_model:
        return dict(model.named_parameters())
    else:
        if isinstance(model, VisionTransformer):
            return {
                k: v
                for k, v in model.named_parameters()
                if k == "conv_proj.weight" or k == "conv_proj.bias"
            }
        else:
            assert False


def get_trainer_worker(workers: Dict[str, Worker]):
    for worker_name in workers:
        if worker_name.endswith(":gpu"):
            return workers[worker_name]

    assert False


def predict_on_worker(
    worker_name: str,
    worker: Worker,
    *,
    run_id: str,
    x_fool: torch.Tensor,
    use_full_model: bool,
    model_hash: bytes,
    state_dict_update_compressed: Optional[bytes] = None
):
    client_hash, output = worker.worker_step(
        EMPTY_HASH,
        {
            "run_id": run_id,
            "test": True,
            "x_fool": x_fool,
            "state_dict_update_compressed": state_dict_update_compressed,
            "is_trainer": False,
            "use_full_model": use_full_model,
            "model_hash": model_hash,
        },
    )
    return worker_name, client_hash, output


def predict_on_workers(
    run_id: str,
    x_fool: torch.Tensor,
    workers: Dict[str, Worker],
    executor: ThreadPoolExecutor,
    use_full_model: bool,
    model_hash: bytes,
    state_dict_update_compressed: Optional[bytes] = None,
):
    predict_on_worker_bound = functools.partial(
        predict_on_worker,
        run_id=run_id,
        x_fool=x_fool,
        use_full_model=use_full_model,
        model_hash=model_hash,
        state_dict_update_compressed=state_dict_update_compressed,
    )

    futures = [
        executor.submit(
            predict_on_worker_bound,
            worker_name,
            worker,
        )
        for worker_name, worker in workers.items()
    ]

    worker_outputs: Dict[str, Dict[str, Any]] = {}
    predictions: Dict[str, torch.Tensor] = {}
    client_hashes = set()
    model_hashes = set()

    for future in as_completed(futures):
        worker_name, client_hash, output = future.result()
        client_hashes.add(client_hash)
        model_hashes.add(output["model_hash"])

        worker_outputs[worker_name] = output
        predictions[worker_name] = output["prediction"]
        assert x_fool.shape[0] == output["prediction"].shape[0]

    assert len(client_hashes) == 1
    assert len(model_hashes) == 1

    return worker_outputs, predictions, list(model_hashes)[0]
