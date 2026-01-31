import os
from itertools import product
from typing import List


def list_all_backdoored_models(rundir: str, platforms: List[str]):
    for platformA, platformB in product(platforms, platforms):
        logs_dir = os.path.join(rundir, platformA + "-" + platformB, "logs")

        if not os.path.exists(logs_dir):
            continue

        for (
            x_index,
            rel_model_path,
            rel_x_path,
            success_type,
        ) in list_all_backdoored_models_from_log_dir(logs_dir):
            yield platformA, platformB, logs_dir, x_index, rel_model_path, rel_x_path, success_type


def list_all_backdoored_models_from_log_dir(logs_dir: str):
    SUFFIX = "_x_fool.pt"

    for filename in sorted(os.listdir(logs_dir)):
        try:
            success_type, x_index, beta, gamma = filename.split("-")
        except ValueError:
            continue

        assert beta == "0.1"
        assert gamma == "10000.0"

        rel_index_dir = f"Index-{x_index}-Weights-{beta}-{gamma}"
        index_dir = os.path.join(logs_dir, rel_index_dir)
        assert os.path.exists(index_dir)

        if success_type == "grad":
            candidate = [f for f in os.listdir(index_dir) if f.endswith(SUFFIX)][0]
            success_run_index = candidate[: -len(SUFFIX)]

            rel_model_path = os.path.join(rel_index_dir, f"{success_run_index}.pt")
            rel_x_path = os.path.join(rel_index_dir, f"{success_run_index}_x_fool.pt")
        elif success_type == "bit_flip" or success_type == "permute":
            if success_type == "bit_flip":
                inter_dir = "success-bitflip-models"
            elif success_type == "permute":
                inter_dir = "success-permute-models"
            else:
                assert False

            success_dir = os.path.join(index_dir, inter_dir)

            candidate = [f for f in os.listdir(success_dir) if f.endswith(SUFFIX)][0]
            success_run_index = candidate[: -len(SUFFIX)]

            rel_model_path = os.path.join(
                rel_index_dir, inter_dir, f"{success_run_index}.pt"
            )
            rel_x_path = os.path.join(
                rel_index_dir, inter_dir, f"{success_run_index}_x_fool.pt"
            )
        else:
            assert False

        yield x_index, rel_model_path, rel_x_path, success_type
