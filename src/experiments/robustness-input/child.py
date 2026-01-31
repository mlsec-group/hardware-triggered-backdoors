# Experiment:
#

import argparse
import hashlib
import os
from functools import partial
from itertools import product
from typing import List

import torch
import torch.nn as nn
import tqdm
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torchvision.models.efficientnet import EfficientNet, FusedMBConv, MBConv
from torchvision.models.resnet import BasicBlock, ResNet
from torchvision.models.vision_transformer import (
    Encoder,
    EncoderBlock,
    MLPBlock,
    VisionTransformer,
)
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.ops.stochastic_depth import StochasticDepth

from common.util import hash_tensor, nextafter
from datasets.common import DatasetLoader
from datasets.loader import get_dataset_loader

PLATFORMS_ABBR = [
    "a40",
    "a100",
    "a100-mig40",
    "rtx6000",
    "h100",
    # "rtx3090", # == a40
]

PLATFORMS = [
    "A40",
    "A100",
    "A100-mig40",
    "RTX6000",
    "H100",
    # "RTX 3090", # == a40
]


@torch.inference_mode()
def process_success(
    n: int,
    model: torch.nn.Module,
    x_orig: torch.Tensor,
    *,
    device: torch.device,
    loader: DatasetLoader,
    ulp_dist: int,
    seed: int,
):
    perturbation_hashes: List[str] = []
    x_hashes: List[str] = []
    x_norm_hashes: List[str] = []
    y: List[int] = []

    generator = torch.Generator().manual_seed(seed)

    for i in range(n):
        perturbation_vector = torch.randint(
            -ulp_dist,
            ulp_dist + 1,
            x_orig.shape,
            generator=generator,
            dtype=torch.int32,
        )
        perturbation_hashes.append(hash_tensor(perturbation_vector).hex())

        x = nextafter(x_orig, perturbation_vector).to(device)
        x_hashes.append(hash_tensor(x).hex())

        x_norm = loader.normalize(x)
        x_norm_hashes.append(hash_tensor(x_norm).hex())

        y.append(int(torch.argmax(model(x_norm)).item()))

    return perturbation_hashes, x_hashes, x_norm_hashes, y


def get_index_dirs(backdoor_basedir: str, input_basedir: str, platform_name: str):
    for platform_i, platform_j in product(PLATFORMS_ABBR, PLATFORMS_ABBR):
        platform_combination = f"{platform_i}-{platform_j}"
        platform_dir = os.path.join(input_basedir, platform_combination)

        if platform_i == platform_j:
            continue

        if platform_name not in [platform_i, platform_j]:
            continue

        if not os.path.exists(platform_dir):
            continue

        for index_subdir in sorted(os.listdir(platform_dir)):
            index_dir = os.path.join(platform_dir, index_subdir)

            with open(os.path.join(index_dir, f"adv-model-{platform_name}.txt")) as f:
                rel_model_path = f.readline().strip()

            with open(os.path.join(index_dir, f"x_fool-{platform_name}.txt")) as f:
                rel_x_path = f.readline().strip()

            data_dir = os.path.join(
                backdoor_basedir,
                platform_combination,
                "logs",
            )
            model_path = os.path.join(data_dir, rel_model_path)
            x_path = os.path.join(data_dir, rel_x_path)

            yield model_path, x_path, platform_combination, index_subdir


def unprocessed_ulps(
    ulps: List[int],
    platform_name: str,
    output_basedir: str,
    platform_combination: str,
    index_subdir: str,
):
    for ulp_dist in ulps:
        output_dir = os.path.join(
            output_basedir, f"ulp-{ulp_dist}", platform_combination, index_subdir
        )

        permutations_path = os.path.join(
            output_dir, f"permutations-{platform_name}.txt"
        )
        x_hashes_path = os.path.join(output_dir, f"x_hashes-{platform_name}.txt")
        x_norm_hashes_path = os.path.join(
            output_dir, f"x_norm_hashes-{platform_name}.txt"
        )
        y_path = os.path.join(output_dir, f"y-{platform_name}.txt")
        seed_path = os.path.join(output_dir, f"seed-{platform_name}.txt")

        if (
            os.path.exists(permutations_path)
            and os.path.exists(x_hashes_path)
            and os.path.exists(x_norm_hashes_path)
            and os.path.exists(y_path)
            and os.path.exists(seed_path)
        ):
            print("SKIP", output_dir, flush=True)
            continue

        yield ulp_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform_name",
        choices=["a40", "a100", "a100-mig40", "rtx6000", "h100"],
        type=str,
        required=True,
    )
    parser.add_argument("--backdoor_basedir", type=str, required=True)
    parser.add_argument("--input_basedir", type=str, required=True)
    parser.add_argument("--output_basedir", type=str, required=True)
    parser.add_argument("--n_noise", type=int, default=1024)
    args = parser.parse_args()

    platform_name = args.platform_name
    backdoor_basedir = args.backdoor_basedir
    input_basedir = args.input_basedir
    output_basedir = args.output_basedir
    n_noise = args.n_noise

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    assert torch.cuda.is_available()

    loader = get_dataset_loader("imagenet", "data")

    n_total = 0
    runs = {}

    for model_path, x_path, platform_combination, index_subdir in get_index_dirs(
        backdoor_basedir, input_basedir, platform_name
    ):
        ulps = list(
            unprocessed_ulps(
                [1, 10, 100, 1000, 10000, 100000],
                platform_name,
                output_basedir,
                platform_combination,
                index_subdir,
            )
        )
        if len(ulps) == 0:
            continue

        runs[model_path, x_path, platform_combination, index_subdir] = ulps
        n_total += len(ulps)

    with tqdm.tqdm(total=n_total) as tbar:
        for (
            model_path,
            x_path,
            platform_combination,
            index_subdir,
        ), ulps in runs.items():
            model: torch.nn.Module = torch.load(
                model_path, weights_only=True, map_location=device
            )
            x_orig: torch.Tensor = torch.load(
                x_path, weights_only=True, map_location=torch.device("cpu")
            )

            for ulp_dist in ulps:
                output_dir = os.path.join(
                    output_basedir,
                    f"ulp-{ulp_dist}",
                    platform_combination,
                    index_subdir,
                )

                permutations_path = os.path.join(
                    output_dir, f"permutations-{platform_name}.txt"
                )
                x_hashes_path = os.path.join(
                    output_dir, f"x_hashes-{platform_name}.txt"
                )
                x_norm_hashes_path = os.path.join(
                    output_dir, f"x_norm_hashes-{platform_name}.txt"
                )
                y_path = os.path.join(output_dir, f"y-{platform_name}.txt")
                seed_path = os.path.join(output_dir, f"seed-{platform_name}.txt")

                print(output_dir, flush=True)
                os.makedirs(output_dir, exist_ok=True)

                h = hashlib.sha256()
                h.update(index_subdir.encode())
                h.update(platform_combination.encode())
                h.update(str(ulp_dist).encode())
                seed = int(h.hexdigest(), 16) % (2**32)

                perturbation_hashes, x_hashes, x_norm_hashes, y = process_success(
                    n_noise,
                    model,
                    x_orig,
                    device=device,
                    loader=loader,
                    ulp_dist=ulp_dist,
                    seed=seed,
                )

                with open(permutations_path, "w") as f:
                    print(*perturbation_hashes, sep="\n", file=f)

                with open(x_hashes_path, "w") as f:
                    print(*x_hashes, sep="\n", file=f)

                with open(x_norm_hashes_path, "w") as f:
                    print(*x_norm_hashes, sep="\n", file=f)

                with open(y_path, "w") as f:
                    print(*y, sep="\n", file=f)

                with open(seed_path, "w") as f:
                    print(seed, file=f)

                tbar.update()


if __name__ == "__main__":
    torch.serialization.add_safe_globals(
        [
            EfficientNet,
            set,
            Conv2dNormActivation,
            FusedMBConv,
            MBConv,
            StochasticDepth,
            SqueezeExcitation,
            BasicBlock,
            nn.AvgPool2d,
            nn.AdaptiveAvgPool2d,
            nn.BatchNorm2d,
            nn.Conv2d,
            nn.Dropout,
            nn.GELU,
            nn.LayerNorm,
            nn.Linear,
            nn.MaxPool2d,
            nn.MultiheadAttention,
            nn.ReLU,
            nn.Sequential,
            nn.SiLU,
            nn.Sigmoid,
            ResNet,
            NonDynamicallyQuantizableLinear,
            VisionTransformer,
            Encoder,
            EncoderBlock,
            MLPBlock,
            partial,
        ]
    )
    main()
