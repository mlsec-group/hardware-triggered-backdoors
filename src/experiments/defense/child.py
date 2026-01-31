# Experiment:
#

import argparse
import json
import os
from functools import partial
from itertools import product
import random
import re
from typing import Generator, Tuple

import numpy as np
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

from common.util import hash_tensor
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

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def get_index_dirs(
    backdoor_basedir: str, input_basedir: str, platform_name: str, other_platform: str
) -> Generator[Tuple[str, str, str, str, str, str], None, None]:
    for platform_i, platform_j in product(PLATFORMS_ABBR, PLATFORMS_ABBR):
        platform_combination = f"{platform_i}-{platform_j}"
        platform_dir = os.path.join(input_basedir, platform_combination)

        if set([other_platform, platform_name]) != set([platform_i, platform_j]):
            continue

        # The execution happens on platform_name and we inject the recorded
        # activations from other_platform
        if platform_i == platform_name:
            other_platform = platform_j
        elif platform_j == platform_name:
            other_platform = platform_i
        else:
            assert False

        if not os.path.exists(platform_dir):
            continue

        for index_subdir in sorted(os.listdir(platform_dir)):
            index_dir = os.path.join(platform_dir, index_subdir)

            with open(os.path.join(index_dir, f"adv-model-{platform_name}.txt")) as f:
                rel_model_path = f.readline().strip()

            with open(os.path.join(index_dir, f"x_fool-{platform_name}.txt")) as f:
                rel_x_path = f.readline().strip()

            y_A_path = os.path.join(index_dir, f"y_fool-{other_platform}.pt")
            y_B_path = os.path.join(index_dir, f"y_fool-{platform_name}.pt")

            data_dir = os.path.join(
                backdoor_basedir,
                platform_combination,
                "logs",
            )
            model_path = os.path.join(data_dir, rel_model_path)
            x_path = os.path.join(data_dir, rel_x_path)

            yield (
                model_path,
                x_path,
                platform_combination,
                index_subdir,
                y_A_path,
                y_B_path,
            )


@torch.inference_mode()
def process_success(model: torch.nn.Module, x_norm: torch.Tensor):
    return model(x_norm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform_name",
        choices=["a40", "a100", "a100-mig40", "rtx6000", "h100"],
        type=str,
        required=True,
    )
    parser.add_argument(
        "--other_platform_name",
        choices=["a40", "a100", "a100-mig40", "rtx6000", "h100"],
        type=str,
        required=True,
    )
    parser.add_argument("--backdoor_basedir", type=str, required=True)
    parser.add_argument("--input_basedir", type=str, required=True)
    parser.add_argument("--output_basedir", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=[
            "float32",
            "bfloat16",
            "float16",
        ],
    )
    parser.add_argument(
        "--use_deterministic", type=str, default="false", choices=["true", "false"]
    )
    args = parser.parse_args()

    assert sum([args.dtype != "float32", args.use_deterministic == "true"]) == 1

    platform_name = args.platform_name
    other_platform_name = args.other_platform_name
    backdoor_basedir = args.backdoor_basedir
    input_basedir = args.input_basedir
    output_basedir = args.output_basedir

    dtype = DTYPES[args.dtype]
    use_deterministic = args.use_deterministic.lower() == "true"

    if use_deterministic:
        print("USE DETERMINISTIC")
        torch.use_deterministic_algorithms(True, warn_only=True)
        assert torch.are_deterministic_algorithms_enabled()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    assert torch.cuda.is_available()

    loader = get_dataset_loader("imagenet", "data")

    index_dirs = list(
        get_index_dirs(
            backdoor_basedir, input_basedir, platform_name, other_platform_name
        )
    )

    for (
        model_path,
        x_path,
        platform_combination,
        index_subdir,
        y_A_path,
        y_B_path,
    ) in tqdm.tqdm(index_dirs):
        outdir = os.path.join(output_basedir, platform_combination, index_subdir)
        os.makedirs(outdir, exist_ok=True)

        y_output_path = os.path.join(outdir, f"y_dtype_{platform_name}.json")
        if os.path.exists(y_output_path):
            continue

        model: torch.nn.Module = torch.load(
            model_path, weights_only=True, map_location=device
        )
        model.eval()
        x_orig: torch.Tensor = torch.load(
            x_path, weights_only=True, map_location=device
        )
        x_norm = loader.normalize(x_orig)

        y_A: torch.Tensor = torch.load(
            y_A_path,
            weights_only=True,
        )
        y_B: torch.Tensor = torch.load(y_B_path, weights_only=True)

        with open(
            os.path.join(
                backdoor_basedir,
                platform_combination,
                "logs",
                index_subdir + "-Weights-0.1-10000.0",
                "init-log.txt",
            )
        ) as f:
            init_label_line = [
                line.strip()
                for line in f.readlines()
                if line.startswith("Initial labels")
            ][0]
            print(init_label_line)
            m = re.match(r"Initial labels:  \[([0-9][0-9]*)\]", init_label_line)
            assert m is not None
            init_label = int(m.group(1))

        label_A = torch.argmax(y_A, dim=-1).item()
        label_B = torch.argmax(y_B, dim=-1).item()

        y_exp = process_success(model, x_norm)
        assert label_B == torch.argmax(y_exp, dim=-1).item()

        if args.dtype == "float32":
            y = process_success(model, x_norm)[0]
        else:
            with torch.autocast(device_type=device.type, dtype=dtype):
                y = process_success(model, x_norm)[0]

        values, indices = torch.topk(y, k=5, largest=True, sorted=True)
        with open(y_output_path, "w") as f:
            json.dump(
                {
                    "init_label": init_label,
                    "label_A": label_A,
                    "label_B": label_B,
                    "label_dtype": torch.argmax(y, dim=-1).item(),
                    "top5_results": [
                        {
                            "class": int(idx),
                            "value": float(val),
                        }
                        for idx, val in zip(indices, values)
                    ],
                    "input_hash": hash_tensor(x_norm).hex(),
                },
                f,
                indent=4,
            )

    print("DONE")


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
