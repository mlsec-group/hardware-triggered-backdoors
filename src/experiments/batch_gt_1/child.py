# Experiment:
#

import argparse
import io
import json
import os
from functools import partial
from itertools import product
import random
import re
from typing import Generator, Tuple
import zipfile

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

TOP_K = 5
MAX_BATCH_SIZE = 256

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


@torch.inference_mode()
def process_success_batch(
    model: torch.nn.Module,
    x_norm: torch.Tensor,
    batch_size: int,
):
    X_norm = x_norm.repeat(batch_size, 1, 1, 1)
    input_hash = hash_tensor(X_norm)

    return model(X_norm), input_hash


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
    args = parser.parse_args()

    platform_name = args.platform_name
    other_platform_name = args.other_platform_name
    backdoor_basedir = args.backdoor_basedir
    input_basedir = args.input_basedir
    output_basedir = args.output_basedir
    model_type = args.model_type

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    assert torch.cuda.is_available()

    loader = get_dataset_loader("imagenet", "data")

    index_dirs = list(
        get_index_dirs(
            backdoor_basedir, input_basedir, platform_name, other_platform_name
        )
    )
    random.shuffle(index_dirs)

    for (
        model_path,
        x_path,
        platform_combination,
        index_subdir,
        y_A_path,
        y_B_path,
    ) in tqdm.tqdm(index_dirs, desc="Index directories"):
        outdir = os.path.join(output_basedir, platform_combination, index_subdir)
        os.makedirs(outdir, exist_ok=True)

        run_donefile = os.path.join(outdir, f"done_{platform_name}")
        if os.path.exists(run_donefile):
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

        batch_infos = []
        labels = []

        zip_path = os.path.join(outdir, f"Ys-topk{TOP_K}-{platform_name}.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            for batch_size in tqdm.tqdm(
                range(1, MAX_BATCH_SIZE + 1), desc="Batch sizes"
            ):
                filename_in_zip = f"Y_{batch_size}.pt"

                Y, input_hash = process_success_batch(
                    model,
                    x_norm,
                    batch_size,
                )

                batch_infos.append(
                    {
                        "batch_size": batch_size,
                        "input_hash": input_hash.hex(),
                        "filename_in_zip": filename_in_zip,
                    }
                )
                labels.append(torch.argmax(Y, dim=-1).tolist())

                # Compute top-k per row
                topk_output = torch.topk(Y, k=TOP_K, dim=-1)

                # Serialize to an in-memory buffer, then write to zip
                buf = io.BytesIO()
                torch.save(topk_output, buf)
                buf.seek(0)

                zf.writestr(filename_in_zip, buf.read())

        meta = {
            "init_label": init_label,
            "label_A": label_A,
            "label_B": label_B,
            "batch_infos": batch_infos,
        }

        with open(os.path.join(outdir, f"meta-{platform_name}.json"), "w") as f:
            json.dump(meta, f, indent=4)

        with open(os.path.join(outdir, f"labels-{platform_name}.json"), "w") as f:
            json.dump(labels, f, indent=4)

        with open(run_donefile, "w") as f:
            print(file=f)

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
