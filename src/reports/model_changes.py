# %%

import os
from functools import partial

import torch
import torch.nn as nn
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
import tqdm

from experiments.util import list_all_backdoored_models
from reports.util import RESEARCH_DIR, efficientnet_bit, resnet_bit, vit_full

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


@torch.inference_mode()
def get_model_diffs(model_name: str, orig_model_path: str):
    device = torch.device("cpu")
    model: torch.nn.Module = torch.load(orig_model_path, weights_only=True)

    backdoored_models = list(
        list_all_backdoored_models(
            os.path.join(RESEARCH_DIR, model_name),
            ["a40", "a100", "h100", "rtx6000", "a100-mig40"],
        )
    )

    for i, (
        _,
        _,
        logs_dir,
        x_index,
        rel_model_path,
        rel_x_path,
        success_type,
    ) in enumerate(tqdm.tqdm(backdoored_models)):
        backdoored_model: torch.nn.Module = torch.load(
            os.path.join(logs_dir, rel_model_path), weights_only=True
        )

        difference = torch.tensor(0, dtype=torch.float64)
        n = 0

        print(os.path.join(logs_dir, rel_model_path))

        for p, p_ in zip(model.parameters(), backdoored_model.parameters()):
            diff = torch.sum(
                torch.abs(p.to(device, torch.float64) - p_.to(device, torch.float64))
            )

            n += p.numel()
            difference += diff

        print(difference, n, difference / n)

        if i == 100:
            break


get_model_diffs(vit_full, "models/imagenet/vit_b_32.pt")
