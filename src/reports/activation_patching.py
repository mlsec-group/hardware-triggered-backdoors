# %%
# ################
# ################
# Experiment: Activation patching
# ################
# ################

import json
import os
from dataclasses import dataclass
from itertools import product
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from reports.util import (
    PLATFORMS_ABBR,
    ModelType,
    efficientnet_bit,
    resnet_bit,
    vit_full,
)


@dataclass
class LayerOutputMeta:
    layer_id: int
    layer_name: str
    module_type: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    input_hash: str
    output_hash: str
    filename: str


def get_index_dirs(
    input_basedir: str, activation_output_basedir: str, platform_name: str
):
    for platform_i, platform_j in product(PLATFORMS_ABBR, PLATFORMS_ABBR):
        platform_combination = f"{platform_i}-{platform_j}"
        platform_dir = os.path.join(input_basedir, platform_combination)

        if platform_i == platform_j:
            continue

        if platform_name not in [platform_i, platform_j]:
            continue

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
            activations_subdir = f"activations-{other_platform}"
            activations_dir = os.path.join(index_dir, activations_subdir)

            try:
                with open(os.path.join(activations_dir, "activations-meta.json")) as f:
                    activations_meta = json.load(f)
                    activation_records = [
                        LayerOutputMeta(**obj) for obj in activations_meta
                    ]
            except FileNotFoundError:
                continue

            progression_dir = os.path.join(
                activation_output_basedir,
                platform_combination,
                index_subdir,
            )

            try:
                progression = torch.load(
                    os.path.join(progression_dir, f"progression-{platform_name}.pt"),
                    weights_only=True,
                    map_location=torch.device("cpu"),
                ).squeeze(1)
            except FileNotFoundError:
                continue

            yield (platform_combination, index_subdir, activation_records, progression)


def process_rundir(
    activations_dir, rundir: str, model_type: ModelType, *, sample_id=None
):
    if model_type == ModelType.VIT_B_32:
        model_name = "ViT"
    elif model_type == ModelType.RESNET18:
        model_name = "ResNet18"
    elif model_type == ModelType.EFFICIENTNET:
        model_name = "EfficientNet"
    else:
        assert False

    for platform_name in PLATFORMS_ABBR:
        y_diff_list = []

        for (
            platform_combination,
            index_subdir,
            activation_records,
            progression_logits,
        ) in get_index_dirs(
            os.path.join("output/backdoor-hooked", rundir),
            os.path.join(activations_dir, rundir),
            platform_name,
        ):
            progression = torch.softmax(progression_logits, dim=-1)

            label_B = torch.argmax(progression[0], dim=-1)
            label_A = torch.argmax(progression[-1], dim=-1)

            HARDCODED_RESNET18_LABELS_SAMPLE_0 = False
            if HARDCODED_RESNET18_LABELS_SAMPLE_0:
                assert model_type == ModelType.RESNET18
                label_B = torch.tensor([765])
                label_A = torch.tensor([857])

            assert label_B.item() != label_A.item()

            DIFF_BETWEEN_CLASSES = True
            if DIFF_BETWEEN_CLASSES:
                y_diff_list.append(
                    (progression[:, label_B] - progression[:, label_A]).numpy()
                )
            else:
                y_diff_list.append((progression - progression[0]).sum(dim=1).numpy())

        assert len(y_diff_list) > 0, "Empty diffs list"
        y_diffs = np.array(y_diff_list)

        print("y_diffs: ", y_diffs.shape, platform_name, model_name)

        if sample_id is not None:
            y_diffs = y_diffs[sample_id : sample_id + 1]

        LAYER_PRINT = False
        if LAYER_PRINT:
            for layer_id in range(y_diffs.shape[1]):
                y_diff = y_diffs[:, layer_id]
                if layer_id > 0 and layer_id < y_diffs.shape[1] - 1:
                    y_diff_prev = y_diffs[:, layer_id - 1]
                    value = (
                        (y_diff.astype(np.float64) - y_diff_prev.astype(np.float64))
                        ** 2
                    ).sum()
                    if value != 0:
                        print(
                            layer_id,
                            activation_records[layer_id - 1].layer_name,
                            activation_records[layer_id - 1].module_type,
                            value,
                        )
                    assert activation_records[layer_id - 1].layer_id == layer_id
                else:
                    print(
                        layer_id,
                        # y_diff
                    )

        Y = y_diffs[:, 1:] - y_diffs[:, :-1]
        # Y = y_diffs

        mu = np.mean(Y, axis=0)
        std = np.std(Y, axis=0)

        plt.plot(np.arange(mu.shape[0]), mu)
        plt.plot([0, mu.shape[0]], [0, 0], "k--")
        plt.fill_between(np.arange(mu.shape[0]), mu - std, mu + std, alpha=0.3)
        plt.title(f"{model_name} - {platform_name}")
        plt.show()

    plt.close()
    return


def activation_patching_experiment():
    activation_dir = os.path.join(
        "/shares/research/<anonymized for review>/activation-patching-level-1/base"
    )
    process_rundir(activation_dir, vit_full, ModelType.VIT_B_32)
    process_rundir(activation_dir, resnet_bit, ModelType.RESNET18)
    process_rundir(activation_dir, efficientnet_bit, ModelType.EFFICIENTNET)


activation_patching_experiment()
