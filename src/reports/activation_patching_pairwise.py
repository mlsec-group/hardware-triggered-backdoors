# %%
# ################
# ################
# Experiment: Activation patching pair-wise
# ################
# ################


import os
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from reports.util import PLATFORMS_ABBR, ModelType, vit_full


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


def get_index_dirs_pairwise(
    input_basedir: str, activation_output_basedir: str, platform_A: str, platform_B
):
    platform_combination = f"{platform_A}-{platform_B}"
    platform_dir = os.path.join(input_basedir, platform_combination)

    if not os.path.exists(platform_dir):
        return

    for index_subdir in sorted(os.listdir(platform_dir)):
        progression_dir = os.path.join(
            activation_output_basedir,
            platform_combination,
            index_subdir,
        )

        progressions_found = [True, True]

        try:
            progression_A = torch.load(
                os.path.join(progression_dir, f"progression-{platform_A}.pt"),
                weights_only=True,
                map_location=torch.device("cpu"),
            ).squeeze(1)
        except FileNotFoundError as e:
            progressions_found[0] = False

        try:
            progression_B = torch.load(
                os.path.join(progression_dir, f"progression-{platform_B}.pt"),
                weights_only=True,
                map_location=torch.device("cpu"),
            ).squeeze(1)
        except FileNotFoundError as e:
            progressions_found[1] = False

        if progressions_found[0] and progressions_found[1]:
            yield (platform_combination, index_subdir, progression_A, progression_B)
        elif progressions_found[0]:
            print(
                "[WARNING] Not found: ",
                os.path.join(progression_dir, f"progression-{platform_B}.pt"),
            )
        elif progressions_found[1]:
            print(
                "[WARNING] Not found: ",
                os.path.join(progression_dir, f"progression-{platform_A}.pt"),
            )
        else:
            print(
                "[WARNING] Not found: ",
                os.path.join(progression_dir, f"progression-{platform_A}.pt"),
                os.path.join(progression_dir, f"progression-{platform_B}.pt"),
            )


def process_rundir_pairwise(activations_dir, rundir: str, model_type: ModelType):
    if model_type == ModelType.VIT_B_32:
        model_name = "ViT"
    elif model_type == ModelType.RESNET18:
        model_name = "ResNet18"
    elif model_type == ModelType.EFFICIENTNET:
        model_name = "EfficientNet"
    else:
        assert False

    platforms = [
        PLATFORMS_ABBR[0],
        PLATFORMS_ABBR[3],
        PLATFORMS_ABBR[4],
        PLATFORMS_ABBR[1],
        PLATFORMS_ABBR[2],
    ]

    fig, axes = plt.subplots(len(platforms), len(platforms))
    fig.set_figheight(40)
    fig.set_figwidth(40)
    fig.suptitle(model_name, fontsize=16, y=0.98)

    m = len(platforms)
    n = len(platforms)

    row_labels = platforms
    col_labels = platforms

    # Column labels (top)
    for j, label in enumerate(col_labels):
        fig.text(
            x=(j + 0.5) / m,
            y=0.95,
            s=label,
            ha="center",
            va="center",
            fontsize=48,
            fontweight="bold",
        )

    # Row labels (left)
    for i, label in enumerate(row_labels):
        fig.text(
            x=0.05,
            y=1 - (i + 0.5) / n,
            s=label,
            ha="center",
            va="center",
            fontsize=48,
            fontweight="bold",
            rotation=90,
        )

    for (i_A, platform_A), (j_B, platform_B) in product(
        enumerate(platforms), enumerate(platforms)
    ):
        if platform_A == platform_B:
            continue

        for (
            platform_combination,
            index_subdir,
            progression_logits_A,
            progression_logits_B,
        ) in get_index_dirs_pairwise(
            os.path.join("output/backdoor-hooked", rundir),
            os.path.join(activations_dir, rundir),
            platform_A,
            platform_B,
        ):
            progression_A = torch.softmax(progression_logits_A, dim=-1)
            progression_B = torch.softmax(progression_logits_B, dim=-1)

            assert torch.all(progression_A[0] == progression_B[-1])
            assert torch.all(progression_B[0] == progression_A[-1])

            label_A = torch.argmax(progression_A[0], dim=-1)
            label_B = torch.argmax(progression_A[-1], dim=-1)

            diffs_A = (progression_A[:, label_A] - progression_A[:, label_B]).numpy()
            diffs_B = (progression_B[:, label_B] - progression_B[:, label_A]).numpy()
            axes[i_A, j_B].plot(np.arange(diffs_A.shape[0]), diffs_A, label="diffA")
            axes[j_B, i_A].plot(np.arange(diffs_B.shape[0]), diffs_B, label="diffB")
            # plt.title(
            #     f"{platform_A} ({label_A}) vs {platform_B} ({label_B}) ({index_subdir}) on {model_name}"
            # )

    plt.show()


def detailled_view(
    activations_dir,
    rundir: str,
    model_type: ModelType,
    platform_A: str,
    platform_B: str,
    index: int,
):
    if model_type == ModelType.VIT_B_32:
        model_name = "ViT"
    elif model_type == ModelType.RESNET18:
        model_name = "ResNet18"
    elif model_type == ModelType.EFFICIENTNET:
        model_name = "EfficientNet"
    else:
        assert False

    input_basedir = os.path.join("output/backdoor-hooked", rundir)
    activation_output_basedir = os.path.join(activations_dir, rundir)

    platform_combination = f"{platform_A}-{platform_B}"
    platform_dir = os.path.join(input_basedir, platform_combination)

    if not os.path.exists(platform_dir):
        raise FileNotFoundError()

    index_subdir = f"Index-{index}"
    progression_dir = os.path.join(
        activation_output_basedir,
        platform_combination,
        index_subdir,
    )

    try:
        progression_logits_A = torch.load(
            os.path.join(progression_dir, f"progression-{platform_A}.pt"),
            weights_only=True,
            map_location=torch.device("cpu"),
        ).squeeze(1)
    except FileNotFoundError as e:
        raise e

    try:
        progression_logits_B = torch.load(
            os.path.join(progression_dir, f"progression-{platform_B}.pt"),
            weights_only=True,
            map_location=torch.device("cpu"),
        ).squeeze(1)
    except FileNotFoundError as e:
        raise e

    progression_A = torch.softmax(progression_logits_A, dim=-1)
    progression_B = torch.softmax(progression_logits_B, dim=-1)

    assert torch.all(progression_A[0] == progression_B[-1])
    assert torch.all(progression_B[0] == progression_A[-1])

    label_A = torch.argmax(progression_A[0], dim=-1)
    label_B = torch.argmax(progression_A[-1], dim=-1)

    diffs_A = (progression_A[:, label_A] - progression_A[:, label_B]).numpy()
    diffs_B = (progression_B[:, label_B] - progression_B[:, label_A]).numpy()
    plt.plot(np.arange(diffs_A.shape[0]), diffs_A, label="diffA")
    plt.plot(np.arange(diffs_B.shape[0]), diffs_B, label="diffB")
    plt.show()


def activation_patching_pairwise_experiment():
    activation_dir = "output/experiments/activation-patching-level-1/base"
    # process_rundir_pairwise(activation_dir, vit_full, ModelType.VIT_B_32)
    # process_rundir_pairwise(activation_dir, resnet_bit, ModelType.RESNET18)
    # process_rundir_pairwise(activation_dir, efficientnet_bit, ModelType.EFFICIENTNET)

    for i in range(1):
        try:
            detailled_view(
                activation_dir, vit_full, ModelType.VIT_B_32, "a40", "a100", i
            )
        except FileNotFoundError:
            pass


activation_patching_pairwise_experiment()
