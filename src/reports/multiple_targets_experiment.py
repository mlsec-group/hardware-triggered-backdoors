# %%

from typing import List, Set, Tuple, Dict
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

from reports.util import (
    ModelType,
    efficientnet_bit,
    efficientnet_bit_2_samples,
    efficientnet_bit_3_samples,
    efficientnet_bit_4_samples,
    efficientnet_bit_5_samples,
    resnet_bit,
    resnet_bit_2_samples,
    resnet_bit_3_samples,
    resnet_bit_4_samples,
    resnet_bit_5_samples,
    vit_conv_layer_only_full,
    vit_full_2_samples,
    vit_full_3_samples,
    vit_full_4_samples,
    vit_full_5_samples,
    extract_success,
    make_plots_beautfiul,
)

make_plots_beautfiul()


def multiple_targets_preprocess(platform_dict):
    print(platform_dict)

    success_rates = []

    for (platform_i, platform_j), record in sorted(
        platform_dict.items(), key=lambda x: x[1][0]
    ):
        success_rates.append(record[0] / record[1])

    return success_rates


def multiple_targets_study(run_paths: List[str], model_type: ModelType) -> None:
    # set(
    #     [
    #         ("a40", "rtx6000"),
    #         ("a100", "a100-mig40"),
    #         ("h100", "a40"),
    #         ("h100", "rtx6000"),
    #     ]
    # )

    # Collect data for boxplots
    data = [
        multiple_targets_preprocess(
            extract_success(run_path, model_type),
        )
        for run_path in run_paths
    ]

    x_positions = list(range(1, len(data) + 1))

    plt.figure()
    plt.plot(x_positions, [np.mean(np.array(dat)) for dat in data])
    # plt.violinplot(data, x_positions)

    plt.xlabel("Number of target samples")
    plt.xticks(x_positions)

    plt.ylabel("Probability of finding a malicious model")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))

    if model_type == ModelType.VIT_B_32:
        plt.title("Multi-target experiment (ViT)")
    if model_type == ModelType.EFFICIENTNET:
        plt.title("Multi-target experiment (EfficientNet)")
    if model_type == ModelType.RESNET18:
        plt.title("Multi-target experiment (ResNet)")

    plt.ylim([0, 1.05])
    plt.show()
    plt.clf()


multiple_targets_study(
    [
        vit_conv_layer_only_full,
        vit_full_2_samples,
        vit_full_3_samples,
        vit_full_4_samples,
        vit_full_5_samples,
    ],
    ModelType.VIT_B_32,
)

multiple_targets_study(
    [
        resnet_bit,
        resnet_bit_2_samples,
        resnet_bit_3_samples,
        resnet_bit_4_samples,
        resnet_bit_5_samples,
    ],
    ModelType.RESNET18,
)

multiple_targets_study(
    [
        efficientnet_bit,
        efficientnet_bit_2_samples,
        efficientnet_bit_3_samples,
        efficientnet_bit_4_samples,
        efficientnet_bit_5_samples,
    ],
    ModelType.EFFICIENTNET,
)
