# %%
# ################
# ################
# Experiment: Selected layer only experiment
# ################
# ################

import numpy as np

from reports.util import (
    ModelType,
    extract_success,
    vit_conv_layer_only_bit,
    vit_conv_layer_only_full,
    vit_full,
)


def selected_layer_only_preprocess(*platform_dicts):
    all_records = []

    records_per_platform = {}

    for platform_dict in platform_dicts:
        for (platform_i, platform_j), record in platform_dict.items():
            all_records.append(record)
            print(platform_i, platform_j, record)
            for platform in [platform_i, platform_j]:
                if platform not in records_per_platform:
                    records_per_platform[platform] = []
                records_per_platform[platform].append(record)

    all_records = np.array(all_records)
    ratios = all_records[:, 0] / all_records[:, 1]

    return np.mean(ratios), np.std(ratios)


def selected_layer_only_study():
    avg_0, std_0 = selected_layer_only_preprocess(
        extract_success(
            vit_conv_layer_only_bit,
            ModelType.VIT_B_32,
        )
    )
    avg_1, std_1 = selected_layer_only_preprocess(
        extract_success(
            vit_conv_layer_only_full,
            ModelType.VIT_B_32,
        )
    )
    avg_2, std_2 = selected_layer_only_preprocess(
        extract_success(
            vit_full,
            ModelType.VIT_B_32,
        )
    )
    print(f"Full: ${avg_2*100:.2f}\\% \\pm {std_2*100:.2f}\\%$")
    print(f"Full (one layer): ${avg_1*100:.2f}\\% \\pm {std_1*100:.2f}\\%$")
    print(f"Bit only (one layer): ${avg_0*100:.2f}\\% \\pm {std_0*100:.2f}\\%$")


selected_layer_only_study()
