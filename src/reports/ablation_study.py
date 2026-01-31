# %%
#
# Experiment: Doing an ablation study, where we compare the full method, the
# bit-flip only method, the permutation only-method, and the base method.
#

import numpy as np

from reports.util import (
    ModelType,
    extract_success,
    vit_full,
    vit_bit,
    vit_perm,
    vit_none,
)


def ablation_preprocess(*platform_dicts):
    all_records = []

    records_per_platform = {}

    for platform_dict in platform_dicts:
        for (platform_i, platform_j), record in platform_dict.items():
            all_records.append(record)
            for platform in [platform_i, platform_j]:
                if platform not in records_per_platform:
                    records_per_platform[platform] = []
                records_per_platform[platform].append(record)

    all_records = np.array(all_records)
    ratios = all_records[:, 0] / all_records[:, 1]

    return np.mean(ratios), np.std(ratios)


def ablation_study():
    avg_0, std_0 = ablation_preprocess(extract_success(vit_full, ModelType.VIT_B_32))
    avg_2, std_2 = ablation_preprocess(extract_success(vit_bit, ModelType.VIT_B_32))
    avg_3, std_3 = ablation_preprocess(extract_success(vit_perm, ModelType.VIT_B_32))
    avg_4, std_4 = ablation_preprocess(extract_success(vit_none, ModelType.VIT_B_32))

    print(f"${avg_0*100:.2f}\\% \\pm {std_0*100:.2f}\\%$", end=" & ")
    print(f"${avg_2*100:.2f}\\% \\pm {std_2*100:.2f}\\%$", end=" & ")
    print(f"${avg_3*100:.2f}\\% \\pm {std_3*100:.2f}\\%$", end=" & ")
    print(f"${avg_4*100:.2f}\\% \\pm {std_4*100:.2f}\\%$", end="\\\\")


ablation_study()
