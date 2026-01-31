# %%
# ################
# One vs rest table
# ################
import os
from typing import List

from reports.util import (
    PLATFORMS,
    PLATFORMS_ABBR,
    RESEARCH_DIR,
    extract_logs_dir,
    vit_one_vs_rest,
    efficientnet_one_vs_rest,
    resnet_one_vs_rest,
)


def collect_success_rates(run_dir: str, platform_abbrs: List[str]):
    """
    Returns a list of success rates aligned with PLATFORMS / PLATFORMS_ABBR.
    """
    run_path = os.path.join(RESEARCH_DIR, run_dir)
    rates = {}

    for i, abbr in enumerate(platform_abbrs):
        order = platform_abbrs[i:] + platform_abbrs[:i]
        platform_group_dir = "-".join(order)

        logs_dir = os.path.join(run_path, platform_group_dir, "logs")

        if not os.path.exists(logs_dir):
            print("SKIP", logs_dir)

        logdir_info = extract_logs_dir(logs_dir, only_if_done=True)

        avg = len(logdir_info.success_files) / len(logdir_info.job_dirs)
        rates[abbr] = (avg, len(logdir_info.success_files), len(logdir_info.job_dirs))

    return rates


# Collect results for each model
vit_rates = collect_success_rates(vit_one_vs_rest, PLATFORMS_ABBR)
resnet_rates = collect_success_rates(
    resnet_one_vs_rest, [abbr for abbr in PLATFORMS_ABBR if abbr != "a100-mig40"]
)
efficientnet_rates = collect_success_rates(efficientnet_one_vs_rest, PLATFORMS_ABBR)

# Print LaTeX table
print("GPU & \\textbf{ViT} & \\textbf{EfficientNet} & \\textbf{ResNet} \\\\")

for gpu, gpu_abbr in zip(PLATFORMS, PLATFORMS_ABBR):
    vit_avg, vit_ok, vit_total = vit_rates[gpu_abbr]
    eff_avg, eff_ok, eff_total = efficientnet_rates[gpu_abbr]
    res_avg, res_ok, res_total = resnet_rates.get(gpu_abbr, [float("nan")] * 3)

    print(
        f"{gpu} & "
        f"${vit_avg*100:.2f}\\%$ & "
        f"${eff_avg*100:.2f}\\%$ & "
        f"${res_avg*100:.2f}\\%$ \\\\"
    )
