import enum
import json
import os
import shutil
from dataclasses import dataclass
from itertools import combinations, product
from typing import List


def make_plots_beautfiul(*, force_tex=False):
    import matplotlib

    has_latex = shutil.which("latex") is not None
    if force_tex:
        assert has_latex

    matplotlib.rcParams["text.usetex"] = has_latex
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.family"] = "STIXGeneral"

    colors = {
        "green": "#798376",
        "darkgreen": "#484E46",
        "blue": "#41678B",
        "darkblue": "#2D4861",
        "orange": "#CB9471",
        "darkorange": "#965B37",
        "red": "#B65555",
        "mint": "#6AA56E",
        "grey": "#616161",
        "lightgrey": "#EEEEEEEE",
    }

    colors_idxes = list(colors)

    def format_running_time(running_time, long=False):
        if long:
            return f"{running_time // 3600:.0f}h {(running_time % 3600) // 60:.0f}m {(running_time % 60):.0f}s"
        return f"{running_time // 60:.0f}m {(running_time % 60):.0f}s"

    return colors


class ModelType(enum.Enum):
    VIT_B_32 = 0
    RESNET18 = 1
    EFFICIENTNET = 2


PLATFORMS_ABBR: List[str] = [
    "h100",
    "a40",
    "a100",
    "a100-mig40",
    "rtx6000",
    # "rtx3090", # == a40
]

PLATFORMS = [
    "H100",
    "A40",
    "A100",
    "A100-mig40",
    "RTX6000",
    # "RTX 3090", # == a40
]

RESEARCH_DIR = "/shares/research/<anonymized for review>/backdoor"

vit_full = "2025-10-25T10:37:02.070783_regal-dove"  # n_samples=100 do_crossover=false

vit_conv_layer_only_bit = "2025-11-29T16:18:22.281825_smiling-beaver"
vit_conv_layer_only_full = "2025-11-30T20:46:56.504730_sunny-bunny"

resnet_bit = (
    "2025-11-03T17:53:11.086550_thick-rhino"  # n_samples=100 do_crossover=false
)

efficientnet_bit = (
    "2025-10-29T13:25:52.233029_fuzzy-rhino"  # n_samples=100 do_crossover=false
)

vit_bit = "2025-12-03T12:37:34.147877_clunky-skunk"
vit_perm = "2025-12-04T17:47:38.306616_thin-yak"
vit_none = "2025-12-06T06:05:02.490226_rich-eagle"

# vit_full_2_samples = "2025-10-16T08:01:51.512621_fast-horse"
vit_full_2_samples = "2025-12-30T11:43:35.183108_noble-wolf"
# vit_full_3_samples = "2025-10-13T19:18:38.704040_fearless-duck"
vit_full_3_samples = "2026-01-01T02:46:40.547927_wet-urial"
# vit_full_4_samples = "2025-10-14T10:49:09.791394_smiling-duck"
vit_full_4_samples = "2026-01-02T11:33:00.331491_zealous-gnu"
vit_full_5_samples = "2026-01-03T22:38:10.282125_red-duck"

resnet_bit_2_samples = "2026-01-09T23:58:33.177507_thin-jaguar"
resnet_bit_3_samples = "2026-01-10T18:40:24.141464_xenophobic-yak"
resnet_bit_4_samples = "2026-01-11T10:52:23.635056_toxic-narwhal"
resnet_bit_5_samples = "2026-01-11T20:14:58.138196_kind-bee"

efficientnet_bit_2_samples = "2026-01-12T11:45:08.343702_broad-rat"
efficientnet_bit_3_samples = "2026-01-13T08:40:06.722833_small-toad"
efficientnet_bit_4_samples = "2026-01-14T06:51:00.513776_grey-quail"
efficientnet_bit_5_samples = "2026-01-15T20:55:04.141364_precious-yak"

vit_one_vs_rest = "2025-11-11T08:56:46.654918_dark-urial"
efficientnet_one_vs_rest = "2026-01-18T21:47:48.066698_funny-penguin"
# deprecated: a100 a100-mig40 is bitwise identical for resnet
# resnet_one_vs_rest = "2026-01-17T20:43:16.414104_literal-newt"
resnet_one_vs_rest = "2026-01-19T13:40:50.045101_noble-penguin"

vit_full_float16 = "2025-12-29T06:55:02.664675_slow-bear"  # (first layer only)
# vit_full_bfloat16 = "2025-12-29T01:09:18.497865_secret-oyster"  # (first layer only)
resnet_bit_float16 = "2026-01-05T11:53:15.439524_hungry-wolf"
# resnet_bit_bfloat16 = "2026-01-05T09:22:51.804198_smiling-fly"
efficientnet_bit_float16 = "2026-01-06T10:21:04.163634_blue-anaconda"
# efficientnet_bit_bfloat16 = "2026-01-06T23:39:07.538310_smashing-falcon"

resnet_bit_bfloat_no_threshold = "2026-01-27T19:13:15.266362_mean-ostrich"
# Obsoleted by: vit_full_bfloat_no_threshold
# vit_bit_bfloat_no_threshold = "2026-01-26T22:29:17.916454_ugly-duck"
efficientnet_bit_bfloat_no_threshold = "2026-01-27T13:34:32.241658_keen-zebra"
vit_full_bfloat_no_threshold = "2026-01-27T23:28:00.466887_cloudy-duck"

resnet_use_deterministic = "2026-01-07T16:31:06.834929_ostreaceous-gorilla"


@dataclass
class LogDirInfo:
    job_dirs: List[str]
    success_files: List[str]


def extract_logs_dir(logs_dir: str, *, only_if_done=False):
    job_dirs = []
    for dirname in os.listdir(logs_dir):
        if not dirname.startswith("Index"):
            continue

        job_dir = os.path.join(logs_dir, dirname)

        if only_if_done:
            if not os.path.exists(os.path.join(job_dir, "job-log.txt")):
                continue

        job_dirs.append(job_dir)

    success_files = [
        os.path.join(logs_dir, filename)
        for filename in os.listdir(logs_dir)
        if filename.startswith("bit_flip-")
        or filename.startswith("grad-")
        or filename.startswith("permute-")
    ]

    return LogDirInfo(job_dirs, success_files)


def extract_success(run_dir, model_type: ModelType, only_if_done=False):
    run_path = os.path.join(RESEARCH_DIR, run_dir)
    platform_dict = {}

    with open(os.path.join(run_path, "meta.json")) as f:
        meta = json.load(f)

    model_path = meta["args"]["model_path"]

    if model_type == ModelType.EFFICIENTNET:
        assert "models/imagenet/efficientnet_v2_s.pt" == model_path

    if model_type == ModelType.RESNET18:
        assert "models/imagenet/resnet18_model.pt" == model_path

    if model_type == ModelType.VIT_B_32:
        assert "models/imagenet/vit_b_32.pt" == model_path

    # n_samples = meta["args"].get("n_samples")

    for platform_i, platform_j in product(PLATFORMS_ABBR, PLATFORMS_ABBR):
        platform_dir = os.path.join(run_path, platform_i + "-" + platform_j)
        logs_dir = os.path.join(platform_dir, "logs")

        if not os.path.exists(platform_dir):
            continue

        if not os.path.exists(logs_dir):
            continue

        logdir_info = extract_logs_dir(logs_dir, only_if_done=only_if_done)

        platform_dict[platform_i, platform_j] = (
            len(logdir_info.success_files),
            len(logdir_info.job_dirs),
        )

    all_combinations = set(combinations(PLATFORMS_ABBR, 2))
    combinations_in_dict = set(platform_dict)
    assert (
        all_combinations == combinations_in_dict
    ), combinations_in_dict.symmetric_difference(combinations_in_dict)
    return {k: v for k, v in platform_dict.items() if len(v) > 0}


def main_preprocess(platform_dict):
    import numpy as np

    successes = {}

    for (platform_i, platform_j), record in platform_dict.items():
        for platform in [platform_i, platform_j]:
            if platform not in successes:
                successes[platform] = []
            successes[platform].append(record)

    results = {}
    for platform, records in successes.items():
        M = np.zeros((len(records), 2))
        for i, record in enumerate(records):
            M[i, 0] = record[0]
            M[i, 1] = record[1]

        if M.shape[0] == 0:
            ratios = []
        else:
            ratios = M[:, 0] / M[:, 1]

        results[platform] = [np.mean(ratios), np.std(ratios)]

    return results
