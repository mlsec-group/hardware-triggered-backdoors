# %%
# ################
# ################
# Experiment: Robustness input
# ################
# ################

import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
import pickle
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from reports.util import PLATFORMS_ABBR, make_plots_beautfiul

COLORS = make_plots_beautfiul()


@dataclass
class Rundir:
    path: str
    name: str


def calc(iterator):
    ulp_to_ratios = defaultdict(list)
    for ulp, platform_i, platform_j, index, y_pred_i, y_pred_j in iterator:
        # Safety check: lists must be same length
        if len(y_pred_i) != len(y_pred_j):
            raise ValueError(
                f"Prediction lists differ in length for (ulp={ulp}, index={index})."
            )

        if len(y_pred_i) == 0:
            continue  # avoid division by zero

        equal_count = sum(a != b for a, b in zip(y_pred_i, y_pred_j))
        ratio = equal_count / len(y_pred_i)

        ulp_to_ratios[ulp].append(ratio)
    return ulp_to_ratios


def plot_ulp_similarity(ax0, ax1, iterator, *, label: str, color: str):
    """
    iterator yields tuples:
        (ulp: int,
         platform_i: str,
         platform_j: str,
         index: int,
         y_pred_i: List[int],
         y_pred_j: List[int])

    For each ulp, compute similarity ratios for all entries:
         ratio = (# matching elements between y_pred_i and y_pred_j) / len(list)
    Then aggregate (mean, std) per ulp and plot.
    """

    hashname = hashlib.sha256(label.encode()).hexdigest()
    filename = f"/tmp/robustness_input-{hashname}.pkl"

    # Store all ratios per ULP
    try:
        with open(filename, "rb") as f:
            ulp_to_ratios = pickle.load(f)
        print("Loaded")
    except (FileNotFoundError, EOFError):
        ulp_to_ratios = calc(iterator)
        with open(filename, "wb") as f:
            pickle.dump(ulp_to_ratios, f)
            print("Cached")

    # Convert to sorted arrays
    ulps = np.array(sorted(ulp_to_ratios.keys()))
    means = np.array([np.mean(ulp_to_ratios[u]) for u in ulps])
    # stds = np.array(
    #     [np.std(ulp_to_ratios[u]) if len(ulp_to_ratios[u]) > 1 else 0 for u in ulps]
    # )

    sems = np.array(
        [
            (
                np.std(ulp_to_ratios[u]) / np.sqrt(len(ulp_to_ratios[u]))
                if len(ulp_to_ratios[u]) > 1
                else 0
            )
            for u in ulps
        ]
    )

    x0 = 0
    y0 = 1

    # ---- Left axis (linear, just x=0) ----
    ax0.plot([x0], [y0], ".", color="black")
    ax0.set_xlim(-0.1, 1)
    ax0.set_xticks([0])
    ax0.set_xlabel("ULP")

    # ---- Right axis (log scale) ----
    ax1.fill_between(ulps, means - sems, means + sems, alpha=0.3)
    ax1.plot(ulps, means, label=label, color=color)
    ax1.set_xscale("log")
    ax1.set_xlim(min(ulps), max(ulps))

    # ---- Connect visually (optional but recommended) ----
    ax0.plot([x0, ulps[0]], [y0, means[0]], linestyle="--", alpha=0.5, color=color)

    ax0.spines.right.set_visible(False)
    ax1.spines.left.set_visible(False)

    ax0.tick_params(right=False)
    ax1.tick_params(left=False)

    # ---- Axis break marks ----
    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    ax0.plot([1, 1], [0, 1], transform=ax0.transAxes, **kwargs)
    ax1.plot([0, 0], [0, 1], transform=ax1.transAxes, **kwargs)

    # ---- Labels & styling ----
    ax0.set_ylabel("Mean similarity ratio")
    ax0.grid(False)
    ax1.set_xlabel("ULP")
    ax1.legend()
    ax1.grid(False)

    for ulp, ratios in ulp_to_ratios.items():
        print(ulp, len(ratios))


def robustness_input_experiment(
    rundirs: List[Rundir], ulps=[1, 10, 100, 1_000, 10_000, 100_000]
):
    def process_indexdir(index_dir: str, platform_i: str, platform_j: str):
        with open(os.path.join(index_dir, f"y-{platform_i}.txt")) as f:
            y_pred_i = [int(line.strip()) for line in f.readlines() if len(line) > 0]

        with open(os.path.join(index_dir, f"y-{platform_j}.txt")) as f:
            y_pred_j = [int(line.strip()) for line in f.readlines() if len(line) > 0]

        assert len(y_pred_i) == len(y_pred_j)

        return y_pred_i, y_pred_j

    def process_platform_combination(
        platform_dir: str, platform_i: str, platform_j: str
    ):
        for index_subdir in os.listdir(platform_dir):
            index = int(index_subdir[len("Index-") :])

            try:
                y_pred_i, y_pred_j = process_indexdir(
                    os.path.join(platform_dir, index_subdir), platform_i, platform_j
                )
            except FileNotFoundError:
                continue

            yield index, y_pred_i, y_pred_j

    def process_ulpdir(ulpdir: str):
        for platform_i, platform_j in product(PLATFORMS_ABBR, PLATFORMS_ABBR):
            platform_dir = os.path.join(ulpdir, f"{platform_i}-{platform_j}")
            if not os.path.exists(platform_dir):
                continue

            for index, y_pred_i, y_pred_j in process_platform_combination(
                platform_dir, platform_i, platform_j
            ):
                yield platform_i, platform_j, index, y_pred_i, y_pred_j

    def process_rundir(rundir: Rundir):
        for ulp in ulps:
            for platform_i, platform_j, index, y_pred_i, y_pred_j in process_ulpdir(
                os.path.join(rundir.path, f"ulp-{ulp}")
            ):
                yield ulp, platform_i, platform_j, index, y_pred_i, y_pred_j

    fig, (ax0, ax1) = plt.subplots(
        1, 2, sharey=True, gridspec_kw={"width_ratios": [1, 10]}, figsize=(8, 4)
    )
    fig.subplots_adjust(wspace=0.025)  # adjust space between Axes
    fig.suptitle("Similarity ratio vs. ULP (with standard error)")

    for rundir, color in zip(rundirs, COLORS):
        plot_ulp_similarity(
            ax0,
            ax1,
            process_rundir(rundir),
            label=rundir.name,
            color=color,
        )
    # plt.tight_layout()
    plt.show()
    plt.legend()
    plt.clf()


robustness_input_experiment(
    [
        Rundir(
            "output/experiments/robustness-input/2025-10-25T10:37:02.070783_regal-dove",
            "ViT",
        ),
        Rundir(
            "output/experiments/robustness-input/2025-10-29T13:25:52.233029_fuzzy-rhino",
            "EfficientNet",
        ),
        Rundir(
            "output/experiments/robustness-input/2025-11-03T17:53:11.086550_thick-rhino",
            "ResNet18",
        ),
    ]
)
