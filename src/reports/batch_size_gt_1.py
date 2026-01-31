# %%

import json
import os
from collections import Counter
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

PLATFORMS_ABBR = [
    "a40",
    "a100",
    "a100-mig40",
    "rtx6000",
    "h100",
    # "rtx3090", # == a40
]


def list_results(rundir: str):
    missing = 0

    for platformA, platformB in product(PLATFORMS_ABBR, PLATFORMS_ABBR):
        platform_subdir = f"{platformA}-{platformB}"
        platform_dir = os.path.join(rundir, platform_subdir)

        if not os.path.exists(platform_dir):
            continue

        for index in range(100):
            index_subdir = f"Index-{index}"
            index_dir = os.path.join(platform_dir, index_subdir)

            if not os.path.exists(index_dir):
                continue

            try:
                with open(os.path.join(index_dir, f"meta-{platformA}.json")) as f:
                    metaA = json.load(f)

                with open(os.path.join(index_dir, f"meta-{platformB}.json")) as f:
                    metaB = json.load(f)

                with open(os.path.join(index_dir, f"labels-{platformA}.json")) as f:
                    labelsA = json.load(f)

                with open(os.path.join(index_dir, f"labels-{platformB}.json")) as f:
                    labelsB = json.load(f)
            except FileNotFoundError as e:
                missing += 1
                print(e)
                continue

            init_label = metaA["init_label"]

            assert metaA["init_label"] == metaB["init_label"]
            assert metaA["label_A"] == metaB["label_B"]
            assert metaA["label_B"] == metaB["label_A"]

            # label_B means "our label here" (not a good naming, sorry)
            assert metaA["label_B"] == labelsA[0][0]
            assert metaB["label_B"] == labelsB[0][0]

            assert len(labelsA) == len(labelsB) == 256

            yield platformA, platformB, index, init_label, labelsA, labelsB

    print("MISSING: ", missing)


def match_probability(A, B):
    cA = Counter(A)
    cB = Counter(B)
    nA, nB = len(A), len(B)
    classes = set(cA) | set(cB)

    return sum((cA[c] / nA) * (cB[c] / nB) for c in classes)


def calc(results):
    results = list(results)

    matrix = {
        (platformA, platformB): {
            "X": [batch_size for batch_size in range(1, 256 + 1)],
            "probas": [[] for _ in range(1, 256 + 1)],
            "total": 0,
        }
        for platformA, platformB in product(PLATFORMS_ABBR, PLATFORMS_ABBR)
    }

    for platformA, platformB, index, init_label, labelsA, labelsB in results:
        matrix[platformA, platformB]["total"] += 1
        matrix[platformB, platformA]["total"] += 1

        for batch_size in range(1, 256 + 1):
            batch_size_index = batch_size - 1

            proba = 1 - match_probability(
                labelsA[batch_size_index], labelsB[batch_size_index]
            )

            matrix[platformA, platformB]["probas"][batch_size_index].append(proba)
            matrix[platformB, platformA]["probas"][batch_size_index].append(proba)

    #    for batch_size_index in range(10):
    #        batch_size = batch_size_index + 1
    #        print(batch_size, matrix["h100", "a100"]["probas"][batch_size_index])

    return matrix


def main():
    rel_dir = "/shares/research/<anonymized for review>/experiments/batch_gt_1/"

    vit_full = "2025-10-25T10:37:02.070783_regal-dove"
    efficientnet_bit = "2025-10-29T13:25:52.233029_fuzzy-rhino"
    resnet_bit = "2025-11-03T17:53:11.086550_thick-rhino"

    matrix_vit = calc(list_results(os.path.join(rel_dir, vit_full)))
    matrix_resnet = calc(list_results(os.path.join(rel_dir, resnet_bit)))
    matrix_efficientnet = calc(list_results(os.path.join(rel_dir, efficientnet_bit)))

    if True:
        fig, axes = plt.subplots(
            len(PLATFORMS_ABBR), len(PLATFORMS_ABBR), figsize=(25, 25)
        )

        for label, matrix in zip(
            ["vit", "resnet", "efficientnet"],
            [matrix_vit, matrix_resnet, matrix_efficientnet],
        ):
            for (platformA_i, platformA), (platformB_j, platformB) in product(
                enumerate(PLATFORMS_ABBR), enumerate(PLATFORMS_ABBR)
            ):
                if platformA == platformB:
                    continue

                cell = matrix[platformA, platformB]
                total = cell["total"]

                if total == 0:
                    print(platformA, platformB, label)
                    axes[platformA_i, platformB_j].plot([], [])
                else:
                    probas = np.array(cell["probas"])
                    mean = np.mean(probas, axis=-1)

                    axes[platformA_i, platformB_j].plot(cell["X"], mean, label=label)
                    axes[platformA_i, platformB_j].set_ylim([0, 1.1])

        plt.show()
        plt.clf()

    plt.figure()

    def plot_combination(m, p1, p2, *, color=None):
        cell = m[p1, p2]
        plt.plot(cell["X"], np.mean(cell["probas"], axis=-1), color=color)

    # Different convolutions, conv implementation invariant of batch size
    plot_combination(matrix_vit, "a40", "a100", color="red")
    plot_combination(matrix_vit, "a40", "a100-mig40", color="red")
    plot_combination(matrix_vit, "rtx6000", "a100", color="red")
    plot_combination(matrix_vit, "rtx6000", "a100-mig40", color="red")

    #
    plot_combination(matrix_vit, "a40", "rtx6000", color="blue")
    plot_combination(matrix_vit, "a40", "h100", color="blue")
    plot_combination(matrix_vit, "rtx6000", "h100", color="blue")

    #
    plot_combination(matrix_vit, "a100", "a100-mig40", color="green")
    plot_combination(matrix_vit, "a100", "h100", color="green")
    plot_combination(matrix_vit, "a100-mig40", "h100", color="green")

    plt.ylabel("V_batch")
    plt.xlabel("batch size k")


main()
