# %%
#

import os
import pickle
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import tqdm
from torch.utils.data import DataLoader

from common.util import hash_tensor
from datasets.loader import get_dataset_loader
from experiments.natural_chimeras.imagenet_labels import IMAGENET_LABELS

N_BATCHES = 11  # should be 12
BATCH_SIZE = 4096


def plot_heatmap(
    ax, fig, heatmap, annot_heatmap, title, x_labels, y_labels, vmin, vmax
):
    # Plot the heatmap with "hot" colormap
    sns.heatmap(
        heatmap,
        annot=annot_heatmap,
        square=True,
        xticklabels=True,
        yticklabels=True,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
    )

    # ax.set_aspect(1 / 1.41)  # Enforce height being 3x smaller than width

    # Customize tick colors
    ax.set_xticks([i + 0.5 for i in range(len(x_labels))])
    ax.set_xticklabels(x_labels, rotation=90)
    # ax.set_xticks([])
    ax.set_yticks([i + 0.5 for i in range(len(y_labels))])
    ax.set_yticklabels(y_labels, rotation=0)

    # Customize title and labels
    ax.set_title(title)
    # ax.set_xlabel("")
    # ax.set_ylabel("")

    # Adjust layout for better fitting
    # fig.tight_layout()


# %%
def gpu_verify_hashes(model_name):
    # This function verifies that the (unnormalized) input data from the test
    # loader is bit-identical to the input that we provided to the model. For
    # this we compute hash(normalize(x)) and compare it against the file x.hash

    basedir = "output/experiments/natural_chimeras/" + model_name
    platforms = sorted(os.listdir(basedir))

    loader = get_dataset_loader("imagenet", "data")
    test_dataset = loader.load_test()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for i, (x, y) in zip(range(N_BATCHES), test_loader):
        x_norm = loader.normalize(x)

        for platform in platforms:
            for run_name in ["run-0", "run-1"]:
                platform_dir = os.path.join(basedir, platform, run_name)

                filepath = os.path.join(platform_dir, f"{i}.pt")
                x_hash_path = filepath + "-x.hash"

                with open(x_hash_path) as f:
                    x_hash = f.read().strip()
                    assert x_hash == hash_tensor(x_norm).hex()

                x_norm_path = os.path.join(platform_dir, f"{i}_norm.pt")
                torch.save(x_norm, x_norm_path)

    return


gpu_verify_hashes("vit_b_32")
gpu_verify_hashes("efficientnet_v2_s")
gpu_verify_hashes("resnet18_model")


# %%
# Find naturally occuring chimera samples in datasets
def gpu_natural_chimeras(ax, model_name, *, no_x_labels=False, no_y_labels=False):
    basedir = "output/experiments/natural_chimeras/" + model_name
    platforms = sorted(os.listdir(basedir))

    # This is used to assert that we used the exact same input on all of
    # the models
    x_hashes = {i: set() for i in range(N_BATCHES)}
    predictions = {platform: [] for platform in platforms}

    for platform in platforms:
        platform_dir = os.path.join(basedir, platform, "run-0")

        for i in range(N_BATCHES):
            filepath = os.path.join(platform_dir, f"{i}.pt")
            x_hash_path = filepath + "-x.hash"

            with open(x_hash_path) as f:
                x_hashes[i].add(f.read())

            y = torch.load(
                filepath, map_location=torch.device("cpu"), weights_only=True
            )
            top2 = torch.topk(y, k=2)
            assert y.shape[0] == BATCH_SIZE
            predictions[platform].append(top2)

    assert all(len(v) == 1 for v in x_hashes.values())

    loader = get_dataset_loader("imagenet", "data")

    test_dataset = loader.load_test()

    heatmap = np.zeros((len(platforms), len(platforms)))
    for (i, platform_i), (j, platform_j) in combinations(enumerate(platforms), 2):
        V_i = torch.vstack([top2.values for top2 in predictions[platform_i]])
        V_j = torch.vstack([top2.values for top2 in predictions[platform_j]])

        P_i = torch.vstack([top2.indices for top2 in predictions[platform_i]])
        P_j = torch.vstack([top2.indices for top2 in predictions[platform_j]])

        assert P_i.shape == (45056, 2)

        mask = P_i[:, 0] != P_j[:, 0]
        c = mask.count_nonzero()
        heatmap[i, j] = c
        heatmap[j, i] = c

        if c > 0:
            for index in torch.argwhere(mask):
                index = index.item()
                batch_index = index // BATCH_SIZE
                offset_index = index % BATCH_SIZE

                assert (
                    batch_index * BATCH_SIZE + offset_index == index
                ), f"{batch_index} * {BATCH_SIZE} + {offset_index} == {batch_index * BATCH_SIZE + offset_index} == {index}"

                x, _ = test_dataset[index]

                top1_label_i = IMAGENET_LABELS[P_i[index, 0].item()]
                top2_label_i = IMAGENET_LABELS[P_i[index, 1].item()]

                top1_label_j = IMAGENET_LABELS[P_j[index, 0].item()]
                top2_label_j = IMAGENET_LABELS[P_j[index, 1].item()]

                print(x.shape)
                print(
                    top1_label_i,
                    top2_label_i,
                    P_i[index],
                    V_i[index],
                )
                print(
                    top1_label_j,
                    top2_label_j,
                    P_j[index],
                    V_j[index],
                )
                fig2 = plt.figure()
                fig2.add_subplot().imshow(x.permute(1, 2, 0))
                plt.figure(fig2)
                plt.title(
                    f"{platform_i} vs {platform_j} \n '{top1_label_i}' vs '{top1_label_j}'"
                )
                plt.show()
                plt.close(fig2)

                break

    plot_heatmap(
        ax,
        fig,
        heatmap,
        heatmap,
        model_name,
        [""] * len(platforms) if no_x_labels else platforms,
        [""] * len(platforms) if no_y_labels else platforms,
        0,
        1,
    )


fig, axes = plt.subplots(1, 3)
gpu_natural_chimeras(axes[0], "vit_b_32")
gpu_natural_chimeras(axes[1], "efficientnet_v2_s", no_y_labels=True)
gpu_natural_chimeras(axes[2], "resnet18_model", no_y_labels=True)
plt.figure(fig)
plt.subplots_adjust(top=1.3)
plt.suptitle("Naturally occurring chimeras", ha="center", va="bottom")
plt.show()
plt.clf()


# %%
#
def gpu_prediction_bitwise_overlap(
    ax, model_name, *, no_x_labels=False, no_y_labels=False
):
    basedir = "output/experiments/natural_chimeras/" + model_name
    platforms = sorted(os.listdir(basedir))

    # This is used to assert that we used the exact same input on all of
    # the models
    x_hashes = {i: set() for i in range(N_BATCHES)}
    y_hashes_singles = {platform: [] for platform in platforms}

    for platform in platforms:
        platform_dir = os.path.join(basedir, platform, "run-0")

        for i in range(N_BATCHES):
            filepath = os.path.join(platform_dir, f"{i}.pt")
            x_hash_path = filepath + "-x.hash"

            with open(x_hash_path) as f:
                x_hashes[i].add(f.read())

            y = torch.load(
                filepath, map_location=torch.device("cpu"), weights_only=True
            )

            for y_ in y:
                y_hashes_singles[platform].append(hash_tensor(y_))

    assert all(len(v) == 1 for v in x_hashes.values())

    heatmap = np.zeros((len(platforms), len(platforms)), dtype=np.int32)
    for (i, platform_i), (j, platform_j) in combinations(enumerate(platforms), 2):
        equal_count = sum(
            [
                o1 == o2
                for o1, o2 in zip(
                    y_hashes_singles[platform_i], y_hashes_singles[platform_j]
                )
            ]
        )
        heatmap[i, j] = equal_count / len(y_hashes_singles[platform_i])
        heatmap[j, i] = equal_count / len(y_hashes_singles[platform_i])

        # pred_i = predictions[platform_i][mask].tolist()
        # pred_j = predictions[platform_j][mask].tolist()

        # print(platform_i, "vs", platform_j)
        # print(*list(zip(pred_i, pred_j)))
        # print()

    plot_heatmap(
        ax,
        fig,
        heatmap,
        heatmap,
        model_name,
        [""] * len(platforms) if no_x_labels else platforms,
        [""] * len(platforms) if no_y_labels else platforms,
        0,
        1,
    )


fig, axes = plt.subplots(1, 3)
gpu_prediction_bitwise_overlap(axes[0], "vit_b_32")
gpu_prediction_bitwise_overlap(axes[1], "efficientnet_v2_s", no_y_labels=True)
gpu_prediction_bitwise_overlap(axes[2], "resnet18_model", no_y_labels=True)
plt.subplots_adjust(top=1.3)
plt.suptitle("Bit-identic calculations", ha="center", va="bottom")
plt.show()
plt.clf()


# %%
# Verify that GPU calculations are deterministic across different runs
def gpu_deterministic(model_name):
    basedir = "output/experiments/natural_chimeras/" + model_name
    platforms = sorted(os.listdir(basedir))

    # This is used to assert that we used the exact same input on all of
    # the models
    x_hashes = {i: set() for i in range(N_BATCHES)}
    y_hashes = {(platform, run_id): [] for platform in platforms for run_id in range(2)}

    for run_id in range(2):
        for platform in platforms:
            platform_dir = os.path.join(basedir, platform, f"run-{run_id}")

            for i in range(N_BATCHES):
                filepath = os.path.join(platform_dir, f"{i}.pt")
                x_hash_path = filepath + "-x.hash"
                y_hash_path = filepath + ".hash"

                with open(x_hash_path) as f:
                    x_hashes[i].add(f.read())

                with open(y_hash_path) as f:
                    y_hashes[(platform, run_id)].append(f.read())

    assert all(len(v) == 1 for v in x_hashes.values())

    for platform in platforms:
        assert y_hashes[(platform, 0)] == y_hashes[(platform, 1)]
    print("Deterministic")


gpu_deterministic("vit_b_32")
gpu_deterministic("efficientnet_v2_s")
gpu_deterministic("resnet18_model")


# %%
# Calculate the difference (on softmax) between each prediction between platforms
def gpu_floating_point_differences(
    ax, model_name, *, no_x_labels=False, no_y_labels=False
):
    basedir = "output/experiments/natural_chimeras/" + model_name
    platforms = sorted(os.listdir(basedir))

    # This is used to assert that we used the exact same input on all of
    # the models

    try:
        with open(f"/tmp/predictions-{model_name}.pkl", "rb") as f:
            predictions = pickle.load(f)
    except FileNotFoundError:
        x_hashes = {i: set() for i in range(N_BATCHES)}
        predictions = {platform: [] for platform in platforms}

        loader = get_dataset_loader("imagenet", "data")
        test_dataset = loader.load_test()
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        for i, (_, label_exp) in tqdm.tqdm(
            zip(range(N_BATCHES), test_loader), total=N_BATCHES
        ):
            print("BATCH: ", i, flush=True)
            for platform in platforms:
                platform_dir = os.path.join(basedir, platform, "run-0")

                filepath = os.path.join(platform_dir, f"{i}.pt")
                x_hash_path = filepath + "-x.hash"

                with open(x_hash_path) as f:
                    x_hash = f.read().strip()
                    x_hashes[i].add(x_hash)

                y_logits = torch.load(
                    filepath, map_location=torch.device("cpu"), weights_only=True
                )
                y = torch.nn.functional.softmax(y_logits, dim=-1)
                assert y.shape[0] == BATCH_SIZE

                pred_exp = y[torch.arange(label_exp.shape[0]), label_exp]
                predictions[platform].append(pred_exp)

    with open(f"/tmp/predictions-{model_name}.pkl", "wb") as f:
        pickle.dump(predictions, f)

    # assert all(len(v) == 1 for v in x_hashes.values())

    platform_differences = {}
    for (i, platform_i), (j, platform_j) in combinations(enumerate(platforms), 2):
        V_i = torch.hstack([pred for pred in predictions[platform_i]])
        V_j = torch.hstack([pred for pred in predictions[platform_j]])

        platform_differences[(platform_i, platform_j)] = V_i - V_j

    for (platform_i, platform_j), differences in platform_differences.items():
        if differences.min() == differences.max() == 0:
            continue

        if platform_i != "A40" and platform_j != "A40":
            continue

        print(differences)

        bins = np.linspace(-0.012, 0.012, 50)
        bins = (
            (-np.logspace(-2, -7, 25)).tolist() + [0] + np.logspace(-7, -2, 25).tolist()
        )
        ax.hist(
            differences,
            bins=bins,
            log=True,
            alpha=0.3,
            label=platform_i + " vs " + platform_j,
        )
    ax.set_title(model_name)


fig, axes = plt.subplots(1, 3)
gpu_floating_point_differences(axes[0], "vit_b_32")
gpu_floating_point_differences(axes[1], "efficientnet_v2_s", no_y_labels=True)
gpu_floating_point_differences(axes[2], "resnet18_model", no_y_labels=True)
plt.figure(fig)
plt.legend()
# plt.subplots_adjust(top=1.3)
plt.suptitle("Floating-point differences between platforms", ha="center", va="bottom")
plt.show()
plt.clf()
