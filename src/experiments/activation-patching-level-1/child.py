# Experiment:
#

import argparse
import copy
import json
import os
import zipfile
from dataclasses import asdict, dataclass
from functools import partial
from itertools import product
from typing import Dict, Generator, List, Tuple

import torch
import torch.nn as nn
import tqdm
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torchvision.models.efficientnet import EfficientNet, FusedMBConv, MBConv
from torchvision.models.resnet import BasicBlock, ResNet
from torchvision.models.vision_transformer import (
    Encoder,
    EncoderBlock,
    MLPBlock,
    VisionTransformer,
)
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.ops.stochastic_depth import StochasticDepth

from common.util import hash_tensor
from datasets.loader import get_dataset_loader

B2B = False  # execute on B and inject from B (instead of from A)
REPLACE_RELU = False  # replace 1st relu with 2nd relu output (should change result)
ADD_NOISE_TO_RELU = False  # add noise to one layer (should change result)

ID_FIRST_LAYER = 1


def implies(a: bool, b: bool):
    return not a or b


assert implies(REPLACE_RELU, B2B)
assert implies(ADD_NOISE_TO_RELU, REPLACE_RELU)

LAYER_4_1_RELU_1 = 68
LAYER_4_1_CONV2 = 69
LAYER_4_1_RELU_2 = 71

REPLACEMENT_KEY = 0xDEADBEEF

PLATFORMS_ABBR = [
    "a40",
    "a100",
    "a100-mig40",
    "rtx6000",
    "h100",
    # "rtx3090", # == a40
]

PLATFORMS = [
    "A40",
    "A100",
    "A100-mig40",
    "RTX6000",
    "H100",
    # "RTX 3090", # == a40
]


# BasicBlock::forward
def forward(self, x: torch.Tensor) -> torch.Tensor:
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)  # LAYER_4_1_RELU_1

    out = self.conv2(out)  # LAYER_4_1_CONV2
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = self.relu(out)  # LAYER_4_1_RELU_2

    return out


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


@dataclass
class LayerOutput:
    meta: LayerOutputMeta
    output: torch.Tensor


@dataclass
class LayerRecord:
    layer_id: int
    layer_name: str
    module_type: str

    input_shapes: List[Tuple[int, ...]]
    output_shape: Tuple[int, ...]

    origin_input_ids: List[int]
    origin_output_id: int
    in_place: bool

    original_input_hash: str
    original_output_hash: str
    replaced_output_hash: str
    has_replaced_output: bool


class LayerOutputInjector:
    def __init__(
        self,
        model: torch.nn.Module,
        replaced_layer_ids: Dict[int, LayerOutput],
        *,
        detailled_layer_records: bool,
    ):
        self.model = model
        self.replaced_layer_ids = replaced_layer_ids
        self.detailled_layer_records = detailled_layer_records

        self.register_fns = []
        self.layer_id = 0

        self.layer_records = []

    def replace_hook(self, layer_name: str):
        def hook(module, input_args, output_args):
            self.layer_id += 1

            if isinstance(output_args, tuple):
                original_output = output_args[0]
            else:
                original_output = output_args

            # --- Original input/output ---
            assert isinstance(input_args, tuple), f"{layer_name}: {type(input_args)}"
            assert [torch.is_tensor(input_arg) for input_arg in input_args]

            assert torch.is_tensor(
                original_output
            ), f"{layer_name}: {type(original_output)}"

            input_shapes = [tuple(input_arg.shape) for input_arg in input_args]
            output_shape = tuple(original_output.shape)

            original_input_hash = hash_tensor(*input_args)
            original_output_hash = hash_tensor(original_output)

            has_replaced_output = True
            replacement = None

            try:
                replacement_layer_output = self.replaced_layer_ids[self.layer_id]
            except KeyError:
                has_replaced_output = False
                replacement = original_output
            else:
                assert layer_name == replacement_layer_output.meta.layer_name

                if not REPLACE_RELU:
                    assert self.layer_id == replacement_layer_output.meta.layer_id
                else:
                    if self.layer_id != LAYER_4_1_RELU_1:
                        assert self.layer_id == replacement_layer_output.meta.layer_id

                if REPLACE_RELU:
                    if self.layer_id == LAYER_4_1_RELU_1:  # layer4.1.relu
                        replacement_layer_output = self.replaced_layer_ids[
                            REPLACEMENT_KEY
                        ]

                replacement_orig = copy.deepcopy(replacement_layer_output.output)

                if REPLACE_RELU and ADD_NOISE_TO_RELU:
                    if self.layer_id == LAYER_4_1_RELU_1:
                        replacement_orig += 1

                if layer_name == "classifier.0":
                    replacement = replacement_orig.reshape(1, 1280)
                else:
                    replacement = replacement_orig

            replaced_output_hash = hash_tensor(replacement)

            assert (
                replacement.shape == output_shape
            ), f"{replacement.shape} == {output_shape}"

            # --- Record layer info (ALWAYS executed) ---
            if self.detailled_layer_records:
                self.layer_records.append(
                    LayerRecord(
                        layer_id=self.layer_id,
                        layer_name=layer_name,
                        module_type=type(module).__name__,
                        input_shapes=input_shapes,
                        output_shape=output_shape,
                        origin_input_ids=[id(input_arg) for input_arg in input_args],
                        origin_output_id=id(original_output),
                        in_place=id(input_args[0]) == id(original_output),
                        original_input_hash=original_input_hash.hex(),
                        original_output_hash=original_output_hash.hex(),
                        replaced_output_hash=replaced_output_hash.hex(),
                        has_replaced_output=has_replaced_output,
                    )
                )

            # --- Return ---
            if isinstance(module, torch.nn.modules.activation.MultiheadAttention):
                return replacement, None
            else:
                return replacement

        return hook

    def __enter__(self):
        self.layer_id = 0

        # Register hooks recursively on *all* submodules
        for layer_name, module in self.model.named_modules():
            # Skip the top-level module (usually not helpful)
            if layer_name == "":
                continue

            self.register_fns.append(
                module.register_forward_hook(self.replace_hook(layer_name))
            )

        return self

    def __exit__(self, *args):
        for register_fn in self.register_fns:
            register_fn.remove()


@torch.inference_mode()
def process_success(
    layer_id: int,
    model: torch.nn.Module,
    x_norm: torch.Tensor,
    replaced_layer_ids: Dict[int, LayerOutput],
    *,
    detailled_layer_records: bool,
):
    with LayerOutputInjector(
        model, replaced_layer_ids, detailled_layer_records=detailled_layer_records
    ) as injector:
        y: torch.Tensor = model(x_norm)

    values, indices = torch.topk(y[0], k=5, largest=True, sorted=True)
    top5_results = [
        {"class": int(idx), "value": float(val)} for idx, val in zip(indices, values)
    ]

    return y, {
        "replaced_layer": layer_id,
        "top5": top5_results,
        "records": [asdict(record) for record in injector.layer_records],
    }


def get_index_dirs(
    backdoor_basedir: str, input_basedir: str, platform_name: str, other_platform: str
) -> Generator[
    Tuple[str, str, str, str, str, List[LayerOutputMeta], str, str], None, None
]:
    if B2B:
        print("CAREFUL: You are using B2B")

    for platform_i, platform_j in product(PLATFORMS_ABBR, PLATFORMS_ABBR):
        platform_combination = f"{platform_i}-{platform_j}"
        platform_dir = os.path.join(input_basedir, platform_combination)

        if set([other_platform, platform_name]) != set([platform_i, platform_j]):
            continue

        # The execution happens on platform_name and we inject the recorded
        # activations from other_platform
        if platform_i == platform_name:
            other_platform = platform_j
        elif platform_j == platform_name:
            other_platform = platform_i
        else:
            assert False

        if B2B:
            other_platform = platform_name

        if not os.path.exists(platform_dir):
            continue

        for index_subdir in sorted(os.listdir(platform_dir)):
            index_dir = os.path.join(platform_dir, index_subdir)

            with open(os.path.join(index_dir, f"adv-model-{platform_name}.txt")) as f:
                rel_model_path = f.readline().strip()

            with open(os.path.join(index_dir, f"x_fool-{platform_name}.txt")) as f:
                rel_x_path = f.readline().strip()

            y_A_path = os.path.join(index_dir, f"y_fool-{other_platform}.pt")
            y_B_path = os.path.join(index_dir, f"y_fool-{platform_name}.pt")

            activations_subdir = f"activations-{other_platform}"
            activations_dir = os.path.join(index_dir, activations_subdir)

            with open(os.path.join(activations_dir, "activations-meta.json")) as f:
                activations_meta = json.load(f)
                activation_records = [
                    LayerOutputMeta(**obj) for obj in activations_meta
                ]

            data_dir = os.path.join(
                backdoor_basedir,
                platform_combination,
                "logs",
            )
            model_path = os.path.join(data_dir, rel_model_path)
            x_path = os.path.join(data_dir, rel_x_path)

            yield (
                model_path,
                x_path,
                platform_combination,
                index_subdir,
                activations_dir,
                activation_records,
                y_A_path,
                y_B_path,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform_name",
        choices=["a40", "a100", "a100-mig40", "rtx6000", "h100"],
        type=str,
        required=True,
    )
    parser.add_argument(
        "--other_platform_name",
        choices=["a40", "a100", "a100-mig40", "rtx6000", "h100"],
        type=str,
        required=True,
    )
    parser.add_argument("--backdoor_basedir", type=str, required=True)
    parser.add_argument("--input_basedir", type=str, required=True)
    parser.add_argument("--output_basedir", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    args = parser.parse_args()

    platform_name = args.platform_name
    other_platform_name = args.other_platform_name
    backdoor_basedir = args.backdoor_basedir
    input_basedir = args.input_basedir
    output_basedir = args.output_basedir
    model_type = args.model_type

    detailled_layer_records = model_type != "efficientnet_v2_s"

    *everything, remaining_rundir = os.path.split(output_basedir)

    if B2B:
        if REPLACE_RELU:
            if ADD_NOISE_TO_RELU:
                output_basedir = os.path.join(
                    *everything, "b2b-relu-noise", remaining_rundir
                )
            else:
                output_basedir = os.path.join(*everything, "b2b-relu", remaining_rundir)
        else:
            output_basedir = os.path.join(*everything, "b2b", remaining_rundir)
    else:
        output_basedir = os.path.join(*everything, "base", remaining_rundir)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    assert torch.cuda.is_available()

    loader = get_dataset_loader("imagenet", "data")

    index_dirs = list(
        get_index_dirs(
            backdoor_basedir, input_basedir, platform_name, other_platform_name
        )
    )

    for (
        model_path,
        x_path,
        platform_combination,
        index_subdir,
        activations_dir,
        activation_records,
        y_A_path,
        y_B_path,
    ) in tqdm.tqdm(index_dirs):
        activations_zip_path = os.path.join(activations_dir, "activations.zip")
        outdir = os.path.join(output_basedir, platform_combination, index_subdir)
        os.makedirs(outdir, exist_ok=True)

        output_progression_path = os.path.join(
            outdir, f"progression-{platform_name}.pt"
        )
        meta_progression_path = os.path.join(outdir, f"meta-{platform_name}.json")
        layer_records_path = os.path.join(outdir, f"layer-records-{platform_name}.json")

        if os.path.exists(output_progression_path):
            print("SKIP: ", output_progression_path)
            continue

        model: torch.nn.Module = torch.load(
            model_path, weights_only=True, map_location=device
        )
        model.eval()
        x_orig: torch.Tensor = torch.load(
            x_path, weights_only=True, map_location=device
        )
        x_norm = loader.normalize(x_orig)

        y_A: torch.Tensor = torch.load(
            y_A_path,
            weights_only=True,
        )
        y_B: torch.Tensor = torch.load(y_B_path, weights_only=True)

        replaced_layers: Dict[int, LayerOutput] = {}

        progression = []
        layer_records = []

        y_B_rerun, run_record = process_success(
            0,
            model,
            x_norm,
            replaced_layers,
            detailled_layer_records=detailled_layer_records,
        )
        assert hash_tensor(y_B_rerun) == hash_tensor(y_B)

        progression.append(y_B_rerun)
        layer_records.append(run_record)

        with zipfile.ZipFile(activations_zip_path) as zfile:
            meta_number_71 = activation_records[LAYER_4_1_RELU_2 - 1]
            assert (
                meta_number_71.layer_id == LAYER_4_1_RELU_2
            ), f"{meta_number_71.layer_id}"

            for layer_id, meta in tqdm.tqdm(
                enumerate(activation_records, ID_FIRST_LAYER)
            ):
                if meta.layer_name == "":
                    continue

                output = torch.load(
                    zfile.open(meta.filename),
                    map_location=device,
                    weights_only=True,
                )

                assert layer_id == meta.layer_id, f"{layer_id} vs {meta.layer_id}"
                assert all(
                    [a == b for a, b in zip(output.shape, meta.output_shape)]
                ), f"Assertion failed: {output.shape}, {meta.output_shape}"

                replaced_layers[meta.layer_id] = LayerOutput(meta, output)

                if REPLACE_RELU:
                    if meta.layer_id == LAYER_4_1_RELU_1:  # layer4.1.relu
                        replaced_layers[REPLACEMENT_KEY] = LayerOutput(
                            meta_number_71,
                            torch.load(
                                zfile.open(meta_number_71.filename),
                                map_location=device,
                                weights_only=True,
                            ),
                        )

                if not REPLACE_RELU:
                    assert (
                        len(replaced_layers) == layer_id
                    ), f"Mismatch in replaced layers: {len(replaced_layers)} == {layer_id}"

                y_B_i, run_record = process_success(
                    layer_id,
                    model,
                    x_norm,
                    replaced_layers,
                    detailled_layer_records=detailled_layer_records,
                )
                progression.append(y_B_i)
                layer_records.append(run_record)

        # We are executing on platform B. By replacing each layer with an output
        # from platform A we iteratively approach output y_A.
        progression_with_endpoints = torch.stack(progression + [y_A])
        print("Saving: ", output_progression_path)
        torch.save(progression_with_endpoints, output_progression_path)
        with open(meta_progression_path, "w") as f:
            json.dump(
                {
                    "B2B": B2B,
                    "REPLACE_RELU": REPLACE_RELU,
                    "ADD_NOISE_TO_RELU": ADD_NOISE_TO_RELU,
                    "DETAILLED_LAYER_RECORDS": detailled_layer_records,
                },
                f,
            )

        with open(layer_records_path, "w") as f:
            json.dump(layer_records, f, indent=4)

    print("DONE")


if __name__ == "__main__":
    torch.serialization.add_safe_globals(
        [
            EfficientNet,
            set,
            Conv2dNormActivation,
            FusedMBConv,
            MBConv,
            StochasticDepth,
            SqueezeExcitation,
            BasicBlock,
            nn.AvgPool2d,
            nn.AdaptiveAvgPool2d,
            nn.BatchNorm2d,
            nn.Conv2d,
            nn.Dropout,
            nn.GELU,
            nn.LayerNorm,
            nn.Linear,
            nn.MaxPool2d,
            nn.MultiheadAttention,
            nn.ReLU,
            nn.Sequential,
            nn.SiLU,
            nn.Sigmoid,
            ResNet,
            NonDynamicallyQuantizableLinear,
            VisionTransformer,
            Encoder,
            EncoderBlock,
            MLPBlock,
            partial,
        ]
    )
    main()
