import sys
from pathlib import Path

import torch
import torch.nn as nn


class CIFARVGG(nn.Module):
    def __init__(
        self, in_channels: int, out_dim: int, hidden_dim=512, num_blocks=3, kernel_size=3
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.filter_sizes = [128] * num_blocks

        self.conv_blocks = [self._vgg_block(in_channels, self.filter_sizes[0])]
        for i in range(1, num_blocks):
            self.conv_blocks.append(
                self._vgg_block(self.filter_sizes[i - 1], self.filter_sizes[i])
            )
        self.conv_blocks = nn.Sequential(*self.conv_blocks)

        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.filter_sizes[-1], hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def _vgg_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, self.kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels, out_channels, self.kernel_size, stride=1, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(self.kernel_size),
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.flatten(x)
        return self.classifier(x)


def load_cifar_vgg(model_path: str | Path, device: torch.device | str):
    """Load the reference CIFAR VGG checkpoint.

    The checkpoint was created from a script where the model class lived in a
    module named ``models``. Register this module under that legacy name so
    unpickling works after moving the class into the native client strategy code.
    """

    legacy_models_module = sys.modules.get("models")
    if legacy_models_module is None:
        sys.modules["models"] = sys.modules[__name__]
    elif not hasattr(legacy_models_module, "CIFARVGG"):
        setattr(legacy_models_module, "CIFARVGG", CIFARVGG)

    model = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    model.eval()
    return model
