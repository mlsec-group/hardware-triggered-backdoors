import os
import sys
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

def get_model_cls_from_name(model_name):
    clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)

    for class_name, cls in clsmembers:
        if class_name == model_name:
            return cls

    return None


class ResNet18(nn.Module):
    def __init__(self, in_dim, out_dim, generator=None):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1, num_classes=out_dim
        )

    def forward(self, x):
        return self.resnet18(x)

    @staticmethod
    def type():
        return "cnn"

    @staticmethod
    def datasets():
        return ["imagenet"]


class EfficientNetB4(nn.Module):
    def __init__(self, in_dim, out_dim, generator=None):
        super(EfficientNetB4, self).__init__()
        self.efficientnet_b4 = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1, num_classes=out_dim
        )

    def forward(self, x):
        return self.efficientnet_b4(x)

    @staticmethod
    def type():
        return "cnn"

    @staticmethod
    def datasets():
        return ["imagenet"]


def maybe_download_resnet18(save_dir):
    # Path to save the model
    save_path = os.path.join(save_dir, "resnet18_model.pt")

    if os.path.exists(save_path):
        return

    torch.hub.set_dir(save_dir)

    os.makedirs(save_dir, exist_ok=True)
    # Load the pre-trained ResNet-18 model from torchvision
    print("Downloading ResNet-18 model...")
    model = models.resnet18(pretrained=True)

    # Switch the model to evaluation mode
    model.eval()

    # Save the state_dict for use without torchvision
    torch.save(model, save_path)
    print(f"Model saved successfully at {save_path}")


def maybe_download_efficientnet_v2_s(save_dir):
    # Path to save the model
    save_path = os.path.join(save_dir, "efficientnet_v2_s.pt")

    if os.path.exists(save_path):
        return

    torch.hub.set_dir(save_dir)

    os.makedirs(save_dir, exist_ok=True)
    # Load the pre-trained EfficientNetB4 model from torchvision
    print("Downloading EfficientNetv2_s model...")
    model = models.efficientnet_v2_s(
        weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )

    # Switch the model to evaluation mode
    model.eval()

    # Save the state_dict for use without torchvision
    torch.save(model, save_path)
    print(f"Model saved successfully at {save_path}")


if __name__ == "__main__":
    maybe_download_resnet18("models/imagenet")
    maybe_download_efficientnet_v2_s("models/imagenet")
