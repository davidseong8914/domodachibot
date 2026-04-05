"""ImageNet mean/std normalization for torchvision pretrained backbones."""

from __future__ import annotations

import torch
import torchvision.transforms.functional as TVF

# Standard values for models trained on ImageNet (RGB, [0,1] per channel before normalize).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_rgb_01chw(x: torch.Tensor) -> torch.Tensor:
    """Normalize float RGB tensor [C, H, W] in [0, 1] for pretrained torchvision models."""
    return TVF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
