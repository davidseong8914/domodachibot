"""Steering model: MobileNet V3 Small (ImageNet) -> (sin(theta), cos(theta))."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class SteerNet(nn.Module):
    """ImageNet-pretrained MobileNet V3 Small; input [B, 3, H, W] float RGB [0,1], output [B, 2] (sin, cos).

    Torchvision's forward applies global average pooling on the feature map, then this module's
    head is a single linear layer (replacing the default 1000-class classifier).
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = mobilenet_v3_small(weights="DEFAULT")
        in_features = self.net.classifier[0].in_features
        self.net.classifier = nn.Linear(in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
