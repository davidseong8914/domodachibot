"""Tiny CNN: image -> (sin(theta), cos(theta)) for steering."""

from __future__ import annotations

import torch
import torch.nn as nn


class SteerNet(nn.Module):
    """Lightweight conv stack; input NCHW float RGB [0,1], output 2D (sin, cos)."""

    def __init__(self, in_ch: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(96, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x).flatten(1)
        return self.head(z)
