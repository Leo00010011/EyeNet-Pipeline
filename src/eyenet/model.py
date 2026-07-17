"""ResNet18 + regression head -> unit gaze vector.

The 128x128 input needs no resize: ResNet18's AdaptiveAvgPool2d makes the
backbone resolution-agnostic, and the eye patch's 128x128 framing is fixed by
F-NORM -- resizing would discard it for no gain.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class GazeResNet18(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        """(B,3,128,128) -> (B,3), L2-normalized to unit length."""
        return F.normalize(self.backbone(x), p=2, dim=1, eps=1e-8)
