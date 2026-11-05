"""
CBL-04: Tile Forensics CNN Architectures
Phase 5 - Member 2 (Tile Forensics CNN & Heatmap Lead)

This module defines neural network architectures for binary tampering detection on
256x256 RGB screenshot tiles (Class 0: Clean, Class 1: Tampered):
1. `TileResNet18`: Industry-standard ResNet-18 modified for tile classification.
2. `SimpleTileCNN`: Custom 4-stage convolutional neural network optimized for fast CPU training.
3. `get_model`: Unified model factory.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)


class SimpleTileCNN(nn.Module):
    """
    Lightweight 4-stage Convolutional Neural Network designed for 256x256 tile classification.

    Architecture:
    - Stage 1: Conv(3 -> 32, 3x3) + BN + LeakyReLU + MaxPool(2x2) -> 128x128
    - Stage 2: Conv(32 -> 64, 3x3) + BN + LeakyReLU + MaxPool(2x2) -> 64x64
    - Stage 3: Conv(64 -> 128, 3x3) + BN + LeakyReLU + MaxPool(2x2) -> 32x32
    - Stage 4: Conv(128 -> 256, 3x3) + BN + LeakyReLU + AdaptiveAvgPool2d((2, 2)) -> 1024
    - Classifier: Linear(1024, 128) -> LeakyReLU -> Dropout(0.3) -> Linear(128, num_classes)

    Total Parameters: ~800k. Excellent throughput on CPU architectures.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            # Stage 1: 256x256 -> 128x128
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stage 2: 128x128 -> 64x64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stage 3: 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stage 4: 32x32 -> 2x2
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits


class TileResNet18(nn.Module):
    """
    ResNet-18 adapted for 256x256 tile binary tampering classification.

    The standard final fully connected layer is replaced with a dropout-regularized
    projection to `num_classes` logits.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Load backbone
        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
                logger.info("Loaded pretrained ResNet-18 ImageNet weights.")
            except Exception as e:
                logger.warning(f"Could not download/load pretrained weights ({e}); initializing from scratch.")
                weights = None

        self.backbone = models.resnet18(weights=weights)

        in_features = self.backbone.fc.in_features
        if dropout > 0.0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def get_model(
    model_name: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = False,
    dropout: float = 0.2,
    **kwargs: Any,
) -> nn.Module:
    """
    Unified model factory.

    Supported architectures:
    - 'resnet18': TileResNet18
    - 'simple_cnn': SimpleTileCNN
    """
    norm_name = model_name.lower().replace("-", "_").replace(" ", "_")
    if norm_name in ("resnet18", "resnet_18", "resnet"):
        return TileResNet18(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    elif norm_name in ("simple_cnn", "simplecnn", "cnn"):
        return SimpleTileCNN(num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unsupported model architecture: '{model_name}'. Choose 'resnet18' or 'simple_cnn'.")

