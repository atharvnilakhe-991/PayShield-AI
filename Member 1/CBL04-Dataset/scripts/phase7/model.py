"""Multi-task ResNet-18 model for synthetic GPay screenshot classification/localization.

Input: float tensor [B, 3, H, W].
Output: dict with ``logits`` [B, 2] and ``mask_logits`` [B, 1, H, W].
The forward pass returns raw logits; apply softmax/sigmoid only for inference or metrics.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class ConvNormAct(nn.Sequential):
    """Small convolution block used by the mask decoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class GPayTamperModel(nn.Module):
    """ResNet-18 encoder with binary classifier and lightweight U-Net decoder.

    ``pretrained`` is opt-in. When false (the default), torchvision receives
    ``weights=None`` and model construction makes no weight download request.
    """

    def __init__(self, pretrained: bool = False, dropout: float = 0.2) -> None:
        super().__init__()
        weights: ResNet18_Weights | None = ResNet18_Weights.DEFAULT if pretrained else None
        encoder = resnet18(weights=weights)

        self.stem_conv = encoder.conv1
        self.stem_bn = encoder.bn1
        self.stem_relu = encoder.relu
        self.stem_pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.classifier_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(512, 2)

        self.decode16 = ConvNormAct(512 + 256, 256)
        self.decode8 = ConvNormAct(256 + 128, 128)
        self.decode4 = ConvNormAct(128 + 64, 64)
        self.decode2 = ConvNormAct(64 + 64, 64)
        self.mask_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    @staticmethod
    def _upsample_to(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input [B, 3, H, W], got {tuple(x.shape)}")

        input_size = x.shape[-2:]
        stem = self.stem_relu(self.stem_bn(self.stem_conv(x)))  # 1/2 resolution
        x = self.stem_pool(stem)
        skip4 = self.layer1(x)  # 1/4
        skip8 = self.layer2(skip4)  # 1/8
        skip16 = self.layer3(skip8)  # 1/16
        bottleneck = self.layer4(skip16)  # 1/32

        pooled = self.classifier_pool(bottleneck).flatten(1)
        logits = self.classifier(self.classifier_dropout(pooled))

        decoded = self.decode16(torch.cat((self._upsample_to(bottleneck, skip16), skip16), dim=1))
        decoded = self.decode8(torch.cat((self._upsample_to(decoded, skip8), skip8), dim=1))
        decoded = self.decode4(torch.cat((self._upsample_to(decoded, skip4), skip4), dim=1))
        decoded = self.decode2(torch.cat((self._upsample_to(decoded, stem), stem), dim=1))
        mask_logits = self.mask_head(decoded)
        mask_logits = F.interpolate(mask_logits, size=input_size, mode="bilinear", align_corners=False)

        return {"logits": logits, "mask_logits": mask_logits}


def build_model(config: dict[str, Any] | None = None) -> GPayTamperModel:
    """Construct from the Phase 7 JSON config; omitted options remain download-free."""
    cfg = config or {}
    return GPayTamperModel(
        pretrained=bool(cfg.get("pretrained", False)),
        dropout=float(cfg.get("dropout", 0.2)),
    )
