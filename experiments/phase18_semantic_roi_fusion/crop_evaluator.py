"""
Phase 18: Inference Crop Evaluator.
Extracts native-resolution crops for inference-selected candidate ROIs and evaluates them
using the verified Phase 17 lightweight ResNet-18 checkpoint (best_crop_resnet18.pt).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.models as models
import torchvision.transforms as T

from experiments.phase18_semantic_roi_fusion.candidate_selector import CandidateROI

logger = logging.getLogger("phase18.crop_eval")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "experiments" / "phase17_visual_forensics" / "checkpoints" / "best_crop_resnet18.pt"

TARGET_CANVAS = (256, 128)  # width=256, height=128
BACKGROUND_RGB = (18, 18, 18)


@dataclass
class CropEvaluationResult:
    candidate_id: str
    field_name: str
    crop_score: float             # Uncalibrated manipulation probability from ResNet-18
    raw_bbox_pixel: Optional[List[int]]
    padded_bbox_pixel: Optional[List[int]]
    native_crop_size: Optional[List[int]]
    target_canvas_size: List[int]
    is_missing: bool
    routing_reason: str


class InferenceCropEvaluator:
    """
    Evaluates candidate payment crops using the Phase 17 ResNet-18 visual tampering classifier.
    """

    CONTEXT_MARGINS = {
        "amount": (0.15, 0.25),
        "receiver": (0.10, 0.20),
        "transaction_id": (0.10, 0.25),
        "date": (0.10, 0.20),
        "status": (0.10, 0.20),
    }

    def __init__(self, checkpoint_path: Path = CHECKPOINT_PATH, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cpu")
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = self._load_model(checkpoint_path)

    def _load_model(self, ckpt_path: Path) -> nn.Module:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Crop model checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def extract_and_evaluate(
        self,
        image: Image.Image,
        candidate: CandidateROI,
    ) -> CropEvaluationResult:
        """
        Extracts native crop for candidate, standardizes canvas, and infers manipulation score.
        """
        if not candidate.is_available or candidate.raw_bbox_pixel is None:
            return CropEvaluationResult(
                candidate_id=candidate.candidate_id,
                field_name=candidate.field_name,
                crop_score=0.5,
                raw_bbox_pixel=None,
                padded_bbox_pixel=None,
                native_crop_size=None,
                target_canvas_size=list(TARGET_CANVAS),
                is_missing=True,
                routing_reason=candidate.routing_reason,
            )

        w_orig, h_orig = image.size
        x0, y0, x1, y1 = candidate.raw_bbox_pixel
        x0 = max(0, min(w_orig - 1, x0))
        y0 = max(0, min(h_orig - 1, y0))
        x1 = max(x0 + 1, min(w_orig, x1))
        y1 = max(y0 + 1, min(h_orig, y1))

        bw = x1 - x0
        bh = y1 - y0

        wm, hm = self.CONTEXT_MARGINS.get(candidate.field_name, (0.15, 0.20))
        pad_x = int(round(bw * wm))
        pad_y = int(round(bh * hm))

        cx0 = max(0, x0 - pad_x)
        cy0 = max(0, y0 - pad_y)
        cx1 = min(w_orig, x1 + pad_x)
        cy1 = min(h_orig, y1 + pad_y)

        patch = image.crop((cx0, cy0, cx1, cy1))
        pw, ph = patch.size

        # Fit into TARGET_CANVAS preserving aspect ratio
        scale = min(TARGET_CANVAS[0] / pw, TARGET_CANVAS[1] / ph)
        new_w = max(1, int(round(pw * scale)))
        new_h = max(1, int(round(ph * scale)))

        resized = patch.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", TARGET_CANVAS, BACKGROUND_RGB)
        offset_x = (TARGET_CANVAS[0] - new_w) // 2
        offset_y = (TARGET_CANVAS[1] - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y))

        # Forward pass
        tensor = self.transform(canvas).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
            prob = float(torch.softmax(out, dim=1)[0, 1].item())

        return CropEvaluationResult(
            candidate_id=candidate.candidate_id,
            field_name=candidate.field_name,
            crop_score=round(prob, 4),
            raw_bbox_pixel=[x0, y0, x1, y1],
            padded_bbox_pixel=[cx0, cy0, cx1, cy1],
            native_crop_size=[pw, ph],
            target_canvas_size=list(TARGET_CANVAS),
            is_missing=False,
            routing_reason=candidate.routing_reason,
        )

