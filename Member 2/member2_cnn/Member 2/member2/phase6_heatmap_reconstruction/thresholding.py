"""
CBL-04: Validation-Only Threshold Optimization Module
Phase 6 - Member 2 (Tile Forensics CNN & Heatmap Lead)

This module searches for the optimal tampering probability threshold strictly on
the Phase 5 validation set (never on the test set or holdout):
- Evaluates candidate thresholds over reconstructed validation heatmaps.
- Computes pixel-level Dice coefficient, IoU, Precision, and Recall for each candidate.
- Selects the threshold maximizing validation Dice/IoU.
- Records detailed search curves to JSON.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
PHASE2_DIR = HERE.parent / "phase2_preprocessing"

for p in [PROJECT_ROOT, HERE, PHASE2_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from member2.phase2_preprocessing.transforms import Preprocessor
    from member2.phase6_heatmap_reconstruction.inference import TileInferencer
except (ImportError, ModuleNotFoundError):
    from inference import TileInferencer  # type: ignore[import-not-found]
    from transforms import Preprocessor  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


@dataclass
class ThresholdOptimizationResult:
    optimal_threshold: float
    best_dice: float
    best_iou: float
    best_precision: float
    best_recall: float
    evaluated_val_images: int
    evaluated_clean_count: int
    evaluated_tampered_count: int
    sweep_results: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ThresholdOptimizer:
    """
    Optimizes the decision threshold strictly on validation images.
    """

    def __init__(
        self,
        inferencer: TileInferencer,
        threshold_candidates: Optional[List[float]] = None,
        default_threshold: float = 0.50,
    ) -> None:
        self.inferencer = inferencer
        self.threshold_candidates = threshold_candidates or [
            0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
            0.55, 0.60, 0.65, 0.70, 0.75, 0.80
        ]
        self.default_threshold = default_threshold

    def _load_ground_truth_mask(self, sample: Any, target_size: Tuple[int, int]) -> np.ndarray:
        """Load and binarize ground truth mask to shape (H, W) in {0, 1}."""
        target_h, target_w = target_size
        mask_path = getattr(sample, "mask_path", None)

        if not mask_path or not Path(mask_path).exists():
            return np.zeros((target_h, target_w), dtype=np.uint8)

        with Image.open(mask_path) as mim:
            mask_arr, _ = Preprocessor.convert_mask_to_binary(mim, foreground_val=1)

        # Ensure shape matches
        if mask_arr.shape != (target_h, target_w):
            mask_arr = np.array(
                Image.fromarray(mask_arr).resize((target_w, target_h), resample=Image.NEAREST),
                dtype=np.uint8,
            )
            mask_arr = (mask_arr > 0).astype(np.uint8)

        return mask_arr

    def optimize_threshold(
        self,
        val_samples: List[Any],
        max_images: Optional[int] = 30,
        random_seed: int = 42,
    ) -> ThresholdOptimizationResult:
        """
        Run threshold sweep across validation images to select the threshold maximizing Dice/F1.
        """
        logger.info(
            f"Starting validation threshold optimization over {len(val_samples)} candidate samples "
            f"(max_images={max_images})..."
        )

        # Select balanced subset of clean and edited validation images
        clean_samples = [s for s in val_samples if getattr(s, "label", 0) == 0]
        edited_samples = [s for s in val_samples if getattr(s, "label", 0) == 1]

        rng = random.Random(random_seed)
        if max_images is not None and max_images < len(val_samples):
            n_each = max_images // 2
            rng.shuffle(clean_samples)
            rng.shuffle(edited_samples)
            selected_samples = clean_samples[:n_each] + edited_samples[:n_each]
        else:
            selected_samples = list(val_samples)

        logger.info(
            f"Selected {len(selected_samples)} validation images for threshold tuning "
            f"({sum(1 for s in selected_samples if getattr(s, 'label', 0) == 0)} clean, "
            f"{sum(1 for s in selected_samples if getattr(s, 'label', 0) == 1)} edited)"
        )

        # Precompute heatmaps and GT masks for selected validation images
        val_data: List[Tuple[np.ndarray, np.ndarray]] = []
        for s in selected_samples:
            img_path = getattr(s, "image_path", None)
            if not img_path or not Path(img_path).exists():
                continue
            heatmap, _, (h, w) = self.inferencer.predict_image(img_path)
            gt_mask = self._load_ground_truth_mask(s, (h, w))
            val_data.append((heatmap, gt_mask))

        # Accumulate pixel counts for each candidate threshold
        sweep_results: Dict[str, Dict[str, float]] = {}
        best_dice = -1.0
        best_thresh = self.default_threshold
        best_metrics: Dict[str, float] = {}

        for th in self.threshold_candidates:
            total_tp = 0
            total_fp = 0
            total_fn = 0
            total_tn = 0

            for heatmap, gt_mask in val_data:
                pred_binary = (heatmap >= th)
                gt_binary = (gt_mask > 0)

                tp = int(np.sum(pred_binary & gt_binary))
                fp = int(np.sum(pred_binary & (~gt_binary)))
                fn = int(np.sum((~pred_binary) & gt_binary))
                tn = int(np.sum((~pred_binary) & (~gt_binary)))

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            dice = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn) if (2.0 * total_tp + total_fp + total_fn) > 0 else 0.0
            iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
            pixel_acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0.0

            res = {
                "threshold": float(th),
                "dice": float(dice),
                "iou": float(iou),
                "precision": float(precision),
                "recall": float(recall),
                "pixel_accuracy": float(pixel_acc),
                "tp_pixels": total_tp,
                "fp_pixels": total_fp,
                "fn_pixels": total_fn,
                "tn_pixels": total_tn,
            }
            sweep_results[f"{th:.2f}"] = res

            logger.info(
                f"Validation sweep: th={th:.2f} -> Dice={dice:.4f}, IoU={iou:.4f}, "
                f"Prec={precision:.4f}, Rec={recall:.4f}"
            )

            # Maximize validation Dice
            if dice > best_dice:
                best_dice = dice
                best_thresh = th
                best_metrics = res

        # Handle edge case where all candidate Dices are zero
        if best_dice <= 0.0:
            logger.warning(
                f"All validation thresholds yielded zero Dice score; falling back to default threshold {self.default_threshold}"
            )
            best_thresh = self.default_threshold
            best_metrics = sweep_results.get(f"{self.default_threshold:.2f}", {})

        logger.info(
            f"[*] Validation-optimal threshold selected: {best_thresh:.2f} "
            f"(Validation Dice={best_metrics.get('dice', 0.0):.4f}, IoU={best_metrics.get('iou', 0.0):.4f})"
        )

        return ThresholdOptimizationResult(
            optimal_threshold=best_thresh,
            best_dice=best_metrics.get("dice", 0.0),
            best_iou=best_metrics.get("iou", 0.0),
            best_precision=best_metrics.get("precision", 0.0),
            best_recall=best_metrics.get("recall", 0.0),
            evaluated_val_images=len(val_data),
            evaluated_clean_count=sum(1 for s in selected_samples if getattr(s, "label", 0) == 0),
            evaluated_tampered_count=sum(1 for s in selected_samples if getattr(s, "label", 0) == 1),
            sweep_results=sweep_results,
        )

