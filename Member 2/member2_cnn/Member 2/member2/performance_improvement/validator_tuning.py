"""
CBL-04: Joint Threshold & Spatial Post-Processing Validation Optimizer
Phase 6 Improvement - Member 2 (Tile Forensics CNN & Heatmap Lead)

This module executes joint 2D parameter search strictly over the Phase 5 validation partition:
- Parameter 1: Decision threshold tau in [0.15, 0.80].
- Parameter 2: Connected-component minimum area in [0, 5, 10, 16, 25, 50] pixels.
- Objective: Maximize validation pixel Dice and IoU.
- Guarantees zero test-set peeking.
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
    from member2.performance_improvement.evaluator import SpatialUNetInferencer
    from member2.performance_improvement.postprocessor import HeatmapPostProcessor, PostProcessingConfig
    from member2.phase2_preprocessing.transforms import Preprocessor
except (ImportError, ModuleNotFoundError):
    from evaluator import SpatialUNetInferencer  # type: ignore[import-not-found]
    from postprocessor import HeatmapPostProcessor, PostProcessingConfig  # type: ignore[import-not-found]
    from transforms import Preprocessor  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


class ValidationTuner:
    """
    Joint parameter optimizer searching over validation images.
    """

    def __init__(
        self,
        inferencer: SpatialUNetInferencer,
        threshold_candidates: Optional[List[float]] = None,
        area_candidates: Optional[List[int]] = None,
    ) -> None:
        self.inferencer = inferencer
        self.threshold_candidates = threshold_candidates or [
            0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80
        ]
        self.area_candidates = area_candidates or [0, 10, 16, 25, 50]

    def _load_gt_mask(self, sample: Any, target_shape: Tuple[int, int]) -> np.ndarray:
        h, w = target_shape
        mask_path = getattr(sample, "mask_path", None)
        if not mask_path or not Path(mask_path).exists():
            return np.zeros((h, w), dtype=np.uint8)

        with Image.open(mask_path) as mim:
            mask_arr, _ = Preprocessor.convert_mask_to_binary(mim, foreground_val=1)

        if mask_arr.shape != (h, w):
            mask_arr = np.array(
                Image.fromarray(mask_arr).resize((w, h), resample=Image.NEAREST),
                dtype=np.uint8,
            )
            mask_arr = (mask_arr > 0).astype(np.uint8)
        return mask_arr

    def tune(
        self,
        val_samples: List[Any],
        max_images: int = 30,
        seed: int = 42,
    ) -> Dict[str, Any]:
        logger.info(
            f"Starting joint validation tuning over {len(val_samples)} candidate validation samples (max={max_images})..."
        )

        clean = [s for s in val_samples if getattr(s, "label", 0) == 0]
        edited = [s for s in val_samples if getattr(s, "label", 0) == 1]

        rng = random.Random(seed)
        n_each = max_images // 2
        rng.shuffle(clean)
        rng.shuffle(edited)
        selected = clean[:n_each] + edited[:n_each]

        # Precompute heatmaps and GT masks for validation images
        val_cache = []
        for s in selected:
            img_path = getattr(s, "image_path")
            hm, _, (h, w) = self.inferencer.predict_image(img_path)
            gt = self._load_gt_mask(s, (h, w))
            val_cache.append((hm, gt))

        best_dice = -1.0
        best_cfg = PostProcessingConfig()
        best_metrics: Dict[str, float] = {}
        all_trials: List[Dict[str, Any]] = []

        for th in self.threshold_candidates:
            for area in self.area_candidates:
                pp = HeatmapPostProcessor(PostProcessingConfig(threshold=th, min_component_area=area))

                total_tp = 0
                total_fp = 0
                total_fn = 0
                total_tn = 0

                for hm, gt in val_cache:
                    pred = pp.process(hm)
                    p_bool = (pred > 0)
                    g_bool = (gt > 0)

                    total_tp += int(np.sum(p_bool & g_bool))
                    total_fp += int(np.sum(p_bool & (~g_bool)))
                    total_fn += int(np.sum((~p_bool) & g_bool))
                    total_tn += int(np.sum((~p_bool) & (~g_bool)))

                prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                dice = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn) if (2.0 * total_tp + total_fp + total_fn) > 0 else 0.0
                iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

                trial = {
                    "threshold": float(th),
                    "min_component_area": int(area),
                    "dice": float(dice),
                    "iou": float(iou),
                    "precision": float(prec),
                    "recall": float(rec),
                    "tp_pixels": total_tp,
                    "fp_pixels": total_fp,
                    "fn_pixels": total_fn,
                }
                all_trials.append(trial)

                if dice > best_dice:
                    best_dice = dice
                    best_cfg = PostProcessingConfig(threshold=th, min_component_area=area)
                    best_metrics = trial

        logger.info(
            f"[*] Validation tuning optimal parameters: threshold={best_cfg.threshold:.2f}, "
            f"min_component_area={best_cfg.min_component_area} "
            f"(Val Dice={best_dice:.4f}, Val IoU={best_metrics.get('iou', 0.0):.4f}, "
            f"Val Prec={best_metrics.get('precision', 0.0):.4f}, Val Rec={best_metrics.get('recall', 0.0):.4f})"
        )

        return {
            "optimal_config": asdict(best_cfg),
            "best_validation_metrics": best_metrics,
            "all_trials": all_trials,
            "evaluated_images_count": len(val_cache),
        }

