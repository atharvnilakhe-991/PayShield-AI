"""
CBL-04: Pixel & Image-Level Tampering Evaluation Engine
Phase 6 - Member 2 (Tile Forensics CNN & Heatmap Lead)

This module computes comprehensive forensic localization metrics on the held-out test set:
- Pixel-level metrics: Accuracy, Precision, Recall, F1, Dice coefficient, IoU, ROC-AUC, PR-AUC.
- Stratified layout family breakdown: 'compact', 'enriched', 'bank_progress', 'recharge'.
- Image-level metrics: Max tile probability, Top-k mean tile probability.
- Tabular per-image forensic logging: exported to CSV and JSON.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
PHASE2_DIR = HERE.parent / "phase2_preprocessing"

for p in [PROJECT_ROOT, HERE, PHASE2_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from member2.phase2_preprocessing.transforms import Preprocessor
except (ImportError, ModuleNotFoundError):
    from transforms import Preprocessor  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


@dataclass
class ImageEvaluationRecord:
    """Individual per-screenshot evaluation record."""

    image_id: str
    source_id: str
    pair_id: str
    layout_family: str
    is_tampered: int
    height: int
    width: int
    total_pixels: int
    tp_pixels: int
    fp_pixels: int
    fn_pixels: int
    tn_pixels: int
    pixel_accuracy: float
    pixel_precision: float
    pixel_recall: float
    dice: float
    iou: float
    max_tile_prob: float
    top_k_mean_prob: float
    tampered_pixel_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Phase6EvaluationResult:
    """Consolidated test evaluation results across pixels, layout families, and images."""

    total_images_evaluated: int
    clean_images_count: int
    tampered_images_count: int
    threshold_used: float
    pixel_metrics: Dict[str, Any]
    image_level_metrics: Dict[str, Any]
    per_family_metrics: Dict[str, Dict[str, Any]]
    per_image_records: List[ImageEvaluationRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["per_image_records"] = [r.to_dict() for r in self.per_image_records]
        return d


class Phase6Evaluator:
    """
    Forensic evaluation coordinator for reconstructed full-resolution heatmaps.
    """

    def __init__(self, threshold: float = 0.50) -> None:
        self.threshold = threshold

    def _load_ground_truth_mask(self, sample: Any, target_size: Tuple[int, int]) -> np.ndarray:
        target_h, target_w = target_size
        mask_path = getattr(sample, "mask_path", None)

        if not mask_path or not Path(mask_path).exists():
            return np.zeros((target_h, target_w), dtype=np.uint8)

        with Image.open(mask_path) as mim:
            mask_arr, _ = Preprocessor.convert_mask_to_binary(mim, foreground_val=1)

        if mask_arr.shape != (target_h, target_w):
            mask_arr = np.array(
                Image.fromarray(mask_arr).resize((target_w, target_h), resample=Image.NEAREST),
                dtype=np.uint8,
            )
            mask_arr = (mask_arr > 0).astype(np.uint8)

        return mask_arr

    def evaluate_test_set(
        self,
        test_samples: List[Any],
        inferencer: Any,
        top_k: int = 3,
        pixel_sample_limit: int = 200000,
    ) -> Phase6EvaluationResult:
        """
        Evaluate full test split across pixel and image localization metrics.
        """
        logger.info(
            f"Evaluating Phase 6 heatmaps over {len(test_samples)} test samples (threshold={self.threshold:.2f})..."
        )

        records: List[ImageEvaluationRecord] = []

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0

        # Subsample pixels for global ROC-AUC and PR-AUC
        sampled_heatmaps: List[float] = []
        sampled_gts: List[int] = []

        # Image-level predictions
        img_true: List[int] = []
        img_pred_max: List[int] = []
        img_scores_max: List[float] = []
        img_pred_topk: List[int] = []
        img_scores_topk: List[float] = []

        family_records: Dict[str, List[ImageEvaluationRecord]] = defaultdict(list)

        for s in test_samples:
            img_path = getattr(s, "image_path", None)
            if not img_path or not Path(img_path).exists():
                continue

            image_id = getattr(s, "image_id", "sample")
            source_id = getattr(s, "source_id", "source")
            pair_id = getattr(s, "pair_id", "pair")
            layout_family = getattr(s, "layout_family", "unknown")
            is_tampered = int(getattr(s, "label", 0))

            # Run inference
            heatmap, tile_preds, (h, w) = inferencer.predict_image(img_path)
            gt_mask = self._load_ground_truth_mask(s, (h, w))

            pred_binary = (heatmap >= self.threshold)
            gt_binary = (gt_mask > 0)

            tp = int(np.sum(pred_binary & gt_binary))
            fp = int(np.sum(pred_binary & (~gt_binary)))
            fn = int(np.sum((~pred_binary) & gt_binary))
            tn = int(np.sum((~pred_binary) & (~gt_binary)))

            total_pixels = h * w
            pixel_acc = (tp + tn) / total_pixels if total_pixels > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if is_tampered == 0 and fp == 0 else 0.0)
            rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if is_tampered == 0 else 0.0)
            dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else (1.0 if is_tampered == 0 and fp == 0 else 0.0)
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else (1.0 if is_tampered == 0 and fp == 0 else 0.0)

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            # Image-level scores
            img_scores = inferencer.compute_image_level_scores(
                tile_preds, heatmap, threshold=self.threshold, top_k=top_k
            )
            max_prob = img_scores["max_tile_prob"]
            topk_prob = img_scores["top_k_mean"]
            tampered_ratio = img_scores["tampered_pixel_ratio"]

            img_true.append(is_tampered)
            img_scores_max.append(max_prob)
            img_pred_max.append(1 if max_prob >= 0.50 else 0)
            img_scores_topk.append(topk_prob)
            img_pred_topk.append(1 if topk_prob >= 0.50 else 0)

            record = ImageEvaluationRecord(
                image_id=image_id,
                source_id=source_id,
                pair_id=pair_id,
                layout_family=layout_family,
                is_tampered=is_tampered,
                height=h,
                width=w,
                total_pixels=total_pixels,
                tp_pixels=tp,
                fp_pixels=fp,
                fn_pixels=fn,
                tn_pixels=tn,
                pixel_accuracy=float(pixel_acc),
                pixel_precision=float(prec),
                pixel_recall=float(rec),
                dice=float(dice),
                iou=float(iou),
                max_tile_prob=float(max_prob),
                top_k_mean_prob=float(topk_prob),
                tampered_pixel_ratio=float(tampered_ratio),
            )
            records.append(record)
            family_records[layout_family].append(record)

            # Subsample pixels for global curve calculation
            flat_h = heatmap.flatten()
            flat_g = gt_binary.flatten()
            step = max(1, len(flat_h) // 2000)
            sampled_heatmaps.extend(flat_h[::step].tolist())
            sampled_gts.extend(flat_g[::step].astype(int).tolist())

        # Aggregate Pixel-Level Metrics
        n_all_pix = total_tp + total_fp + total_fn + total_tn
        global_pix_acc = (total_tp + total_tn) / n_all_pix if n_all_pix > 0 else 0.0
        global_pix_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        global_pix_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        global_pix_dice = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn) if (2.0 * total_tp + total_fp + total_fn) > 0 else 0.0
        global_pix_iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

        try:
            pixel_roc_auc = float(roc_auc_score(sampled_gts, sampled_heatmaps)) if len(set(sampled_gts)) > 1 else 0.5
        except Exception:
            pixel_roc_auc = 0.5

        try:
            pixel_pr_auc = float(average_precision_score(sampled_gts, sampled_heatmaps)) if len(set(sampled_gts)) > 1 else 0.0
        except Exception:
            pixel_pr_auc = 0.0

        # Aggregate Image-Level Metrics
        img_acc_max = float(accuracy_score(img_true, img_pred_max))
        img_prec_max = float(precision_score(img_true, img_pred_max, zero_division=0))
        img_rec_max = float(recall_score(img_true, img_pred_max, zero_division=0))
        img_f1_max = float(f1_score(img_true, img_pred_max, zero_division=0))

        try:
            img_roc_auc_max = float(roc_auc_score(img_true, img_scores_max)) if len(set(img_true)) > 1 else 0.5
        except Exception:
            img_roc_auc_max = 0.5

        cm_max = confusion_matrix(img_true, img_pred_max, labels=[0, 1])
        cm_dict_max = {
            "tn": int(cm_max[0, 0]),
            "fp": int(cm_max[0, 1]),
            "fn": int(cm_max[1, 0]),
            "tp": int(cm_max[1, 1]),
        }

        # Per-Family Breakdown
        per_family: Dict[str, Dict[str, Any]] = {}
        for fam, frecs in sorted(family_records.items()):
            fam_tp = sum(r.tp_pixels for r in frecs)
            fam_fp = sum(r.fp_pixels for r in frecs)
            fam_fn = sum(r.fn_pixels for r in frecs)
            fam_tn = sum(r.tn_pixels for r in frecs)
            fam_pix = fam_tp + fam_fp + fam_fn + fam_tn

            fam_prec = fam_tp / (fam_tp + fam_fp) if (fam_tp + fam_fp) > 0 else 0.0
            fam_rec = fam_tp / (fam_tp + fam_fn) if (fam_tp + fam_fn) > 0 else 0.0
            fam_dice = (2.0 * fam_tp) / (2.0 * fam_tp + fam_fp + fam_fn) if (2.0 * fam_tp + fam_fp + fam_fn) > 0 else 0.0
            fam_iou = fam_tp / (fam_tp + fam_fp + fam_fn) if (fam_tp + fam_fp + fam_fn) > 0 else 0.0

            per_family[fam] = {
                "support": len(frecs),
                "clean_count": sum(1 for r in frecs if r.is_tampered == 0),
                "tampered_count": sum(1 for r in frecs if r.is_tampered == 1),
                "pixel_accuracy": float((fam_tp + fam_tn) / fam_pix) if fam_pix > 0 else 0.0,
                "pixel_precision": float(fam_prec),
                "pixel_recall": float(fam_rec),
                "dice": float(fam_dice),
                "iou": float(fam_iou),
                "mean_image_dice": float(np.mean([r.dice for r in frecs])),
                "mean_image_iou": float(np.mean([r.iou for r in frecs])),
            }

        pixel_metrics = {
            "global_pixel_accuracy": float(global_pix_acc),
            "global_pixel_precision": float(global_pix_prec),
            "global_pixel_recall": float(global_pix_rec),
            "global_dice": float(global_pix_dice),
            "global_iou": float(global_pix_iou),
            "pixel_roc_auc": float(pixel_roc_auc),
            "pixel_pr_auc": float(pixel_pr_auc),
            "mean_image_dice": float(np.mean([r.dice for r in records])),
            "mean_image_iou": float(np.mean([r.iou for r in records])),
            "mean_image_pixel_acc": float(np.mean([r.pixel_accuracy for r in records])),
            "total_pixels_evaluated": int(n_all_pix),
            "total_tp_pixels": int(total_tp),
            "total_fp_pixels": int(total_fp),
            "total_fn_pixels": int(total_fn),
            "total_tn_pixels": int(total_tn),
        }

        image_level_metrics = {
            "classifier_method": "max_tile_prob",
            "decision_threshold": 0.50,
            "accuracy": float(img_acc_max),
            "precision": float(img_prec_max),
            "recall": float(img_rec_max),
            "f1": float(img_f1_max),
            "roc_auc": float(img_roc_auc_max),
            "confusion_matrix": cm_dict_max,
            "notes": "Exploratory synthetic test classification only; does not claim real-world detection proof.",
        }

        return Phase6EvaluationResult(
            total_images_evaluated=len(records),
            clean_images_count=sum(1 for r in records if r.is_tampered == 0),
            tampered_images_count=sum(1 for r in records if r.is_tampered == 1),
            threshold_used=self.threshold,
            pixel_metrics=pixel_metrics,
            image_level_metrics=image_level_metrics,
            per_family_metrics=per_family,
            per_image_records=records,
        )

    def save_results(
        self,
        eval_result: Phase6EvaluationResult,
        output_dir: Union[str, Path],
    ) -> Tuple[Path, Path, Path]:
        """Save results to JSON and per-image CSV."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Pixel metrics JSON
        pix_json = out_path / "pixel_metrics_test.json"
        with open(pix_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "threshold_used": eval_result.threshold_used,
                    "total_images_evaluated": eval_result.total_images_evaluated,
                    "clean_images_count": eval_result.clean_images_count,
                    "tampered_images_count": eval_result.tampered_images_count,
                    "pixel_metrics": eval_result.pixel_metrics,
                    "per_family_metrics": eval_result.per_family_metrics,
                },
                f,
                indent=2,
            )

        # 2. Image level metrics JSON
        img_json = out_path / "image_metrics_test.json"
        with open(img_json, "w", encoding="utf-8") as f:
            json.dump(eval_result.image_level_metrics, f, indent=2)

        # 3. Per image CSV
        csv_path = out_path / "per_image_results.csv"
        rows = [r.to_dict() for r in eval_result.per_image_records]
        if rows:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        logger.info(f"Saved evaluation artifacts to {pix_json}, {img_json}, and {csv_path}")
        return pix_json, img_json, csv_path

