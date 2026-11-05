"""
CBL-04: Phase 7 Robustness Evaluation Under Controlled Degradations
Member 2 - Tile Forensics CNN & Heatmap Lead

This module evaluates the resilience of the improved model under 5 realistic degradations:
1. JPEG Compression: Q in [95, 85, 70, 50]
2. Resizing & Resampling: Downscale-upscale cycles [75%, 50%]
3. Gaussian Blur: sigma in [0.5, 1.0, 1.5]
4. Brightness Variations: factors in [0.90, 1.10]
5. Contrast Variations: factors in [0.90, 1.10]

Strict guarantees:
- All transforms are applied strictly in-memory to image copies.
- Zero modifications to original disk files.
- Zero model retraining.
- Both classification and localization metrics are measured per condition.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent

try:
    from member2.performance_improvement.evaluator import SpatialUNetInferencer
    from member2.performance_improvement.postprocessor import HeatmapPostProcessor, PostProcessingConfig
    from member2.phase2_preprocessing.transforms import Preprocessor
except (ImportError, ModuleNotFoundError):
    from evaluator import SpatialUNetInferencer  # type: ignore[import-not-found]
    from postprocessor import HeatmapPostProcessor, PostProcessingConfig  # type: ignore[import-not-found]
    from transforms import Preprocessor  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-Memory Image Transformations
# ---------------------------------------------------------------------------

def apply_jpeg_compression(img: Image.Image, quality: int) -> Image.Image:
    """Simulate lossy JPEG transmission compression."""
    buf = io.BytesIO()
    rgb = img.convert("RGB")
    rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def apply_resize_resample(img: Image.Image, scale: float) -> Image.Image:
    """Simulate resolution downscaling followed by upscaling back to native size."""
    w, h = img.size
    down_w = max(int(w * scale), 32)
    down_h = max(int(h * scale), 32)
    down = img.resize((down_w, down_h), resample=Image.BILINEAR)
    up = down.resize((w, h), resample=Image.BILINEAR)
    return up


def apply_gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    """Apply mild Gaussian optical/sensor blur."""
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_brightness(img: Image.Image, factor: float) -> Image.Image:
    """Adjust image brightness level."""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def apply_contrast(img: Image.Image, factor: float) -> Image.Image:
    """Adjust image contrast dynamics."""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


class RobustnessEvaluator:
    """Systematically evaluates model degradation curves under controlled perturbations."""

    def __init__(
        self,
        inferencer: SpatialUNetInferencer,
        postprocessor: Optional[HeatmapPostProcessor] = None,
        artifacts_dir: Optional[Path] = None,
    ) -> None:
        self.inferencer = inferencer
        self.postprocessor = postprocessor or HeatmapPostProcessor(
            PostProcessingConfig(threshold=0.80, min_component_area=50)
        )
        self.artifacts_dir = artifacts_dir or HERE / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _load_ground_truth_mask(self, sample: Any, target_shape: Tuple[int, int]) -> np.ndarray:
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

    def evaluate_condition(
        self,
        samples: List[Any],
        transform_fn: Callable[[Image.Image], Image.Image],
        condition_name: str,
        pixel_threshold_for_classification: int = 1000,
    ) -> Dict[str, Any]:
        """
        Evaluate full test set (or subset) under a specific perturbation function.
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0

        y_true: List[int] = []
        y_pred: List[int] = []
        tampered_dices: List[float] = []

        for s in samples:
            img_path = getattr(s, "image_path", None)
            if not img_path or not Path(img_path).exists():
                continue

            is_tampered = int(getattr(s, "is_tampered", getattr(s, "label", 0)))

            # Open image and apply in-memory transform
            with Image.open(img_path) as im:
                orig_rgb = Preprocessor.convert_to_rgb(im)
                transformed_im = transform_fn(orig_rgb)
                trans_arr = np.array(transformed_im, dtype=np.uint8)

            h, w = trans_arr.shape[:2]
            gt_mask = self._load_ground_truth_mask(s, (h, w))

            # Run inference on transformed array
            heatmap, _, _ = self.inferencer.predict_image(trans_arr)
            pred_mask = self.postprocessor.process(heatmap)

            p_bool = (pred_mask > 0)
            g_bool = (gt_mask > 0)

            tp = int(np.sum(p_bool & g_bool))
            fp = int(np.sum(p_bool & (~g_bool)))
            fn = int(np.sum((~p_bool) & g_bool))
            tn = int(np.sum((~p_bool) & (~g_bool)))

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            detected_px = int(np.sum(pred_mask > 0))
            y_true.append(is_tampered)
            y_pred.append(1 if detected_px >= pixel_threshold_for_classification else 0)

            if is_tampered == 1:
                dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
                tampered_dices.append(dice)

        tot_eval = total_tp + total_fp + total_fn + total_tn
        pix_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        pix_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        global_dice = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn) if (2.0 * total_tp + total_fp + total_fn) > 0 else 0.0
        global_iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

        # Classification metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        cls_acc = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
        cls_prec = float(precision_score(y_true, y_pred, zero_division=0)) if y_true else 0.0
        cls_rec = float(recall_score(y_true, y_pred, zero_division=0)) if y_true else 0.0
        cls_f1 = float(f1_score(y_true, y_pred, zero_division=0)) if y_true else 0.0

        return {
            "condition": condition_name,
            "sample_count": len(y_true),
            "screenshot_accuracy": cls_acc,
            "screenshot_precision": cls_prec,
            "screenshot_recall": cls_rec,
            "screenshot_f1": cls_f1,
            "global_pixel_dice": float(global_dice),
            "global_pixel_iou": float(global_iou),
            "pixel_precision": float(pix_prec),
            "pixel_recall": float(pix_rec),
            "mean_tampered_dice": float(np.mean(tampered_dices)) if tampered_dices else 0.0,
            "total_tp_pixels": total_tp,
            "total_fp_pixels": total_fp,
            "total_fn_pixels": total_fn,
        }

    def run_robustness_suite(
        self,
        samples: List[Any],
        max_samples_for_suite: int = 30,
    ) -> Dict[str, Any]:
        """
        Run the complete multi-condition robustness test suite.
        Uses balanced subset if specified for rapid CPU turnaround while preserving fidelity.
        """
        logger.info(f"Phase 7: Running robustness suite on {len(samples)} candidate samples...")

        # Select balanced test samples
        clean = [s for s in samples if getattr(s, "is_tampered", getattr(s, "label", 0)) == 0]
        edited = [s for s in samples if getattr(s, "is_tampered", getattr(s, "label", 0)) == 1]

        n_each = max_samples_for_suite // 2
        eval_samples = clean[:n_each] + edited[:n_each]
        logger.info(f"Robustness evaluation subset: {len(eval_samples)} screenshots ({len(clean[:n_each])} clean, {len(edited[:n_each])} edited)")

        conditions: List[Tuple[str, Callable[[Image.Image], Image.Image]]] = [
            ("Baseline (Clean Native)", lambda im: im),
            ("JPEG Quality 95 (High)", lambda im: apply_jpeg_compression(im, 95)),
            ("JPEG Quality 85 (Standard)", lambda im: apply_jpeg_compression(im, 85)),
            ("JPEG Quality 70 (Medium)", lambda im: apply_jpeg_compression(im, 70)),
            ("JPEG Quality 50 (Heavy)", lambda im: apply_jpeg_compression(im, 50)),
            ("Resample 75% Scale", lambda im: apply_resize_resample(im, 0.75)),
            ("Resample 50% Scale", lambda im: apply_resize_resample(im, 0.50)),
            ("Gaussian Blur (sigma=0.5)", lambda im: apply_gaussian_blur(im, 0.5)),
            ("Gaussian Blur (sigma=1.0)", lambda im: apply_gaussian_blur(im, 1.0)),
            ("Gaussian Blur (sigma=1.5)", lambda im: apply_gaussian_blur(im, 1.5)),
            ("Brightness -10% (0.90x)", lambda im: apply_brightness(im, 0.90)),
            ("Brightness +10% (1.10x)", lambda im: apply_brightness(im, 1.10)),
            ("Contrast -10% (0.90x)", lambda im: apply_contrast(im, 0.90)),
            ("Contrast +10% (1.10x)", lambda im: apply_contrast(im, 1.10)),
        ]

        results_list: List[Dict[str, Any]] = []
        for name, fn in conditions:
            logger.info(f"Evaluating condition: {name}...")
            res = self.evaluate_condition(eval_samples, fn, condition_name=name)
            results_list.append(res)

        output_bundle = {
            "evaluation_subset_size": len(eval_samples),
            "conditions_tested": len(conditions),
            "results": results_list,
        }

        with open(self.artifacts_dir / "phase7_robustness_results.json", "w", encoding="utf-8") as f:
            json.dump(output_bundle, f, indent=2)

        return output_bundle

