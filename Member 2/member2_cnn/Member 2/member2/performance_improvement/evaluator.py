"""
CBL-04: Improved Evaluation & Forensic Localization Engine
Phase 6 Improvement - Member 2 (Tile Forensics CNN & Heatmap Lead)

This module handles:
1. SpatialUNetInferencer: Runs U-Net inference on screenshots, yielding pixel-level heatmaps.
2. ImprovedEvaluator: Evaluates pixel and image localization metrics with post-processing.
3. Audited metrics reporting: reports both global pixel Dice and separate clean/tampered image Dices.
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
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
PHASE2_DIR = HERE.parent / "phase2_preprocessing"
PHASE3_DIR = HERE.parent / "phase3_tile_extraction"

for p in [PROJECT_ROOT, HERE, PHASE2_DIR, PHASE3_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from member2.performance_improvement.models import TileSpatialUNet
    from member2.performance_improvement.postprocessor import HeatmapPostProcessor, PostProcessingConfig
    from member2.performance_improvement.spatial_reconstructor import SpatialHeatmapReconstructor
    from member2.phase2_preprocessing.transforms import Preprocessor
    from member2.phase3_tile_extraction.tile_extractor import TileConfig, TileExtractor, pad_image_if_needed
except (ImportError, ModuleNotFoundError):
    from models import TileSpatialUNet  # type: ignore[import-not-found]
    from postprocessor import HeatmapPostProcessor, PostProcessingConfig  # type: ignore[import-not-found]
    from spatial_reconstructor import SpatialHeatmapReconstructor  # type: ignore[import-not-found]
    from tile_extractor import TileConfig, TileExtractor, pad_image_if_needed  # type: ignore[import-not-found]
    from transforms import Preprocessor  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class SpatialUNetInferencer:
    """
    Inference engine that executes TileSpatialUNet forward passes and spatial heatmap blending.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
        batch_size: int = 16,
        reconstructor: Optional[SpatialHeatmapReconstructor] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        self.device = device or torch.device("cpu")
        self.batch_size = batch_size
        self.reconstructor = reconstructor or SpatialHeatmapReconstructor()

        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model = TileSpatialUNet()
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.tile_extractor = TileExtractor(TileConfig(tile_size=256, stride=128))
        logger.info(f"Loaded SpatialUNetInferencer from {self.checkpoint_path} on {self.device}")

    def _prepare_tile_tensor(self, crops: List[np.ndarray]) -> torch.Tensor:
        arr = np.ascontiguousarray(np.stack(crops, axis=0))  # (B, H, W, C)
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        return tensor.sub_(mean).div_(std)

    @torch.no_grad()
    def predict_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Tuple[int, int]]:
        """
        Generate continuous spatial heatmap for an image.
        """
        if isinstance(image_input, (str, Path)):
            with Image.open(image_input) as img:
                img_arr = np.array(Preprocessor.convert_to_rgb(img), dtype=np.uint8)
        elif isinstance(image_input, Image.Image):
            img_arr = np.array(Preprocessor.convert_to_rgb(image_input), dtype=np.uint8)
        elif isinstance(image_input, np.ndarray):
            img_arr = image_input
        else:
            raise TypeError(f"Unsupported input type: {type(image_input)}")

        orig_h, orig_w = img_arr.shape[:2]
        ts = self.reconstructor.tile_size
        padded_img = pad_image_if_needed(img_arr, target_w=ts, target_h=ts)

        x_coords, y_coords = self.tile_extractor.compute_grid_coordinates(orig_w, orig_h)
        tile_crops = []
        tile_coords = []
        idx = 0

        for y in y_coords:
            for x in x_coords:
                crop = padded_img[y : y + ts, x : x + ts, :]
                tile_crops.append(crop)
                tile_coords.append((x, y, idx))
                idx += 1

        n_tiles = len(tile_crops)
        all_spatial_maps: List[np.ndarray] = []

        for i in range(0, n_tiles, self.batch_size):
            b_crops = tile_crops[i : i + self.batch_size]
            b_tensor = self._prepare_tile_tensor(b_crops).to(self.device)

            logits = self.model(b_tensor)
            probs = torch.sigmoid(logits)[:, 0, :, :].cpu().numpy()  # (B, 256, 256)
            for j in range(probs.shape[0]):
                all_spatial_maps.append(probs[j])

        tile_predictions = []
        for (tx, ty, tidx), smap in zip(tile_coords, all_spatial_maps):
            tile_predictions.append({
                "tile_index": tidx,
                "tile_x": tx,
                "tile_y": ty,
                "spatial_prob_map": smap,
                "max_prob": float(np.max(smap)),
            })

        heatmap = self.reconstructor.reconstruct(
            height=orig_h,
            width=orig_w,
            tile_predictions=tile_predictions,
        )

        return heatmap, tile_predictions, (orig_h, orig_w)


class ImprovedEvaluator:
    """
    Evaluator computing audited pixel and image metrics with post-processing.
    """

    def __init__(self, postprocessor: Optional[HeatmapPostProcessor] = None) -> None:
        self.postprocessor = postprocessor or HeatmapPostProcessor()

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

    def evaluate_dataset(
        self,
        samples: List[Any],
        inferencer: SpatialUNetInferencer,
        pixel_sample_limit: int = 200000,
    ) -> Dict[str, Any]:
        """
        Evaluate full dataset split and compute audited pixel and image metrics.
        """
        records: List[Dict[str, Any]] = []

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0

        sampled_probs: List[float] = []
        sampled_gts: List[int] = []

        img_true: List[int] = []
        img_scores_max: List[float] = []
        img_pred_max: List[int] = []

        family_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for s in samples:
            img_path = getattr(s, "image_path", None)
            if not img_path or not Path(img_path).exists():
                continue

            img_id = getattr(s, "image_id", "sample")
            src_id = getattr(s, "source_id", "source")
            pair_id = getattr(s, "pair_id", "pair")
            fam = getattr(s, "layout_family", "unknown")
            is_tampered = int(getattr(s, "label", 0))

            heatmap, tile_preds, (h, w) = inferencer.predict_image(img_path)
            gt_mask = self._load_ground_truth_mask(s, (h, w))

            # Apply post-processing
            pred_mask = self.postprocessor.process(heatmap)

            pred_bool = (pred_mask > 0)
            gt_bool = (gt_mask > 0)

            tp = int(np.sum(pred_bool & gt_bool))
            fp = int(np.sum(pred_bool & (~gt_bool)))
            fn = int(np.sum((~pred_bool) & gt_bool))
            tn = int(np.sum((~pred_bool) & (~gt_bool)))

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            total_pix = h * w
            acc = (tp + tn) / total_pix if total_pix > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if is_tampered == 0 and fp == 0 else 0.0)
            rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if is_tampered == 0 else 0.0)

            # Per-image Dice:
            if is_tampered == 0:
                img_dice = 1.0 if fp == 0 else 0.0
                img_iou = 1.0 if fp == 0 else 0.0
            else:
                img_dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
                img_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

            # Image-level score: max heatmap probability or area fraction
            max_p = float(np.max(heatmap))
            img_true.append(is_tampered)
            img_scores_max.append(max_p)
            img_pred_max.append(1 if max_p >= self.postprocessor.config.threshold else 0)

            rec_dict = {
                "image_id": img_id,
                "source_id": src_id,
                "pair_id": pair_id,
                "layout_family": fam,
                "is_tampered": is_tampered,
                "height": h,
                "width": w,
                "tp_pixels": tp,
                "fp_pixels": fp,
                "fn_pixels": fn,
                "tn_pixels": tn,
                "pixel_accuracy": float(acc),
                "pixel_precision": float(prec),
                "pixel_recall": float(rec),
                "dice": float(img_dice),
                "iou": float(img_iou),
                "max_prob": max_p,
            }
            records.append(rec_dict)
            family_records[fam].append(rec_dict)

            # Subsample for global curves
            flat_h = heatmap.flatten()
            flat_g = gt_bool.flatten()
            step = max(1, len(flat_h) // 2000)
            sampled_probs.extend(flat_h[::step].tolist())
            sampled_gts.extend(flat_g[::step].astype(int).tolist())

        # Global Pixel Metrics
        n_all = total_tp + total_fp + total_fn + total_tn
        glob_acc = (total_tp + total_tn) / n_all if n_all > 0 else 0.0
        glob_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        glob_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        glob_dice = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn) if (2.0 * total_tp + total_fp + total_fn) > 0 else 0.0
        glob_iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

        try:
            pix_roc_auc = float(roc_auc_score(sampled_gts, sampled_probs)) if len(set(sampled_gts)) > 1 else 0.5
        except Exception:
            pix_roc_auc = 0.5

        try:
            pix_pr_auc = float(average_precision_score(sampled_gts, sampled_probs)) if len(set(sampled_gts)) > 1 else 0.0
        except Exception:
            pix_pr_auc = 0.0

        # Mean Image Dice (All vs Tampered-Only)
        clean_records = [r for r in records if r["is_tampered"] == 0]
        tampered_records = [r for r in records if r["is_tampered"] == 1]

        mean_dice_all = float(np.mean([r["dice"] for r in records]))
        mean_iou_all = float(np.mean([r["iou"] for r in records]))
        mean_dice_tampered = float(np.mean([r["dice"] for r in tampered_records])) if tampered_records else 0.0
        mean_iou_tampered = float(np.mean([r["iou"] for r in tampered_records])) if tampered_records else 0.0

        # Image-Level Classification Metrics
        img_acc = float(accuracy_score(img_true, img_pred_max))
        img_prec = float(precision_score(img_true, img_pred_max, zero_division=0))
        img_rec = float(recall_score(img_true, img_pred_max, zero_division=0))
        img_f1 = float(f1_score(img_true, img_pred_max, zero_division=0))
        try:
            img_roc_auc = float(roc_auc_score(img_true, img_scores_max)) if len(set(img_true)) > 1 else 0.5
        except Exception:
            img_roc_auc = 0.5

        cm = confusion_matrix(img_true, img_pred_max, labels=[0, 1])
        cm_dict = {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}

        # Per-Family Breakdown
        per_family = {}
        for fam, frecs in sorted(family_records.items()):
            ftp = sum(r["tp_pixels"] for r in frecs)
            ffp = sum(r["fp_pixels"] for r in frecs)
            ffn = sum(r["fn_pixels"] for r in frecs)
            ftn = sum(r["tn_pixels"] for r in frecs)
            fpix = ftp + ffp + ffn + ftn

            fprec = ftp / (ftp + ffp) if (ftp + ffp) > 0 else 0.0
            frec = ftp / (ftp + ffn) if (ftp + ffn) > 0 else 0.0
            fdice = (2.0 * ftp) / (2.0 * ftp + ffp + ffn) if (2.0 * ftp + ffp + ffn) > 0 else 0.0
            fiou = ftp / (ftp + ffp + ffn) if (ftp + ffp + ffn) > 0 else 0.0

            fam_tamp = [r for r in frecs if r["is_tampered"] == 1]
            per_family[fam] = {
                "support": len(frecs),
                "clean_count": len(frecs) - len(fam_tamp),
                "tampered_count": len(fam_tamp),
                "pixel_accuracy": float((ftp + ftn) / fpix) if fpix > 0 else 0.0,
                "pixel_precision": float(fprec),
                "pixel_recall": float(frec),
                "global_dice": float(fdice),
                "global_iou": float(fiou),
                "mean_image_dice": float(np.mean([r["dice"] for r in frecs])),
                "mean_image_dice_tampered": float(np.mean([r["dice"] for r in fam_tamp])) if fam_tamp else 0.0,
            }

        return {
            "total_images": len(records),
            "clean_images": len(clean_records),
            "tampered_images": len(tampered_records),
            "postprocessing_config": asdict(self.postprocessor.config),
            "pixel_metrics": {
                "global_pixel_accuracy": float(glob_acc),
                "global_pixel_precision": float(glob_prec),
                "global_pixel_recall": float(glob_rec),
                "global_pixel_dice": float(glob_dice),
                "global_pixel_iou": float(glob_iou),
                "pixel_roc_auc": float(pix_roc_auc),
                "pixel_pr_auc": float(pix_pr_auc),
                "mean_image_dice_all": float(mean_dice_all),
                "mean_image_iou_all": float(mean_iou_all),
                "mean_image_dice_tampered_only": float(mean_dice_tampered),
                "mean_image_iou_tampered_only": float(mean_iou_tampered),
                "total_tp_pixels": int(total_tp),
                "total_fp_pixels": int(total_fp),
                "total_fn_pixels": int(total_fn),
                "total_tn_pixels": int(total_tn),
            },
            "image_level_metrics": {
                "accuracy": float(img_acc),
                "precision": float(img_prec),
                "recall": float(img_rec),
                "f1": float(img_f1),
                "roc_auc": float(img_roc_auc),
                "confusion_matrix": cm_dict,
            },
            "per_family_metrics": per_family,
            "per_image_records": records,
        }

