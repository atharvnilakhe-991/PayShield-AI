#!/usr/bin/env python3
"""
Phase 21: Merge CRAFT Detections, Score Candidate Crops, and Evaluate Aggregations
1. Merges all 4 CRAFT extraction shards into unified cache and manifest.
2. Performs strict leakage-safe OOF candidate crop scoring using frozen Phase 19 ResNet18 fold checkpoints.
3. Preprocessing exactly matches Phase 19: (256, 128) canvas with (18, 18, 18) background and ImageNet normalization.
4. Evaluates 5 predefined screenshot aggregation strategies (Max, Top-3, Top-5, Conf-Filtered Mean, Conservative Inconclusive).
5. Computes screenshot-level and crop-level metrics, category breakdowns, matched-pair deltas, and correlations.
6. Outputs full CSVs and JSON summary under experiments/phase21_craft_word_forensics/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.models as models

WORKSPACE_ROOT = Path("/Users/heemanshupjain/Downloads/Member 3")
PHASE21_DIR = WORKSPACE_ROOT / "experiments" / "phase21_craft_word_forensics"
DATA_DIR = PHASE21_DIR / "data"
RESULTS_DIR = PHASE21_DIR / "results"
DATASET_ROOT = WORKSPACE_ROOT / "Hackathon-VIT" / "CBL04-Dataset"
PHASE19_CKPT_DIR = WORKSPACE_ROOT / "experiments" / "phase19_local_branch_redevelopment" / "checkpoints"
FOLD_ASSIGNMENTS_CSV = WORKSPACE_ROOT / "experiments" / "phase19_local_branch_redevelopment" / "data" / "fold_assignments.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase21.merge_and_score")

TARGET_CANVAS = (256, 128)  # width=256, height=128
BACKGROUND_RGB = (18, 18, 18)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_roc_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    desc_idx = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_idx]
    y_scores_sorted = y_scores[desc_idx]

    distinct_idx = np.where(np.diff(y_scores_sorted))[0]
    threshold_idxs = np.r_[distinct_idx, y_true.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if fps[-1] <= 0 or tps[-1] <= 0:
        return 0.5, 0.0

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    roc_auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))

    # PR-AUC
    tps_all = np.cumsum(y_true_sorted)
    fps_all = np.cumsum(1 - y_true_sorted)
    recalls = tps_all / max(1, tps_all[-1])
    precisions = tps_all / np.maximum(1, tps_all + fps_all)
    recalls = np.r_[0.0, recalls]
    precisions = np.r_[1.0, precisions]
    pr_auc = float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))

    return round(roc_auc, 4), round(pr_auc, 4)


def compute_binary_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.50) -> Dict[str, Any]:
    y_pred = (y_scores >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    n_pos = tp + fn
    n_neg = tn + fp
    total = len(y_true)

    acc = (tp + tn) / max(1, total)
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, n_pos)
    spec = tn / max(1, n_neg)
    fpr = fp / max(1, n_neg)
    fnr = fn / max(1, n_pos)

    f1 = 2 * prec * rec / max(1e-6, (prec + rec))
    prec_neg = tn / max(1, (tn + fn))
    rec_neg = tn / max(1, n_neg)
    f1_neg = 2 * prec_neg * rec_neg / max(1e-6, (prec_neg + rec_neg))
    macro_f1 = (f1 + f1_neg) / 2.0

    return {
        "threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round(acc, 4), "precision": round(prec, 4), "recall": round(rec, 4),
        "specificity": round(spec, 4), "fpr": round(fpr, 4), "fnr": round(fnr, 4),
        "f1_score": round(f1, 4), "macro_f1": round(macro_f1, 4)
    }


def compute_box_overlap(box1: List[int], box2: List[int]) -> Tuple[float, float]:
    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    a2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = a1 + a2 - inter
    iou = inter / union if union > 0 else 0.0
    cov2 = inter / a2 if a2 > 0 else 0.0
    return round(iou, 4), round(cov2, 4)


def letterbox_crop_to_tensor(crop_pil: Image.Image) -> torch.Tensor:
    """
    Exactly reproduces Phase 19 extract_standardized_crop + T.ToTensor + T.Normalize.
    Canvas: (256, 128) [Width=256, Height=128], Background: (18, 18, 18).
    LANCZOS aspect-ratio preserving resize.
    """
    pw, ph = crop_pil.size
    scale = min(TARGET_CANVAS[0] / pw, TARGET_CANVAS[1] / ph)
    new_w = max(1, int(round(pw * scale)))
    new_h = max(1, int(round(ph * scale)))
    resized = crop_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", TARGET_CANVAS, BACKGROUND_RGB)
    canvas.paste(resized, ((TARGET_CANVAS[0] - new_w) // 2, (TARGET_CANVAS[1] - new_h) // 2))

    t = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8).reshape(128, 256, 3)
    t = t.permute(2, 0, 1).float() / 255.0  # (3, 128, 256)
    return (t - IMAGENET_MEAN) / IMAGENET_STD


def load_phase19_models(device: torch.device) -> Dict[int, Dict[str, Any]]:
    models_dict = {}
    for f in range(5):
        ckpt_path = PHASE19_CKPT_DIR / f"best_crop_resnet18_fold{f}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing Phase 19 checkpoint: {ckpt_path}")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(512, 2)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device)
        model.eval()
        sha = compute_sha256(ckpt_path)
        models_dict[f] = {
            "model": model,
            "filename": ckpt_path.name,
            "sha256": sha,
            "best_epoch": ckpt.get("epoch", -1),
            "best_val_auc": ckpt.get("best_val_auc", -1.0)
        }
        logger.info("Loaded Fold %d model: %s (SHA: %s...)", f, ckpt_path.name, sha[:16])
    return models_dict


def merge_shards(total_shards: int = 4) -> List[Dict[str, Any]]:
    merged = []
    seen_ids = set()
    for s in range(total_shards):
        p = DATA_DIR / f"craft_shard_{s}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing shard file: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            iid = r["image_id"]
            if iid in seen_ids:
                raise ValueError(f"Duplicate image_id across shards: {iid}")
            seen_ids.add(iid)
            merged.append(r)
    logger.info("Merged %d screenshot records from %d shards.", len(merged), total_shards)
    return merged


def run_candidate_scoring_and_evaluation():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Executing Phase 21 Candidate Scoring on device: %s", device)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Merge shards & write detection cache
    all_screenshots = merge_shards(4)
    cache_path = DATA_DIR / "craft_detections_cache.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(all_screenshots, f, indent=2)
    logger.info("Saved unified cache: %s (%s)", cache_path, compute_sha256(cache_path)[:16])

    # 2. Load Phase 19 Fold Classifiers
    classifiers = load_phase19_models(device)

    # 3. Prepare CSV writers
    manifest_csv = DATA_DIR / "craft_detections_manifest.csv"
    manifest_fp = open(manifest_csv, "w", newline="", encoding="utf-8")
    m_writer = csv.DictWriter(manifest_fp, fieldnames=[
        "image_id", "pair_id", "split", "fold", "label", "edit_category", "layout_family",
        "candidate_idx", "orig_x1", "orig_y1", "orig_x2", "orig_y2",
        "exp_x1", "exp_y1", "exp_x2", "exp_y2", "width", "height", "aspect_ratio",
        "detector_confidence", "iou_with_gt", "coverage_of_gt", "is_gt_crop", "crop_label"
    ])
    m_writer.writeheader()

    oof_crop_csv = RESULTS_DIR / "oof_crop_predictions.csv"
    oof_crop_fp = open(oof_crop_csv, "w", newline="", encoding="utf-8")
    oof_c_writer = csv.DictWriter(oof_crop_fp, fieldnames=[
        "image_id", "pair_id", "fold", "candidate_idx", "label", "edit_category",
        "p_crop_tampered", "detector_confidence", "width", "height", "aspect_ratio",
        "iou_with_gt", "coverage_of_gt", "is_gt_crop", "crop_label",
        "checkpoint_evaluated", "checkpoint_sha256"
    ])
    oof_c_writer.writeheader()

    val_crop_csv = RESULTS_DIR / "val_crop_predictions.csv"
    val_crop_fp = open(val_crop_csv, "w", newline="", encoding="utf-8")
    val_c_writer = csv.DictWriter(val_crop_fp, fieldnames=[
        "image_id", "pair_id", "candidate_idx", "label", "edit_category",
        "p_crop_tampered", "detector_confidence", "width", "height", "aspect_ratio",
        "iou_with_gt", "coverage_of_gt", "is_gt_crop", "crop_label"
    ])
    val_c_writer.writeheader()

    # Trackers
    total_candidates_scored = 0
    oof_screenshot_results = []
    val_screenshot_results = []

    # Crop level metric collections
    oof_crop_y_true = []
    oof_crop_scores = []
    val_crop_y_true = []
    val_crop_scores = []

    # Correlation trackers
    corr_crop_scores = []
    corr_widths = []
    corr_heights = []
    corr_areas = []
    corr_aspect_ratios = []
    corr_detector_confs = []

    train_items = [it for it in all_screenshots if it["split"] == "train"]
    val_items = [it for it in all_screenshots if it["split"] == "val"]

    logger.info("Audited dataset counts: %d train, %d validation", len(train_items), len(val_items))

    # --- PART A: SCORING TRAINING CROPS (STRICT OOF) ---
    logger.info("--- Starting Strict Out-Of-Fold Scoring on 868 Training Screenshots ---")
    t0 = time.time()

    for idx, item in enumerate(train_items):
        iid = item["image_id"]
        pid = item["pair_id"]
        fold = item["fold"]
        label = item["label"]
        cat = item.get("edit_category", "none")
        layout = item.get("layout_family", "compact")
        gt_boxes = item.get("bounding_boxes_xyxy", [])
        is_edited = (label == "edited")

        # Held-out fold model
        model = classifiers[fold]["model"]
        ckpt_name = classifiers[fold]["filename"]
        ckpt_sha = classifiers[fold]["sha256"]

        img_path = DATASET_ROOT / item["image_path"]
        with Image.open(img_path) as im:
            img_rgb = im.convert("RGB")

        candidates = item["candidates"]
        batch_tensors = []
        batch_meta = []

        best_img_iou = 0.0
        best_img_cov = 0.0

        for cand in candidates:
            c_idx = cand["candidate_idx"]
            orig_b = cand["original_bbox"]
            exp_b = cand["expanded_bbox"]
            w = cand["width"]
            h = cand["height"]
            ar = cand["aspect_ratio"]
            d_conf = cand["detector_confidence"]

            cand_iou, cand_cov = 0.0, 0.0
            if is_edited and len(gt_boxes) > 0:
                for gt_b in gt_boxes:
                    iou, cov = compute_box_overlap(orig_b, gt_b)
                    if iou > cand_iou: cand_iou = iou
                    if cov > cand_cov: cand_cov = cov
                if cand_iou > best_img_iou: best_img_iou = cand_iou
                if cand_cov > best_img_cov: best_img_cov = cand_cov

            is_gt_crop = (cand_iou >= 0.30 or cand_cov >= 0.50)
            if not is_edited:
                crop_label = 0
            else:
                if is_gt_crop:
                    crop_label = 1
                elif cand_iou < 0.05 and cand_cov < 0.05:
                    crop_label = 0
                else:
                    crop_label = -1  # Ambiguous boundary crop

            # Write manifest row
            m_writer.writerow({
                "image_id": iid, "pair_id": pid, "split": "train", "fold": fold,
                "label": label, "edit_category": cat, "layout_family": layout,
                "candidate_idx": c_idx,
                "orig_x1": orig_b[0], "orig_y1": orig_b[1], "orig_x2": orig_b[2], "orig_y2": orig_b[3],
                "exp_x1": exp_b[0], "exp_y1": exp_b[1], "exp_x2": exp_b[2], "exp_y2": exp_b[3],
                "width": w, "height": h, "aspect_ratio": ar,
                "detector_confidence": d_conf,
                "iou_with_gt": cand_iou, "coverage_of_gt": cand_cov,
                "is_gt_crop": is_gt_crop, "crop_label": crop_label
            })

            # Crop and standardize to (256, 128)
            crop_im = img_rgb.crop(exp_b)
            t = letterbox_crop_to_tensor(crop_im)
            batch_tensors.append(t)
            batch_meta.append((c_idx, d_conf, w, h, ar, cand_iou, cand_cov, is_gt_crop, crop_label))

        # Forward pass on MPS/CPU
        cand_probs = []
        if len(batch_tensors) > 0:
            batch_x = torch.stack(batch_tensors, dim=0).to(device)
            with torch.no_grad():
                logits = model(batch_x)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            for i, p_val in enumerate(probs):
                c_idx, d_conf, w, h, ar, c_iou, c_cov, is_gt, c_lbl = batch_meta[i]
                p_float = float(p_val)
                cand_probs.append(p_float)
                total_candidates_scored += 1

                oof_c_writer.writerow({
                    "image_id": iid, "pair_id": pid, "fold": fold, "candidate_idx": c_idx,
                    "label": label, "edit_category": cat,
                    "p_crop_tampered": round(p_float, 5),
                    "detector_confidence": d_conf,
                    "width": w, "height": h, "aspect_ratio": ar,
                    "iou_with_gt": c_iou, "coverage_of_gt": c_cov,
                    "is_gt_crop": is_gt, "crop_label": c_lbl,
                    "checkpoint_evaluated": ckpt_name, "checkpoint_sha256": ckpt_sha
                })

                if c_lbl != -1:
                    oof_crop_y_true.append(c_lbl)
                    oof_crop_scores.append(p_float)

                corr_crop_scores.append(p_float)
                corr_widths.append(w)
                corr_heights.append(h)
                corr_areas.append(w * h)
                corr_aspect_ratios.append(ar)
                corr_detector_confs.append(d_conf)

        # 5 Screenshot-level Aggregations
        scores_arr = np.array(cand_probs) if cand_probs else np.array([0.5])
        confs_arr = np.array([m[1] for m in batch_meta]) if batch_meta else np.array([0.0])

        s_max = float(np.max(scores_arr))
        s_top3 = float(np.mean(np.sort(scores_arr)[-3:])) if len(scores_arr) >= 3 else s_max
        s_top5 = float(np.mean(np.sort(scores_arr)[-5:])) if len(scores_arr) >= 5 else s_max

        # Strategy 4: Mean of candidates above fixed detector confidence 0.50
        conf_mask = (confs_arr >= 0.50)
        s_conf_filt_mean = float(np.mean(scores_arr[conf_mask])) if np.sum(conf_mask) > 0 else float(np.mean(scores_arr))

        # Strategy 5: Conservative aggregation
        # Decisive if score_max >= 0.65 (tampered) or <= 0.35 (clean), otherwise INCONCLUSIVE
        if s_max >= 0.65:
            cons_pred = "tampered"
            is_decisive = True
        elif s_max <= 0.35:
            cons_pred = "clean"
            is_decisive = True
        else:
            cons_pred = "inconclusive"
            is_decisive = False

        oof_screenshot_results.append({
            "image_id": iid, "pair_id": pid, "split": "train", "fold": fold,
            "label": label, "edit_category": cat, "layout_family": layout,
            "candidate_count": len(cand_probs),
            "score_max": round(s_max, 5),
            "score_top3": round(s_top3, 5),
            "score_top5": round(s_top5, 5),
            "score_conf_filt_mean": round(s_conf_filt_mean, 5),
            "conservative_prediction": cons_pred,
            "is_decisive": is_decisive,
            "best_gt_iou": round(best_img_iou, 4),
            "best_gt_coverage": round(best_img_cov, 4)
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == len(train_items):
            rate = (idx + 1) / (time.time() - t0)
            logger.info("OOF train screenshots: %d/%d (%.1f img/s)", idx + 1, len(train_items), rate)

    # --- PART B: SCORING VALIDATION CROPS (5-FOLD ENSEMBLE) ---
    logger.info("--- Starting 5-Fold Ensemble Scoring on 108 Frozen Validation Screenshots ---")
    t_val_start = time.time()

    for idx, item in enumerate(val_items):
        iid = item["image_id"]
        pid = item["pair_id"]
        label = item["label"]
        cat = item.get("edit_category", "none")
        layout = item.get("layout_family", "compact")
        gt_boxes = item.get("bounding_boxes_xyxy", [])
        is_edited = (label == "edited")

        img_path = DATASET_ROOT / item["image_path"]
        with Image.open(img_path) as im:
            img_rgb = im.convert("RGB")

        candidates = item["candidates"]
        batch_tensors = []
        batch_meta = []

        best_img_iou = 0.0
        best_img_cov = 0.0

        for cand in candidates:
            c_idx = cand["candidate_idx"]
            orig_b = cand["original_bbox"]
            exp_b = cand["expanded_bbox"]
            w = cand["width"]
            h = cand["height"]
            ar = cand["aspect_ratio"]
            d_conf = cand["detector_confidence"]

            cand_iou, cand_cov = 0.0, 0.0
            if is_edited and len(gt_boxes) > 0:
                for gt_b in gt_boxes:
                    iou, cov = compute_box_overlap(orig_b, gt_b)
                    if iou > cand_iou: cand_iou = iou
                    if cov > cand_cov: cand_cov = cov
                if cand_iou > best_img_iou: best_img_iou = cand_iou
                if cand_cov > best_img_cov: best_img_cov = cand_cov

            is_gt_crop = (cand_iou >= 0.30 or cand_cov >= 0.50)
            if not is_edited:
                crop_label = 0
            else:
                if is_gt_crop:
                    crop_label = 1
                elif cand_iou < 0.05 and cand_cov < 0.05:
                    crop_label = 0
                else:
                    crop_label = -1

            m_writer.writerow({
                "image_id": iid, "pair_id": pid, "split": "val", "fold": -1,
                "label": label, "edit_category": cat, "layout_family": layout,
                "candidate_idx": c_idx,
                "orig_x1": orig_b[0], "orig_y1": orig_b[1], "orig_x2": orig_b[2], "orig_y2": orig_b[3],
                "exp_x1": exp_b[0], "exp_y1": exp_b[1], "exp_x2": exp_b[2], "exp_y2": exp_b[3],
                "width": w, "height": h, "aspect_ratio": ar,
                "detector_confidence": d_conf,
                "iou_with_gt": cand_iou, "coverage_of_gt": cand_cov,
                "is_gt_crop": is_gt_crop, "crop_label": crop_label
            })

            crop_im = img_rgb.crop(exp_b)
            t = letterbox_crop_to_tensor(crop_im)
            batch_tensors.append(t)
            batch_meta.append((c_idx, d_conf, w, h, ar, cand_iou, cand_cov, is_gt_crop, crop_label))

        cand_probs = []
        if len(batch_tensors) > 0:
            batch_x = torch.stack(batch_tensors, dim=0).to(device)
            # Ensemble over all 5 fold models
            ensemble_probs = np.zeros(len(batch_tensors), dtype=np.float64)
            with torch.no_grad():
                for f in range(5):
                    logits = classifiers[f]["model"](batch_x)
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    ensemble_probs += probs / 5.0

            for i, p_val in enumerate(ensemble_probs):
                c_idx, d_conf, w, h, ar, c_iou, c_cov, is_gt, c_lbl = batch_meta[i]
                p_float = float(p_val)
                cand_probs.append(p_float)
                total_candidates_scored += 1

                val_c_writer.writerow({
                    "image_id": iid, "pair_id": pid, "candidate_idx": c_idx,
                    "label": label, "edit_category": cat,
                    "p_crop_tampered": round(p_float, 5),
                    "detector_confidence": d_conf,
                    "width": w, "height": h, "aspect_ratio": ar,
                    "iou_with_gt": c_iou, "coverage_of_gt": c_cov,
                    "is_gt_crop": is_gt, "crop_label": c_lbl
                })

                if c_lbl != -1:
                    val_crop_y_true.append(c_lbl)
                    val_crop_scores.append(p_float)

        scores_arr = np.array(cand_probs) if cand_probs else np.array([0.5])
        confs_arr = np.array([m[1] for m in batch_meta]) if batch_meta else np.array([0.0])

        s_max = float(np.max(scores_arr))
        s_top3 = float(np.mean(np.sort(scores_arr)[-3:])) if len(scores_arr) >= 3 else s_max
        s_top5 = float(np.mean(np.sort(scores_arr)[-5:])) if len(scores_arr) >= 5 else s_max

        conf_mask = (confs_arr >= 0.50)
        s_conf_filt_mean = float(np.mean(scores_arr[conf_mask])) if np.sum(conf_mask) > 0 else float(np.mean(scores_arr))

        if s_max >= 0.65:
            cons_pred = "tampered"
            is_decisive = True
        elif s_max <= 0.35:
            cons_pred = "clean"
            is_decisive = True
        else:
            cons_pred = "inconclusive"
            is_decisive = False

        val_screenshot_results.append({
            "image_id": iid, "pair_id": pid, "split": "val", "label": label,
            "edit_category": cat, "layout_family": layout,
            "candidate_count": len(cand_probs),
            "score_max": round(s_max, 5),
            "score_top3": round(s_top3, 5),
            "score_top5": round(s_top5, 5),
            "score_conf_filt_mean": round(s_conf_filt_mean, 5),
            "conservative_prediction": cons_pred,
            "is_decisive": is_decisive,
            "best_gt_iou": round(best_img_iou, 4),
            "best_gt_coverage": round(best_img_cov, 4)
        })

    logger.info("Validation scoring finished in %.2fs", time.time() - t_val_start)

    # Close CSVs
    manifest_fp.close()
    oof_crop_fp.close()
    val_crop_fp.close()

    # Save screenshot predictions
    oof_ss_path = RESULTS_DIR / "oof_screenshot_predictions.csv"
    with open(oof_ss_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(oof_screenshot_results[0].keys()))
        writer.writeheader()
        writer.writerows(oof_screenshot_results)

    val_ss_path = RESULTS_DIR / "val_screenshot_predictions.csv"
    with open(val_ss_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(val_screenshot_results[0].keys()))
        writer.writeheader()
        writer.writerows(val_screenshot_results)

    logger.info("Saved OOF screenshot predictions: %s", oof_ss_path)
    logger.info("Saved Val screenshot predictions: %s", val_ss_path)

    # --- PART C: COMPREHENSIVE STATISTICAL EVALUATION ---
    def evaluate_screenshot_partition(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        y_true = np.array([1 if r["label"] == "edited" else 0 for r in records])
        n_pos = int(np.sum(y_true == 1))
        n_neg = int(np.sum(y_true == 0))
        tot = len(records)

        strategies = ["score_max", "score_top3", "score_top5", "score_conf_filt_mean"]
        strategy_metrics = {}

        for strat in strategies:
            scores = np.array([r[strat] for r in records])
            clean_scores = scores[y_true == 0]
            edited_scores = scores[y_true == 1]
            roc_auc, pr_auc = compute_roc_pr_auc(y_true, scores)

            m03 = compute_binary_metrics(y_true, scores, 0.30)
            m05 = compute_binary_metrics(y_true, scores, 0.50)
            m07 = compute_binary_metrics(y_true, scores, 0.70)

            # Decisive coverage: outside [0.45, 0.55]
            dec_cnt = int(np.sum((scores < 0.45) | (scores > 0.55)))
            dec_cov = dec_cnt / tot

            strategy_metrics[strat] = {
                "roc_auc": roc_auc, "pr_auc": pr_auc,
                "decisive_count": dec_cnt, "decisive_coverage": round(dec_cov, 4),
                "clean_distribution": {
                    "mean": round(float(np.mean(clean_scores)), 4),
                    "median": round(float(np.median(clean_scores)), 4),
                    "std": round(float(np.std(clean_scores)), 4),
                    "min": round(float(np.min(clean_scores)), 4),
                    "max": round(float(np.max(clean_scores)), 4)
                },
                "edited_distribution": {
                    "mean": round(float(np.mean(edited_scores)), 4),
                    "median": round(float(np.median(edited_scores)), 4),
                    "std": round(float(np.std(edited_scores)), 4),
                    "min": round(float(np.min(edited_scores)), 4),
                    "max": round(float(np.max(edited_scores)), 4)
                },
                "mean_separation_delta": round(float(np.mean(edited_scores) - np.mean(clean_scores)), 4),
                "metrics_at_03": m03,
                "metrics_at_05": m05,
                "metrics_at_07": m07,
            }

        # Strategy 5: Conservative INCONCLUSIVE Evaluation
        cons_preds = [r["conservative_prediction"] for r in records]
        decisive_mask = np.array([r["is_decisive"] for r in records])
        n_decisive = int(np.sum(decisive_mask))
        n_inconclusive = tot - n_decisive
        inconclusive_rate = n_inconclusive / tot

        # Metrics on decisive subset
        if n_decisive > 0:
            y_true_dec = y_true[decisive_mask]
            y_pred_dec = np.array([1 if p == "tampered" else 0 for p in np.array(cons_preds)[decisive_mask]])
            c_tp = int(np.sum((y_true_dec == 1) & (y_pred_dec == 1)))
            c_fp = int(np.sum((y_true_dec == 0) & (y_pred_dec == 1)))
            c_tn = int(np.sum((y_true_dec == 0) & (y_pred_dec == 0)))
            c_fn = int(np.sum((y_true_dec == 1) & (y_pred_dec == 0)))
            c_acc = (c_tp + c_tn) / n_decisive
            c_prec = c_tp / max(1, c_tp + c_fp)
            c_rec = c_tp / max(1, c_tp + c_fn)
            c_spec = c_tn / max(1, c_tn + c_fp)
            c_f1 = 2 * c_prec * c_rec / max(1e-6, c_prec + c_rec)
        else:
            c_tp = c_fp = c_tn = c_fn = 0
            c_acc = c_prec = c_rec = c_spec = c_f1 = 0.0

        conservative_summary = {
            "total_screenshots": tot,
            "decisive_count": n_decisive,
            "decisive_coverage": round(n_decisive / tot, 4),
            "inconclusive_count": n_inconclusive,
            "inconclusive_rate": round(inconclusive_rate, 4),
            "decisive_tp": c_tp, "decisive_fp": c_fp, "decisive_tn": c_tn, "decisive_fn": c_fn,
            "decisive_accuracy": round(c_acc, 4),
            "decisive_precision": round(c_prec, 4),
            "decisive_recall": round(c_rec, 4),
            "decisive_specificity": round(c_spec, 4),
            "decisive_f1": round(c_f1, 4),
        }

        # Matched-pair differences
        by_pair = {}
        for r in records:
            pid = r["pair_id"]
            lbl = r["label"]
            by_pair.setdefault(pid, {})[lbl] = r["score_max"]

        pair_diffs = []
        edited_higher = 0
        clean_higher = 0
        tied = 0
        for pid, vals in by_pair.items():
            if "clean" in vals and "edited" in vals:
                diff = vals["edited"] - vals["clean"]
                pair_diffs.append(diff)
                if diff > 1e-4:
                    edited_higher += 1
                elif diff < -1e-4:
                    clean_higher += 1
                else:
                    tied += 1

        pair_summary = {
            "total_pairs": len(pair_diffs),
            "mean_pair_difference": round(float(np.mean(pair_diffs)), 4) if pair_diffs else 0.0,
            "median_pair_difference": round(float(np.median(pair_diffs)), 4) if pair_diffs else 0.0,
            "std_pair_difference": round(float(np.std(pair_diffs)), 4) if pair_diffs else 0.0,
            "pairs_edited_higher": edited_higher,
            "pairs_clean_higher": clean_higher,
            "pairs_tied": tied,
            "percentage_edited_higher": round(edited_higher / max(1, len(pair_diffs)) * 100, 2)
        }

        # Category breakdowns for score_max
        cat_metrics = {}
        cats = sorted(list(set(r["edit_category"] for r in records if r["edit_category"] and r["edit_category"] != "none")))
        for cat in cats:
            cat_recs = [r for r in records if r["edit_category"] == cat or r["label"] == "clean"]
            c_y = np.array([1 if r["label"] == "edited" else 0 for r in cat_recs])
            c_s = np.array([r["score_max"] for r in cat_recs])
            c_roc, c_pr = compute_roc_pr_auc(c_y, c_s)
            c_m05 = compute_binary_metrics(c_y, c_s, 0.50)
            c_ed_scores = c_s[c_y == 1]
            c_cl_scores = c_s[c_y == 0]
            cat_metrics[cat] = {
                "n_edited": int(np.sum(c_y == 1)),
                "n_clean": int(np.sum(c_y == 0)),
                "roc_auc": c_roc, "pr_auc": c_pr,
                "recall_at_05": c_m05["recall"],
                "precision_at_05": c_m05["precision"],
                "f1_at_05": c_m05["f1_score"],
                "mean_edited": round(float(np.mean(c_ed_scores)), 4),
                "mean_clean": round(float(np.mean(c_cl_scores)), 4),
                "delta": round(float(np.mean(c_ed_scores) - np.mean(c_cl_scores)), 4)
            }

        # Layout family breakdown
        layout_metrics = {}
        layouts = sorted(list(set(r["layout_family"] for r in records if r.get("layout_family"))))
        for lay in layouts:
            lay_recs = [r for r in records if r["layout_family"] == lay]
            l_y = np.array([1 if r["label"] == "edited" else 0 for r in lay_recs])
            l_s = np.array([r["score_max"] for r in lay_recs])
            l_roc, l_pr = compute_roc_pr_auc(l_y, l_s)
            l_m05 = compute_binary_metrics(l_y, l_s, 0.50)
            layout_metrics[lay] = {
                "n_total": len(lay_recs),
                "n_edited": int(np.sum(l_y == 1)),
                "n_clean": int(np.sum(l_y == 0)),
                "roc_auc": l_roc, "pr_auc": l_pr,
                "accuracy_at_05": l_m05["accuracy"],
                "macro_f1_at_05": l_m05["macro_f1"]
            }

        return {
            "total_screenshots": tot,
            "positive_screenshots": n_pos,
            "negative_screenshots": n_neg,
            "strategy_metrics": strategy_metrics,
            "conservative_summary": conservative_summary,
            "matched_pair_summary": pair_summary,
            "category_breakdown": cat_metrics,
            "layout_breakdown": layout_metrics,
        }

    oof_eval = evaluate_screenshot_partition(oof_screenshot_results)
    val_eval = evaluate_screenshot_partition(val_screenshot_results)

    # Crop-level metrics
    oof_crop_y_true = np.array(oof_crop_y_true, dtype=np.int32)
    oof_crop_scores = np.array(oof_crop_scores, dtype=np.float64)
    c_oof_roc, c_oof_pr = compute_roc_pr_auc(oof_crop_y_true, oof_crop_scores)
    c_oof_m05 = compute_binary_metrics(oof_crop_y_true, oof_crop_scores, 0.50)

    val_crop_y_true = np.array(val_crop_y_true, dtype=np.int32)
    val_crop_scores = np.array(val_crop_scores, dtype=np.float64)
    c_val_roc, c_val_pr = compute_roc_pr_auc(val_crop_y_true, val_crop_scores)
    c_val_m05 = compute_binary_metrics(val_crop_y_true, val_crop_scores, 0.50)

    crop_level_summary = {
        "oof": {
            "total_unambiguous_crops": len(oof_crop_y_true),
            "positive_crops": int(np.sum(oof_crop_y_true == 1)),
            "negative_crops": int(np.sum(oof_crop_y_true == 0)),
            "roc_auc": c_oof_roc, "pr_auc": c_oof_pr,
            "accuracy": c_oof_m05["accuracy"],
            "precision": c_oof_m05["precision"],
            "recall": c_oof_m05["recall"],
            "specificity": c_oof_m05["specificity"],
            "fpr": c_oof_m05["fpr"], "fnr": c_oof_m05["fnr"],
            "f1_score": c_oof_m05["f1_score"], "macro_f1": c_oof_m05["macro_f1"],
        },
        "val": {
            "total_unambiguous_crops": len(val_crop_y_true),
            "positive_crops": int(np.sum(val_crop_y_true == 1)),
            "negative_crops": int(np.sum(val_crop_y_true == 0)),
            "roc_auc": c_val_roc, "pr_auc": c_val_pr,
            "accuracy": c_val_m05["accuracy"],
            "precision": c_val_m05["precision"],
            "recall": c_val_m05["recall"],
            "specificity": c_val_m05["specificity"],
            "fpr": c_val_m05["fpr"], "fnr": c_val_m05["fnr"],
            "f1_score": c_val_m05["f1_score"], "macro_f1": c_val_m05["macro_f1"],
        }
    }

    # Correlations
    def calc_corr(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    correlations = {
        "score_vs_width": round(calc_corr(corr_crop_scores, corr_widths), 4),
        "score_vs_height": round(calc_corr(corr_crop_scores, corr_heights), 4),
        "score_vs_area": round(calc_corr(corr_crop_scores, corr_areas), 4),
        "score_vs_aspect_ratio": round(calc_corr(corr_crop_scores, corr_aspect_ratios), 4),
        "score_vs_detector_confidence": round(calc_corr(corr_crop_scores, corr_detector_confs), 4),
    }

    # Distribution of candidates per screenshot
    c_counts_oof = [r["candidate_count"] for r in oof_screenshot_results]
    c_counts_val = [r["candidate_count"] for r in val_screenshot_results]

    candidate_distribution_summary = {
        "oof_candidates_per_screenshot": {
            "mean": round(float(np.mean(c_counts_oof)), 2),
            "median": round(float(np.median(c_counts_oof)), 2),
            "std": round(float(np.std(c_counts_oof)), 2),
            "min": int(np.min(c_counts_oof)),
            "max": int(np.max(c_counts_oof))
        },
        "val_candidates_per_screenshot": {
            "mean": round(float(np.mean(c_counts_val)), 2),
            "median": round(float(np.median(c_counts_val)), 2),
            "std": round(float(np.std(c_counts_val)), 2),
            "min": int(np.min(c_counts_val)),
            "max": int(np.max(c_counts_val))
        },
        "correlations": correlations
    }

    # Historical comparisons dictionary
    historical_comparisons = {
        "phase15_global_cnn": {
            "approach": "Global Multi-Task ResNet18 (320x160 whole screenshot)",
            "supervision": "Screenshot-level label",
            "val_roc_auc": 0.5000, "decisive_coverage": 0.0000,
            "mean_clean": 0.7000, "mean_edited": 0.6995, "delta": -0.0005
        },
        "phase17_oracle_crop": {
            "approach": "Oracle Ground-Truth Target Field Crop (ResNet18)",
            "supervision": "Crop-level label (oracle ground truth)",
            "val_roc_auc": 0.9997, "decisive_coverage": 0.9907,
            "mean_clean": 0.0041, "mean_edited": 0.9856, "delta": 0.9815
        },
        "phase18_semantic_fusion": {
            "approach": "Global CNN + Semantic Heuristics + Multi-task Routing",
            "supervision": "Screenshot-level fusion",
            "val_roc_auc": 0.5185, "decisive_coverage": 0.0185,
            "mean_clean": 0.4986, "mean_edited": 0.5042, "delta": 0.0056
        },
        "phase19_ocr_guided_local": {
            "approach": "EasyOCR Keyword Routing + 4-Field Crop Classifier (ResNet18)",
            "supervision": "Crop-level trained, screenshot max pooled",
            "val_roc_auc": 0.7788, "decisive_coverage": 0.5463,
            "mean_clean": 0.3541, "mean_edited": 0.5898, "delta": 0.2357
        },
        "phase20_dense_mil": {
            "approach": "Uniform 4x2 / 6x3 Dense Patches + Multi-scale Attention MIL",
            "supervision": "Screenshot-level supervised Attention MIL",
            "val_roc_auc": 0.5002, "decisive_coverage": 0.0000,
            "mean_clean": 0.5003, "mean_edited": 0.5003, "delta": 0.0000
        },
        "phase21_craft_word_forensics": {
            "approach": "CRAFT Scene Text Word Detection + Frozen Phase 19 ResNet18 (Max Aggregation)",
            "supervision": "Zero OCR text recognition, pure visual scene text candidate scoring",
            "val_roc_auc": val_eval["strategy_metrics"]["score_max"]["roc_auc"],
            "val_pr_auc": val_eval["strategy_metrics"]["score_max"]["pr_auc"],
            "decisive_coverage": val_eval["strategy_metrics"]["score_max"]["decisive_coverage"],
            "mean_clean": val_eval["strategy_metrics"]["score_max"]["clean_distribution"]["mean"],
            "mean_edited": val_eval["strategy_metrics"]["score_max"]["edited_distribution"]["mean"],
            "delta": val_eval["strategy_metrics"]["score_max"]["mean_separation_delta"]
        }
    }

    # Full summary JSON
    final_summary = {
        "metadata": {
            "phase": "Phase 21 — OCR-Free Word-Level Visual Forensics",
            "date": "2026-11-04",
            "total_screenshots": len(all_screenshots),
            "total_candidates_scored": total_candidates_scored,
            "device": str(device)
        },
        "oof_screenshot_evaluation": oof_eval,
        "val_screenshot_evaluation": val_eval,
        "crop_level_evaluation": crop_level_summary,
        "candidate_distribution_summary": candidate_distribution_summary,
        "historical_comparisons": historical_comparisons
    }

    summary_json_path = RESULTS_DIR / "phase21_evaluation_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    logger.info("Saved final evaluation summary to %s (%s)", summary_json_path, compute_sha256(summary_json_path)[:16])
    logger.info("Candidate scoring and statistical evaluation completed successfully!")
    return final_summary


if __name__ == "__main__":
    run_candidate_scoring_and_evaluation()
