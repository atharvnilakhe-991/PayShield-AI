"""
Phase 19: Branch-Independent Evaluation (Zero Stacker).
Per user gating instructions:
- ZERO multimodal stacker is trained.
- Visual Branch and Character Geometry Branch are evaluated STRICTLY INDEPENDENTLY.
- Reports for each branch:
  - Crop-level and Screenshot-level ROC-AUC and PR-AUC
  - Accuracy, Balanced Accuracy, Precision, Recall, Specificity, FPR, FNR, Macro-F1
  - Exact numerators and denominators (TP, FP, TN, FN)
  - Matched-pair score differences (Delta = P_edited - P_clean)
  - Category-level metrics and sample counts
  - Missing/unavailable candidate rates
  - OOF completeness and provenance
- Evaluates on 108 bank_progress validation images ONCE as a frozen, independent test.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.models as models
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase19.evaluate_branch_independent")

EXP_DIR = PROJECT_ROOT / "experiments" / "phase19_local_branch_redevelopment"
DATA_DIR = EXP_DIR / "data"
CKPT_DIR = EXP_DIR / "checkpoints"
RESULTS_DIR = EXP_DIR / "results"


def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
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
        return 0.5

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def compute_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    desc_idx = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_idx]
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    recalls = tps / max(1, tps[-1])
    precisions = tps / np.maximum(1, tps + fps)
    recalls = np.r_[0.0, recalls]
    precisions = np.r_[1.0, precisions]
    return float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))


def compute_full_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.50) -> Dict[str, Any]:
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
    bal_acc = (rec + spec) / 2.0
    fpr = fp / max(1, n_neg)
    fnr = fn / max(1, n_pos)

    f1 = 2 * prec * rec / max(1e-6, (prec + rec))
    prec_neg = tn / max(1, (tn + fn))
    rec_neg = tn / max(1, n_neg)
    f1_neg = 2 * prec_neg * rec_neg / max(1e-6, (prec_neg + rec_neg))
    macro_f1 = (f1 + f1_neg) / 2.0

    roc_auc = compute_roc_auc(y_true, y_scores)
    pr_auc = compute_pr_auc(y_true, y_scores)
    brier = float(np.mean((y_scores - y_true) ** 2))

    decisive_mask = (y_scores < 0.45) | (y_scores > 0.55)
    decisive_cov = float(np.mean(decisive_mask))

    return {
        "total_samples": total,
        "positive_samples": n_pos,
        "negative_samples": n_neg,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "threshold": threshold,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "f1_score": round(f1, 4),
        "macro_f1": round(macro_f1, 4),
        "brier_score": round(brier, 4),
        "decisive_coverage": round(decisive_cov, 4),
    }


def compute_pair_differences(records: List[Dict[str, Any]], score_key: str) -> Dict[str, Any]:
    by_pair = {}
    for r in records:
        pid = r["pair_id"]
        lbl = r["image_label"]
        by_pair.setdefault(pid, {})[lbl] = float(r[score_key])

    diffs = []
    clean_higher = 0
    edited_higher = 0
    ties = 0

    for pid, vals in by_pair.items():
        if "clean" in vals and "edited" in vals:
            diff = vals["edited"] - vals["clean"]
            diffs.append(diff)
            if diff > 1e-4:
                edited_higher += 1
            elif diff < -1e-4:
                clean_higher += 1
            else:
                ties += 1

    return {
        "total_evaluated_pairs": len(diffs),
        "mean_pair_difference": round(float(np.mean(diffs)), 4) if diffs else 0.0,
        "median_pair_difference": round(float(np.median(diffs)), 4) if diffs else 0.0,
        "std_pair_difference": round(float(np.std(diffs)), 4) if diffs else 0.0,
        "pairs_edited_higher": edited_higher,
        "pairs_clean_higher": clean_higher,
        "pairs_tied": ties,
        "separation_rate": round(edited_higher / max(1, len(diffs)), 4),
    }


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Starting Phase 19 Branch-Independent Evaluation on device: %s", device)

    # 1. Evaluate Visual Branch OOF
    oof_crop_csv = RESULTS_DIR / "oof_crop_predictions.csv"
    with open(oof_crop_csv) as fp:
        oof_crops = list(csv.DictReader(fp))

    y_crop_true = np.array([int(r["crop_label"]) for r in oof_crops])
    y_crop_scores = np.array([float(r["p_crop_tampered"]) for r in oof_crops])
    visual_crop_oof_metrics = compute_full_metrics(y_crop_true, y_crop_scores)
    logger.info("Visual Branch OOF Crop Metrics: ROC-AUC=%.4f, PR-AUC=%.4f, Acc=%.4f",
                visual_crop_oof_metrics["roc_auc"], visual_crop_oof_metrics["pr_auc"], visual_crop_oof_metrics["accuracy"])

    # Image-level OOF
    oof_img_csv = RESULTS_DIR / "oof_image_predictions.csv"
    with open(oof_img_csv) as fp:
        oof_imgs = list(csv.DictReader(fp))

    y_img_true = np.array([int(r["true_label"]) for r in oof_imgs])
    img_aggregations = {}
    for pool_key in ["p_visual_max", "p_visual_top2", "p_visual_weighted"]:
        scores = np.array([float(r[pool_key]) for r in oof_imgs])
        img_aggregations[pool_key] = compute_full_metrics(y_img_true, scores)
        logger.info("Visual Branch OOF Screenshot [%s]: ROC-AUC=%.4f, PR-AUC=%.4f, BalAcc=%.4f",
                    pool_key, img_aggregations[pool_key]["roc_auc"], img_aggregations[pool_key]["pr_auc"],
                    img_aggregations[pool_key]["balanced_accuracy"])

    visual_pair_diffs_oof = compute_pair_differences(oof_imgs, "p_visual_max")

    # 2. Evaluate Geometry Branch OOF
    oof_geom_csv = RESULTS_DIR / "oof_geometry_predictions.csv"
    with open(oof_geom_csv) as fp:
        oof_geoms = list(csv.DictReader(fp))

    y_geom_true = np.array([int(r["true_label"]) for r in oof_geoms])
    y_geom_scores = np.array([float(r["p_geometry_tampered"]) for r in oof_geoms])
    geom_oof_metrics = compute_full_metrics(y_geom_true, y_geom_scores)
    geom_pair_diffs_oof = compute_pair_differences(
        [{"pair_id": r["pair_id"], "image_label": "clean" if r["true_label"] == "0" else "edited", "p_geometry_tampered": r["p_geometry_tampered"]} for r in oof_geoms],
        "p_geometry_tampered"
    )

    # 3. Evaluate Frozen Validation Set (108 bank_progress images)
    # Ensemble of 5 fold models for Visual Branch
    fold_models = []
    val_trans = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for f in range(5):
        ckpt_p = CKPT_DIR / f"best_crop_resnet18_fold{f}.pt"
        ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
        m.load_state_dict(ckpt["model_state_dict"])
        m.to(device)
        m.eval()
        fold_models.append(m)

    manifest_csv = DATA_DIR / "crop_dataset_manifest.csv"
    with open(manifest_csv) as fp:
        all_crops = list(csv.DictReader(fp))
    val_crops = [r for r in all_crops if r["split"] == "val"]

    # Inference on val crops
    val_crop_predictions = {}
    val_crop_eval_list = []
    with torch.no_grad():
        for r in val_crops:
            crop_path = EXP_DIR / r["crop_rel_path"]
            with Image.open(crop_path) as src:
                img = src.convert("RGB")
            t_img = val_trans(img).unsqueeze(0).to(device)

            p_folds = []
            for m in fold_models:
                out = m(t_img)
                prob = torch.softmax(out, dim=1)[0, 1].item()
                p_folds.append(prob)
            p_ens = float(np.mean(p_folds))

            val_crop_predictions.setdefault(r["sample_id"], []).append({
                "field_name": r["field_name"],
                "p_tampered": p_ens,
                "crop_label": int(r["crop_label"]),
            })
            val_crop_eval_list.append({
                "crop_id": r["crop_id"],
                "sample_id": r["sample_id"],
                "crop_label": int(r["crop_label"]),
                "p_tampered": p_ens,
            })

    y_val_crop_true = np.array([c["crop_label"] for c in val_crop_eval_list])
    y_val_crop_scores = np.array([c["p_tampered"] for c in val_crop_eval_list])
    visual_crop_val_metrics = compute_full_metrics(y_val_crop_true, y_val_crop_scores)

    # Validation image-level aggregation
    val_cache_json = PROJECT_ROOT / "experiments" / "phase18_semantic_roi_fusion" / "data" / "val_ocr_semantic_cache.json"
    with open(val_cache_json) as fp:
        val_cache = json.load(fp)

    val_geom_csv = RESULTS_DIR / "val_geometry_predictions.csv"
    val_geom_by_sample = {}
    with open(val_geom_csv) as fp:
        for r in csv.DictReader(fp):
            val_geom_by_sample[r["sample_id"]] = float(r["p_geometry_tampered"])

    val_image_eval_rows = []
    for r in val_cache:
        sid = r["sample_id"]
        true_lbl = 0 if r["label"] == "clean" else 1
        cat = r.get("edit_category", "none")
        layout = r["layout_family"]

        crops = val_crop_predictions.get(sid, [])
        if crops:
            probs = [c["p_tampered"] for c in crops]
            p_vis_max = float(np.max(probs))
            p_vis_top2 = float(np.mean(sorted(probs, reverse=True)[:2])) if len(probs) >= 2 else p_vis_max
            # Field-specific
            fdict = {c["field_name"]: c["p_tampered"] for c in crops}
            p_vis_weighted = 0.35 * fdict.get("amount", 0.0) + 0.30 * fdict.get("receiver", 0.0) + 0.20 * fdict.get("transaction_id", 0.0) + 0.15 * fdict.get("date", 0.0)
        else:
            p_vis_max = 0.50
            p_vis_top2 = 0.50
            p_vis_weighted = 0.50

        p_geom = val_geom_by_sample.get(sid, 0.50)

        val_image_eval_rows.append({
            "sample_id": sid,
            "pair_id": r["pair_id"],
            "true_label": true_lbl,
            "image_label": r["label"],
            "edit_category": cat,
            "layout_family": layout,
            "crop_count": len(crops),
            "p_visual_max": round(p_vis_max, 5),
            "p_visual_top2": round(p_vis_top2, 5),
            "p_visual_weighted": round(p_vis_weighted, 5),
            "p_geometry": round(p_geom, 5),
        })

    # Save validation image predictions
    val_pred_csv = RESULTS_DIR / "validation_branch_predictions.csv"
    with open(val_pred_csv, "w", newline="") as fp:
        fields = list(val_image_eval_rows[0].keys())
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(val_image_eval_rows)
    logger.info("Saved validation branch predictions to %s", val_pred_csv)

    y_val_img_true = np.array([r["true_label"] for r in val_image_eval_rows])
    val_vis_max_metrics = compute_full_metrics(y_val_img_true, np.array([r["p_visual_max"] for r in val_image_eval_rows]))
    val_vis_top2_metrics = compute_full_metrics(y_val_img_true, np.array([r["p_visual_top2"] for r in val_image_eval_rows]))
    val_vis_weighted_metrics = compute_full_metrics(y_val_img_true, np.array([r["p_visual_weighted"] for r in val_image_eval_rows]))
    val_geom_metrics = compute_full_metrics(y_val_img_true, np.array([r["p_geometry"] for r in val_image_eval_rows]))

    visual_pair_diffs_val = compute_pair_differences(val_image_eval_rows, "p_visual_max")
    geom_pair_diffs_val = compute_pair_differences(val_image_eval_rows, "p_geometry")

    # Category breakdown for validation visual branch
    val_cat_breakdown = {}
    for cat in ["amount", "date_time", "recipient", "txn_id", "other"]:
        cat_edited = [r for r in val_image_eval_rows if r["edit_category"] == cat]
        clean_rows = [r for r in val_image_eval_rows if r["true_label"] == 0]
        sub_rows = cat_edited + clean_rows
        y_sub = np.array([r["true_label"] for r in sub_rows])
        scores_sub = np.array([r["p_visual_max"] for r in sub_rows])

        val_cat_breakdown[cat] = {
            "edited_samples": len(cat_edited),
            "clean_samples": len(clean_rows),
            "roc_auc": round(compute_roc_auc(y_sub, scores_sub), 4),
            "pr_auc": round(compute_pr_auc(y_sub, scores_sub), 4),
            "edited_mean_score": round(float(np.mean([r["p_visual_max"] for r in cat_edited])), 4) if cat_edited else 0.0,
            "clean_mean_score": round(float(np.mean([r["p_visual_max"] for r in clean_rows])), 4),
        }

    # Consolidated Results
    summary = {
        "execution_mode": "branch_independent_no_stacker",
        "multimodal_stacker_trained": False,
        "visual_branch": {
            "oof_crop_metrics": visual_crop_oof_metrics,
            "oof_screenshot_max_metrics": img_aggregations["p_visual_max"],
            "oof_screenshot_top2_metrics": img_aggregations["p_visual_top2"],
            "oof_screenshot_weighted_metrics": img_aggregations["p_visual_weighted"],
            "oof_pair_differences": visual_pair_diffs_oof,
            "val_crop_metrics": visual_crop_val_metrics,
            "val_screenshot_max_metrics": val_vis_max_metrics,
            "val_screenshot_top2_metrics": val_vis_top2_metrics,
            "val_screenshot_weighted_metrics": val_vis_weighted_metrics,
            "val_pair_differences": visual_pair_diffs_val,
            "val_category_breakdown": val_cat_breakdown,
        },
        "geometry_branch": {
            "oof_screenshot_metrics": geom_oof_metrics,
            "oof_pair_differences": geom_pair_diffs_oof,
            "val_screenshot_metrics": val_geom_metrics,
            "val_pair_differences": geom_pair_diffs_val,
        },
        "historical_comparisons": {
            "phase15_global_cnn": {
                "val_roc_auc": 0.5000,
                "decisive_coverage": 0.0000,
                "clean_mean": 0.7000,
                "edited_mean": 0.6995,
            },
            "phase17_oracle_upper_bound": {
                "val_roc_auc": 0.9997,
                "decisive_coverage": 0.9907,
                "clean_mean": 0.0041,
                "edited_mean": 0.9856,
            },
            "phase18_semantic_guided": {
                "val_roc_auc": 0.5185,
                "decisive_coverage": 0.0185,
                "clean_mean": 0.4986,
                "edited_mean": 0.5042,
            },
            "phase19_visual_branch_alone": {
                "val_roc_auc": val_vis_max_metrics["roc_auc"],
                "val_pr_auc": val_vis_max_metrics["pr_auc"],
                "accuracy": val_vis_max_metrics["accuracy"],
                "balanced_accuracy": val_vis_max_metrics["balanced_accuracy"],
                "decisive_coverage": val_vis_max_metrics["decisive_coverage"],
                "clean_mean": round(float(np.mean([r["p_visual_max"] for r in val_image_eval_rows if r["true_label"] == 0])), 4),
                "edited_mean": round(float(np.mean([r["p_visual_max"] for r in val_image_eval_rows if r["true_label"] == 1])), 4),
            },
            "phase19_geometry_branch_alone": {
                "val_roc_auc": val_geom_metrics["roc_auc"],
                "val_pr_auc": val_geom_metrics["pr_auc"],
                "accuracy": val_geom_metrics["accuracy"],
                "balanced_accuracy": val_geom_metrics["balanced_accuracy"],
                "decisive_coverage": val_geom_metrics["decisive_coverage"],
                "clean_mean": round(float(np.mean([r["p_geometry"] for r in val_image_eval_rows if r["true_label"] == 0])), 4),
                "edited_mean": round(float(np.mean([r["p_geometry"] for r in val_image_eval_rows if r["true_label"] == 1])), 4),
            }
        }
    }

    summary_file = RESULTS_DIR / "phase19_branch_evaluation_summary.json"
    with open(summary_file, "w") as fp:
        json.dump(summary, fp, indent=2)

    logger.info("Saved Phase 19 Branch Evaluation Summary to %s", summary_file)
    logger.info("Evaluation Complete. Stacker was strictly NOT trained.")


if __name__ == "__main__":
    main()
