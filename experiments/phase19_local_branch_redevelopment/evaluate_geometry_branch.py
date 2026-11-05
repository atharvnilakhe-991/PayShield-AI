"""
Phase 19: Character-Level Geometry Feature Branch Evaluation.
Extracts character-level geometric features:
- Baseline alignment residual variance (sigma^2_baseline)
- Character/word aspect-ratio variance
- Inter-word spacing variance
- Font scale ratios (amount to payee, amount to transaction ID)
- Token height std dev

Evaluates a cross-validated Logistic Regression / Ridge classifier on the 434 training pairs (868 images)
using the exact same 5-fold split as the visual branch.
Also generates predictions for the 108 validation images.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.phase19_local_branch_redevelopment.geometry_branch import CharacterGeometryExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase19.geometry_eval")

EXP_DIR = PROJECT_ROOT / "experiments" / "phase19_local_branch_redevelopment"
DATA_DIR = EXP_DIR / "data"
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


def run_geometry_evaluation():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    extractor = CharacterGeometryExtractor()

    # 1. Load train OCR cache
    train_cache_path = DATA_DIR / "train_ocr_cache.json"
    if not train_cache_path.exists():
        logger.error("train_ocr_cache.json not found! Waiting for shards to merge.")
        return

    with open(train_cache_path) as fp:
        train_records = json.load(fp)

    # Load fold assignments
    fold_csv = DATA_DIR / "fold_assignments.csv"
    pair_to_fold = {}
    with open(fold_csv) as fp:
        for r in csv.DictReader(fp):
            pair_to_fold[r["pair_id"]] = int(r["fold"])

    # 2. Extract train features
    train_features = []
    for r in train_records:
        feats = extractor.extract_geometry_features(
            sample_id=r["sample_id"],
            image_dimensions=tuple(r["image_dimensions"]),
            raw_tokens=r.get("raw_tokens", []),
            detailed_fields=r.get("detailed_fields", {}),
        )
        fold = pair_to_fold.get(r["pair_id"], 0)
        label = 0 if r["label"] == "clean" else 1
        train_features.append({
            "sample_id": r["sample_id"],
            "pair_id": r["pair_id"],
            "fold": fold,
            "true_label": label,
            "edit_category": r.get("edit_category", "none"),
            "layout_family": r["layout_family"],
            "char_height_mean": feats.char_height_mean,
            "char_height_std": feats.char_height_std,
            "char_aspect_variance": feats.char_aspect_variance,
            "baseline_deviation_variance": feats.baseline_deviation_variance,
            "inter_word_gap_variance": feats.inter_word_gap_variance,
            "font_ratio_amount_payee": feats.font_ratio_amount_payee,
            "font_ratio_amount_txnid": feats.font_ratio_amount_txnid,
            "rule_anomaly_score": feats.geometry_anomaly_score,
        })

    feature_names = [
        "char_height_mean",
        "char_height_std",
        "char_aspect_variance",
        "baseline_deviation_variance",
        "inter_word_gap_variance",
        "font_ratio_amount_payee",
        "font_ratio_amount_txnid",
    ]

    # Convert to numpy matrices
    X = np.array([[row[col] for col in feature_names] for row in train_features], dtype=np.float64)
    y = np.array([row["true_label"] for row in train_features], dtype=np.float64)
    folds = np.array([row["fold"] for row in train_features], dtype=np.int64)

    # 3. 5-Fold Cross-Validation for Logistic Regression Stacker
    # Normalize features per fold
    oof_probs = np.zeros(len(train_features), dtype=np.float64)

    # Sigmoid helper
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))

    # Fit L2-regularized logistic regression via gradient descent per fold
    for f in range(5):
        val_mask = (folds == f)
        train_mask = ~val_mask

        X_tr = X[train_mask]
        y_tr = y[train_mask]
        X_va = X[val_mask]

        # Standardize using train fold statistics
        mean_tr = np.mean(X_tr, axis=0)
        std_tr = np.std(X_tr, axis=0)
        std_tr[std_tr < 1e-6] = 1.0

        X_tr_norm = np.hstack([np.ones((len(X_tr), 1)), (X_tr - mean_tr) / std_tr])
        X_va_norm = np.hstack([np.ones((len(X_va), 1)), (X_va - mean_tr) / std_tr])

        # Gradient descent with L2 penalty
        w = np.zeros(X_tr_norm.shape[1], dtype=np.float64)
        lr = 0.05
        l2_lambda = 0.01

        for _ in range(500):
            preds = sigmoid(X_tr_norm @ w)
            grad = (X_tr_norm.T @ (preds - y_tr)) / len(y_tr) + l2_lambda * w
            grad[0] -= l2_lambda * w[0]  # don't penalize bias
            w -= lr * grad

        val_preds = sigmoid(X_va_norm @ w)
        oof_probs[val_mask] = val_preds

    for idx, row in enumerate(train_features):
        row["p_geometry_tampered"] = round(float(oof_probs[idx]), 5)

    oof_geom_auc = compute_roc_auc(y, oof_probs)
    oof_geom_pr = compute_pr_auc(y, oof_probs)
    rule_auc = compute_roc_auc(y, np.array([r["rule_anomaly_score"] for r in train_features]))

    logger.info("Training OOF Geometry ML Model: ROC-AUC=%.4f, PR-AUC=%.4f", oof_geom_auc, oof_geom_pr)
    logger.info("Training OOF Geometry Rule Score: ROC-AUC=%.4f", rule_auc)

    # Save OOF geometry predictions
    oof_csv = RESULTS_DIR / "oof_geometry_predictions.csv"
    with open(oof_csv, "w", newline="") as fp:
        fields = list(train_features[0].keys())
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(train_features)
    logger.info("Saved OOF geometry predictions to %s", oof_csv)

    # 4. Extract Validation Features
    val_cache_path = PROJECT_ROOT / "experiments" / "phase18_semantic_roi_fusion" / "data" / "val_ocr_semantic_cache.json"
    with open(val_cache_path) as fp:
        val_records = json.load(fp)

    val_features = []
    for r in val_records:
        feats = extractor.extract_geometry_features(
            sample_id=r["sample_id"],
            image_dimensions=tuple(r["image_dimensions"]),
            raw_tokens=r.get("raw_tokens", []),
            detailed_fields=r.get("detailed_fields", {}),
        )
        label = 0 if r["label"] == "clean" else 1
        val_features.append({
            "sample_id": r["sample_id"],
            "pair_id": r["pair_id"],
            "true_label": label,
            "edit_category": r.get("edit_category", "none"),
            "layout_family": r["layout_family"],
            "char_height_mean": feats.char_height_mean,
            "char_height_std": feats.char_height_std,
            "char_aspect_variance": feats.char_aspect_variance,
            "baseline_deviation_variance": feats.baseline_deviation_variance,
            "inter_word_gap_variance": feats.inter_word_gap_variance,
            "font_ratio_amount_payee": feats.font_ratio_amount_payee,
            "font_ratio_amount_txnid": feats.font_ratio_amount_txnid,
            "rule_anomaly_score": feats.geometry_anomaly_score,
        })

    # Fit full model on all training data, evaluate on validation
    mean_full = np.mean(X, axis=0)
    std_full = np.std(X, axis=0)
    std_full[std_full < 1e-6] = 1.0

    X_full_norm = np.hstack([np.ones((len(X), 1)), (X - mean_full) / std_full])
    X_val = np.array([[row[col] for col in feature_names] for row in val_features], dtype=np.float64)
    X_val_norm = np.hstack([np.ones((len(X_val), 1)), (X_val - mean_full) / std_full])
    y_val = np.array([row["true_label"] for row in val_features], dtype=np.float64)

    w_full = np.zeros(X_full_norm.shape[1], dtype=np.float64)
    for _ in range(800):
        preds = sigmoid(X_full_norm @ w_full)
        grad = (X_full_norm.T @ (preds - y)) / len(y) + 0.01 * w_full
        grad[0] -= 0.01 * w_full[0]
        w_full -= 0.05 * grad

    val_probs = sigmoid(X_val_norm @ w_full)
    for idx, row in enumerate(val_features):
        row["p_geometry_tampered"] = round(float(val_probs[idx]), 5)

    val_geom_auc = compute_roc_auc(y_val, val_probs)
    val_geom_pr = compute_pr_auc(y_val, val_probs)
    logger.info("Frozen Validation Geometry ML Model: ROC-AUC=%.4f, PR-AUC=%.4f", val_geom_auc, val_geom_pr)

    val_csv = RESULTS_DIR / "val_geometry_predictions.csv"
    with open(val_csv, "w", newline="") as fp:
        fields = list(val_features[0].keys())
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(val_features)
    logger.info("Saved validation geometry predictions to %s", val_csv)

    # Save summary json
    summary = {
        "train_oof_roc_auc": round(oof_geom_auc, 4),
        "train_oof_pr_auc": round(oof_geom_pr, 4),
        "train_rule_score_roc_auc": round(rule_auc, 4),
        "val_roc_auc": round(val_geom_auc, 4),
        "val_pr_auc": round(val_geom_pr, 4),
        "learned_weights": {feat: round(float(w_full[i+1]), 4) for i, feat in enumerate(feature_names)},
        "bias": round(float(w_full[0]), 4),
    }
    with open(RESULTS_DIR / "geometry_branch_metrics.json", "w") as fp:
        json.dump(summary, fp, indent=2)


if __name__ == "__main__":
    run_geometry_evaluation()

