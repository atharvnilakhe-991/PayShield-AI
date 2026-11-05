"""
Phase 8 OOF Stacker Training Script.
Trains an L2-regularized logistic regression stacker on verified OOF training rows (868 samples).
Evaluates regularization candidates strictly on the validation partition (108 samples).
Preserves held-out test data untouched.
Saves model weights, scaler, and metadata to JSON format (no unsafe pickle).
Generates detailed validation metrics and feature attribution reports.
"""

import csv
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DATASET_ROOT = WORKSPACE_ROOT / "Hackathon-VIT" / "CBL04-Dataset"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_stacker")

FEATURE_COLUMNS = [
    "cnn_score",
    "cnn_pred_class",
    "ocr_overall_confidence",
    "ocr_amount_extracted",
    "ocr_recipient_extracted",
    "ocr_ref_id_extracted",
    "ocr_date_extracted",
    "semantic_manipulation_score",
    "amount_anomaly",
    "recipient_anomaly",
    "date_anomaly",
    "ref_id_anomaly",
    "status_anomaly",
    "upi_handle_anomaly",
    "missing_critical_fields_count",
    "branch_cnn_available",
    "branch_ocr_available",
    "branch_semantic_available",
]


def load_dataset_features(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Loads feature matrix X, ground truth vector y, and sample IDs from feature CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing feature CSV: {csv_path}")

    sample_ids = []
    X_rows = []
    y_list = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sample_ids.append(r["sample_id"])
            y_list.append(int(r["true_label"]))
            row_feats = []
            for col in FEATURE_COLUMNS:
                val = float(r.get(col, 0.0))
                if col == "missing_critical_fields_count":
                    val = val / 4.0  # Normalize to [0, 1]
                row_feats.append(val)
            X_rows.append(row_feats)

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    return X, y, sample_ids


def compute_standardizer(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes mean and std for feature scaling. Binary features remain unscaled."""
    means = np.zeros(X.shape[1], dtype=np.float64)
    stds = np.ones(X.shape[1], dtype=np.float64)

    # Standardize continuous features only
    continuous_indices = [0, 2, 7, 14]  # cnn_score, ocr_overall_confidence, semantic_manipulation_score, missing_critical_fields_count
    for idx in continuous_indices:
        col = X[:, idx]
        mean = float(np.mean(col))
        std = float(np.std(col))
        means[idx] = mean
        stds[idx] = std if std > 1e-6 else 1.0

    return means, stds


def apply_standardizer(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return (X - means) / stds


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def loss_and_grad(params: np.ndarray, X: np.ndarray, y: np.ndarray, l2_reg: float) -> Tuple[float, np.ndarray]:
    w = params[:-1]
    b = params[-1]
    n = len(y)

    z = np.dot(X, w) + b
    p = sigmoid(z)

    # Cross entropy loss with numerical clipping
    eps = 1e-12
    p_clipped = np.clip(p, eps, 1.0 - eps)
    ce_loss = -np.mean(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped))
    reg_loss = 0.5 * l2_reg * np.sum(w ** 2)
    total_loss = float(ce_loss + reg_loss)

    # Gradients
    dz = (p - y) / n
    dw = np.dot(X.T, dz) + l2_reg * w
    db = np.sum(dz)

    grad = np.concatenate([dw, [db]])
    return total_loss, grad


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    n = len(y_true)
    acc = (tp + tn) / n if n > 0 else 0.0

    # Class 1 (Edited)
    prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1 = 2 * prec_1 * rec_1 / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0.0

    # Class 0 (Clean)
    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0

    macro_f1 = (f1_1 + f1_0) / 2.0

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "edited_class": {
            "precision": round(prec_1, 4),
            "recall": round(rec_1, 4),
            "f1": round(f1_1, 4),
            "support": int(np.sum(y_true == 1)),
        },
        "clean_class": {
            "precision": round(prec_0, 4),
            "recall": round(rec_0, 4),
            "f1": round(f1_0, 4),
            "support": int(np.sum(y_true == 0)),
        },
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
    }


def train_and_evaluate():
    train_csv = DATASET_ROOT / "metadata" / "phase8" / "oof_stacker_train_features.csv"
    val_csv = DATASET_ROOT / "metadata" / "phase8" / "validation_features.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Feature CSVs not ready: train={train_csv.exists()}, val={val_csv.exists()}")

    X_train_raw, y_train, train_ids = load_dataset_features(train_csv)
    X_val_raw, y_val, val_ids = load_dataset_features(val_csv)

    logger.info("Loaded Training: %d samples (clean=%d, edited=%d)", len(y_train), np.sum(y_train == 0), np.sum(y_train == 1))
    logger.info("Loaded Validation: %d samples (clean=%d, edited=%d)", len(y_val), np.sum(y_val == 0), np.sum(y_val == 1))

    # Standardize
    means, stds = compute_standardizer(X_train_raw)
    X_train = apply_standardizer(X_train_raw, means, stds)
    X_val = apply_standardizer(X_val_raw, means, stds)

    # Grid search over L2 regularization candidates
    l2_candidates = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    best_candidate = None
    best_macro_f1 = -1.0
    candidate_results = []

    np.random.seed(20261103)

    for l2_reg in l2_candidates:
        init_params = np.zeros(X_train.shape[1] + 1, dtype=np.float64)
        res = minimize(
            fun=loss_and_grad,
            x0=init_params,
            args=(X_train, y_train, l2_reg),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 500, "ftol": 1e-9},
        )

        w = res.x[:-1]
        b = float(res.x[-1])

        # Validate
        z_val = np.dot(X_val, w) + b
        p_val = sigmoid(z_val)
        y_val_pred = (p_val >= 0.5).astype(int)
        val_metrics = calculate_metrics(y_val, y_val_pred)

        # Train metrics
        z_train = np.dot(X_train, w) + b
        p_train = sigmoid(z_train)
        y_train_pred = (p_train >= 0.5).astype(int)
        train_metrics = calculate_metrics(y_train, y_train_pred)

        record = {
            "l2_regularization": l2_reg,
            "train_macro_f1": train_metrics["macro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_confusion": val_metrics["confusion_matrix"],
            "weights_l2_norm": float(np.linalg.norm(w)),
        }
        candidate_results.append(record)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_candidate = {
                "l2_reg": l2_reg,
                "weights": w,
                "bias": b,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "optimization_result": res,
            }

    logger.info("Best L2 Candidate: %s (Val Macro-F1: %.4f)", best_candidate["l2_reg"], best_macro_f1)

    # Save model artifact
    best_w = best_candidate["weights"]
    best_b = best_candidate["bias"]

    weights_dict = {col: round(float(best_w[i]), 6) for i, col in enumerate(FEATURE_COLUMNS)}
    scaler_dict = {
        col: {"mean": round(float(means[i]), 6), "std": round(float(stds[i]), 6)}
        for i, col in enumerate(FEATURE_COLUMNS)
    }

    model_artifact = {
        "model_type": "LogisticRegression_L2",
        "stacker_version": "v1.0.0-phase8",
        "training_date": "2026-11-03",
        "random_seed": 20261103,
        "feature_schema": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "l2_regularization": best_candidate["l2_reg"],
        "intercept": round(best_b, 6),
        "weights": weights_dict,
        "scaler": scaler_dict,
        "training_samples_count": len(y_train),
        "validation_samples_count": len(y_val),
        "operating_threshold": 0.50,
        "score_type": "raw_stacker_logistic_output",
        "calibration_status": "uncalibrated",
    }

    models_dir = WORKSPACE_ROOT / "models" / "phase8"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_json_p = models_dir / "stacker_model.json"
    with open(model_json_p, "w", encoding="utf-8") as f:
        json.dump(model_artifact, f, indent=2)
    logger.info("Saved stacker model artifact to %s", model_json_p)

    # Save validation metrics report
    reports_dir = DATASET_ROOT / "reports" / "phase8"
    reports_dir.mkdir(parents=True, exist_ok=True)
    val_report_p = reports_dir / "stacker_validation_metrics.json"
    report_payload = {
        "model_version": "v1.0.0-phase8",
        "evaluation_partition": "validation (bank_progress layout family, 108 samples)",
        "held_out_test_partition_status": "untouched",
        "candidate_hyperparameters_evaluated": candidate_results,
        "selected_hyperparameters": {
            "l2_regularization": best_candidate["l2_reg"],
            "operating_threshold": 0.50,
        },
        "validation_metrics": best_candidate["val_metrics"],
        "training_metrics": best_candidate["train_metrics"],
        "learned_feature_weights": weights_dict,
    }
    with open(val_report_p, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    logger.info("Saved validation metrics to %s", val_report_p)

    return model_artifact, report_payload


if __name__ == "__main__":
    train_and_evaluate()

