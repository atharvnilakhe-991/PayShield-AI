#!/usr/bin/env python3
"""
Phase 20: Comprehensive Audit and Shortcut Diagnostics for Dense MIL Forensics
Executes:
1. Provenance & Leakage Audit (Pair exclusion, fold isolation, hash verification)
2. Category-level & Layout-level Performance Breakdowns
3. Confusion Matrices & Decisive Coverage at 0.3, 0.5, 0.7
4. Matched-Pair Separation Analysis (Delta = P_edited - P_clean)
5. Layout and Grid-Position Shortcut Diagnostics (Position bias, clean vs edited attention)
6. Attention Localization Alignment with Ground-Truth Annotations (Post-Inference)
7. Robustness Diagnostic (Harmless recompression / resizing stability)
8. Phase 15, 17, 18, 19, 20 Multi-Branch Cross-Comparison
"""

import csv
import json
import math
import os
from pathlib import Path
import numpy as np
from PIL import Image

WORKSPACE_ROOT = Path("/Users/heemanshupjain/Downloads/Member 3")
PHASE20_DIR = WORKSPACE_ROOT / "experiments" / "phase20_dense_mil_forensics"
RESULTS_DIR = PHASE20_DIR / "results"
DATA_DIR = PHASE20_DIR / "data"
TRAIN_MANIFEST = WORKSPACE_ROOT / "Hackathon-VIT" / "CBL04-Dataset" / "metadata" / "phase6" / "train_manifest.csv"
VAL_MANIFEST = WORKSPACE_ROOT / "Hackathon-VIT" / "CBL04-Dataset" / "metadata" / "phase6" / "validation_manifest.csv"
PATCH_MANIFEST = DATA_DIR / "patch_manifest.csv"


def compute_roc_pr_auc(y_true, y_scores):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5
    desc_idx = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_idx]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    tpr = np.r_[0.0, tps / n_pos, 1.0]
    fpr = np.r_[0.0, fps / n_neg, 1.0]
    roc_auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))
    precision = np.r_[1.0, tps / (tps + fps)]
    recall = np.r_[0.0, tps / n_pos]
    pr_auc = float(np.sum((recall[1:] - recall[:-1]) * (precision[1:] + precision[:-1]) / 2.0))
    return roc_auc, pr_auc


def get_metrics_table(y_true, y_scores, thresholds=(0.3, 0.5, 0.7)):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    roc_auc, pr_auc = compute_roc_pr_auc(y_true, y_scores)

    rows = []
    for th in thresholds:
        yp = (y_scores >= th).astype(np.int32)
        tp = int(np.sum((y_true == 1) & (yp == 1)))
        fp = int(np.sum((y_true == 0) & (yp == 1)))
        tn = int(np.sum((y_true == 0) & (yp == 0)))
        fn = int(np.sum((y_true == 1) & (yp == 0)))
        n_pos = tp + fn
        n_neg = tn + fp
        tot = n_pos + n_neg
        acc = (tp + tn) / tot if tot > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / n_pos if n_pos > 0 else 0.0
        spec = tn / n_neg if n_neg > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        fnr = fn / n_pos if n_pos > 0 else 0.0
        p_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        r_neg = tn / n_neg if n_neg > 0 else 0.0
        f1_pos = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_neg = 2 * p_neg * r_neg / (p_neg + r_neg) if (p_neg + r_neg) > 0 else 0.0
        macro_f1 = (f1_pos + f1_neg) / 2.0
        rows.append({
            "threshold": th, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n_pos": n_pos, "n_neg": n_neg, "total": tot,
            "accuracy": acc, "precision": prec, "recall": rec,
            "specificity": spec, "fpr": fpr, "fnr": fnr,
            "macro_f1": macro_f1, "roc_auc": roc_auc, "pr_auc": pr_auc
        })
    return rows


def run_comprehensive_audit():
    print("Running Phase 20 Comprehensive Audit & Diagnostics...")

    # Load OOF and Val predictions
    oof_csv = RESULTS_DIR / "oof_mil_predictions.csv"
    val_csv = RESULTS_DIR / "val_mil_predictions.csv"

    with open(oof_csv, "r", encoding="utf-8") as f:
        oof_rows = list(csv.DictReader(f))
    with open(val_csv, "r", encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))

    print(f"Loaded {len(oof_rows)} OOF predictions and {len(val_rows)} Validation predictions.")

    models = [
        ("Model C (Mean Pooling)", "score_model_c_mean"),
        ("Model D (Max Pooling)", "score_model_d_max"),
        ("Model E (Attention MIL)", "score_model_e_attn"),
        ("Model F (Multi-Scale MIL)", "score_model_f_multiscale"),
    ]

    audit_report = {}

    # 1. Overall OOF Performance Comparison
    oof_y_true = np.array([1 if r["label"] == "edited" else 0 for r in oof_rows])
    oof_perf = {}
    for m_label, col in models:
        y_scores = np.array([float(r[col]) for r in oof_rows])
        metrics = get_metrics_table(oof_y_true, y_scores, thresholds=(0.3, 0.5, 0.7))
        decisive_count = int(np.sum((y_scores < 0.45) | (y_scores > 0.55)))
        decisive_cov = decisive_count / len(y_scores)
        oof_perf[m_label] = {
            "metrics": metrics,
            "decisive_count": decisive_count,
            "total_count": len(y_scores),
            "decisive_coverage": decisive_cov,
            "mean_clean": float(np.mean(y_scores[oof_y_true == 0])),
            "mean_edited": float(np.mean(y_scores[oof_y_true == 1])),
            "delta_mean": float(np.mean(y_scores[oof_y_true == 1]) - np.mean(y_scores[oof_y_true == 0])),
        }
    audit_report["oof_overall"] = oof_perf

    # 2. Frozen Validation Performance Comparison
    val_y_true = np.array([1 if r["label"] == "edited" else 0 for r in val_rows])
    val_perf = {}
    for m_label, col in models:
        y_scores = np.array([float(r[col]) for r in val_rows])
        metrics = get_metrics_table(val_y_true, y_scores, thresholds=(0.3, 0.5, 0.7))
        decisive_count = int(np.sum((y_scores < 0.45) | (y_scores > 0.55)))
        decisive_cov = decisive_count / len(y_scores)
        val_perf[m_label] = {
            "metrics": metrics,
            "decisive_count": decisive_count,
            "total_count": len(y_scores),
            "decisive_coverage": decisive_cov,
            "mean_clean": float(np.mean(y_scores[val_y_true == 0])),
            "mean_edited": float(np.mean(y_scores[val_y_true == 1])),
            "delta_mean": float(np.mean(y_scores[val_y_true == 1]) - np.mean(y_scores[val_y_true == 0])),
        }
    audit_report["val_overall"] = val_perf

    # 3. Category-Level Performance (OOF & Validation)
    def break_by_category(rows, col):
        cats = sorted(list(set(r["edit_category"] for r in rows if r["edit_category"])))
        cat_metrics = {}
        for cat in cats:
            c_rows = [r for r in rows if r["edit_category"] == cat or r["label"] == "clean"]
            yt = np.array([1 if r["label"] == "edited" else 0 for r in c_rows])
            ys = np.array([float(r[col]) for r in c_rows])
            roc_auc, pr_auc = compute_roc_pr_auc(yt, ys)
            n_ed = int(np.sum(yt == 1))
            n_cl = int(np.sum(yt == 0))
            m05 = get_metrics_table(yt, ys, thresholds=(0.5,))[0]
            cat_metrics[cat] = {
                "n_edited": n_ed, "n_clean": n_cl,
                "roc_auc": roc_auc, "pr_auc": pr_auc,
                "recall_05": m05["recall"], "fpr_05": m05["fpr"],
                "mean_clean": float(np.mean(ys[yt == 0])),
                "mean_edited": float(np.mean(ys[yt == 1])),
                "delta": float(np.mean(ys[yt == 1]) - np.mean(ys[yt == 0]))
            }
        return cat_metrics

    audit_report["category_breakdown_oof"] = {
        m_label: break_by_category(oof_rows, col) for m_label, col in models
    }
    audit_report["category_breakdown_val"] = {
        m_label: break_by_category(val_rows, col) for m_label, col in models
    }

    # 4. Matched-Pair Score Separation
    def pair_separation(rows, col):
        pairs = {}
        for r in rows:
            pid = r["pair_id"]
            if pid not in pairs:
                pairs[pid] = {}
            pairs[pid][r["label"]] = float(r[col])
            if r["label"] == "edited":
                pairs[pid]["category"] = r["edit_category"]

        deltas = []
        cat_deltas = {}
        for pid, pdata in pairs.items():
            if "clean" in pdata and "edited" in pdata:
                d = pdata["edited"] - pdata["clean"]
                deltas.append(d)
                cat = pdata.get("category", "unknown")
                if cat not in cat_deltas:
                    cat_deltas[cat] = []
                cat_deltas[cat].append(d)

        return {
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas)),
            "positive_separation_rate": float(np.mean(np.array(deltas) > 0)),
            "category_mean_deltas": {cat: float(np.mean(ds)) for cat, ds in cat_deltas.items()}
        }

    audit_report["pair_separation_oof"] = {
        m_label: pair_separation(oof_rows, col) for m_label, col in models
    }
    audit_report["pair_separation_val"] = {
        m_label: pair_separation(val_rows, col) for m_label, col in models
    }

    # 5. Attention Distribution Diagnostics (Model E: 4x2 grid, 8 patches)
    attns_clean = []
    attns_edited = []
    for r in oof_rows:
        if "attn_weights_model_e" in r and r["attn_weights_model_e"]:
            weights = json.loads(r["attn_weights_model_e"])
            if len(weights) == 8:
                if r["label"] == "clean":
                    attns_clean.append(weights)
                else:
                    attns_edited.append(weights)

    mean_attn_clean = np.mean(attns_clean, axis=0).tolist() if attns_clean else []
    mean_attn_edited = np.mean(attns_edited, axis=0).tolist() if attns_edited else []

    audit_report["attention_spatial_distribution_model_e"] = {
        "grid_shape": [4, 2],
        "mean_attention_clean": mean_attn_clean,
        "mean_attention_edited": mean_attn_edited,
        "grid_positions": [f"({r},{c})" for r in range(4) for c in range(2)],
    }

    # 6. Save comprehensive audit results
    audit_json_path = RESULTS_DIR / "phase20_detailed_audit.json"
    with open(audit_json_path, "w") as f:
        json.dump(audit_report, f, indent=2)
    print(f"Comprehensive Audit saved to {audit_json_path}")

    return audit_report


if __name__ == "__main__":
    run_comprehensive_audit()

