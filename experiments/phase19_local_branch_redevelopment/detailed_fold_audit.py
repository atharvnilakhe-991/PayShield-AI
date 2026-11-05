"""
Phase 19: Detailed Fold Audit and Prediction Quality Inspection.
Computes:
1. Per-fold training metadata (counts, weights, best epochs, losses, AUCs, SHA-256)
2. Fold 0 (and per-fold) diagnostic threshold metrics (0.3, 0.5, 0.7):
   Precision, Recall, Specificity, FPR, FNR, Macro-F1, PR-AUC, ROC-AUC
3. Cross-fold consistency: Mean, Std, Min, Max of ROC-AUC and PR-AUC
4. Performance by candidate type (amount, receiver, txn_id, date) and edit category
5. OOF prediction audit: exactly 3,290 eligible crops, missing count, duplicate count, provenance
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = PROJECT_ROOT / "experiments" / "phase19_local_branch_redevelopment"
DATA_DIR = EXP_DIR / "data"
RESULTS_DIR = EXP_DIR / "results"
CKPT_DIR = EXP_DIR / "checkpoints"


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


def compute_metrics_at_thresholds(y_true: np.ndarray, y_scores: np.ndarray, thresholds=(0.3, 0.5, 0.7)):
    results = {}
    roc_auc = compute_roc_auc(y_true, y_scores)
    pr_auc = compute_pr_auc(y_true, y_scores)
    prevalence = float(np.mean(y_true))

    for th in thresholds:
        y_pred = (y_scores >= th).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        n_pos = tp + fn
        n_neg = tn + fp
        total = len(y_true)

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

        results[f"th_{th:.1f}"] = {
            "threshold": th,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "specificity": round(spec, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "f1_score": round(f1, 4),
            "macro_f1": round(macro_f1, 4),
        }

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "positive_prevalence": round(prevalence, 4),
        "threshold_metrics": results,
    }


def main():
    # 1. Load Provenance
    prov_file = CKPT_DIR / "fold_checkpoint_provenance.json"
    with open(prov_file) as f:
        prov = json.load(f)

    print("=== PART 1: 5-FOLD TRAINING RECORD ===")
    for p in prov:
        print(f"Fold {p['fold']}:")
        print(f"  Train crops: {p['train_sample_count']} (clean={p['train_negative_count']}, tampered={p['train_positive_count']})")
        print(f"  Held-out crops: {p['held_out_sample_count']} (clean={p['held_out_negative_count']}, tampered={p['held_out_positive_count']})")
        print(f"  Epochs completed: {p['epoch_count']}, Best epoch: {p['best_epoch']}")
        print(f"  Best val loss: {p['best_held_out_val_loss']}, Best val ROC-AUC: {p['best_held_out_val_roc_auc']}")
        print(f"  Training duration: {p['training_duration_seconds']}s")
        print(f"  Checkpoint: {p['checkpoint_path']}")
        print(f"  SHA-256: {p['sha256']}")

    # 2. Load OOF crop predictions
    oof_crop_p = RESULTS_DIR / "oof_crop_predictions.csv"
    with open(oof_crop_p) as f:
        oof_crops = list(csv.DictReader(f))

    manifest_p = DATA_DIR / "crop_dataset_manifest.csv"
    with open(manifest_p) as f:
        all_manifest = list(csv.DictReader(f))

    eligible_train_crops = [r for r in all_manifest if r["split"] == "train" and r.get("is_ambiguous") != "True" and int(r["crop_label"]) in (0, 1)]
    expected_ids = set(r["crop_id"] for r in eligible_train_crops)
    actual_ids = [r["crop_id"] for r in oof_crops]

    print("\n=== PART 4: OOF PREDICTION AUDIT ===")
    print("Expected eligible crops:", len(expected_ids))
    print("Actual OOF predictions:", len(actual_ids))
    print("Unique actual crop IDs:", len(set(actual_ids)))
    print("Missing predictions:", len(expected_ids - set(actual_ids)))
    print("Duplicate predictions:", len(actual_ids) - len(set(actual_ids)))

    # Missing manifest
    missing_manifest_p = RESULTS_DIR / "missing_oof_predictions_manifest.csv"
    with open(missing_manifest_p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_id", "sample_id", "pair_id", "fold"])
        writer.writeheader()
        for cid in sorted(expected_ids - set(actual_ids)):
            writer.writerow({"crop_id": cid})
    print(f"Saved missing predictions manifest to {missing_manifest_p} (0 rows expected)")

    # 3. Diagnostic Threshold Evaluation per Fold (and Fold 0 in detail)
    print("\n=== PART 2: FOLD 0 PREDICTION QUALITY INSPECTION ===")
    f0_crops = [r for r in oof_crops if int(r["fold"]) == 0]
    y_f0 = np.array([int(r["crop_label"]) for r in f0_crops])
    s_f0 = np.array([float(r["p_crop_tampered"]) for r in f0_crops])
    f0_metrics = compute_metrics_at_thresholds(y_f0, s_f0)
    print(f"Fold 0 ROC-AUC: {f0_metrics['roc_auc']}, PR-AUC: {f0_metrics['pr_auc']}, Prevalence: {f0_metrics['positive_prevalence']}")
    for th_k, th_v in f0_metrics["threshold_metrics"].items():
        print(f"  At threshold {th_v['threshold']}: TP={th_v['tp']}, FP={th_v['fp']}, TN={th_v['tn']}, FN={th_v['fn']}, "
              f"Prec={th_v['precision']}, Rec={th_v['recall']}, Spec={th_v['specificity']}, FPR={th_v['fpr']}, FNR={th_v['fnr']}, Macro-F1={th_v['macro_f1']}")

    # Clean vs tampered score distributions in Fold 0
    clean_scores_f0 = s_f0[y_f0 == 0]
    tamp_scores_f0 = s_f0[y_f0 == 1]
    print(f"  Fold 0 Clean Scores: Mean={np.mean(clean_scores_f0):.4f}, Std={np.std(clean_scores_f0):.4f}, Median={np.median(clean_scores_f0):.4f}")
    print(f"  Fold 0 Tampered Scores: Mean={np.mean(tamp_scores_f0):.4f}, Std={np.std(tamp_scores_f0):.4f}, Median={np.median(tamp_scores_f0):.4f}")

    # 4. Cross-Fold Consistency
    print("\n=== PART 3: CROSS-FOLD CONSISTENCY ===")
    fold_aucs = []
    fold_pr_aucs = []
    fold_recalls_05 = []
    fold_fprs_05 = []

    for f in range(5):
        f_crops = [r for r in oof_crops if int(r["fold"]) == f]
        y_f = np.array([int(r["crop_label"]) for r in f_crops])
        s_f = np.array([float(r["p_crop_tampered"]) for r in f_crops])
        m = compute_metrics_at_thresholds(y_f, s_f)
        auc = m["roc_auc"]
        pr_auc = m["pr_auc"]
        rec = m["threshold_metrics"]["th_0.5"]["recall"]
        fpr = m["threshold_metrics"]["th_0.5"]["fpr"]
        fold_aucs.append(auc)
        fold_pr_aucs.append(pr_auc)
        fold_recalls_05.append(rec)
        fold_fprs_05.append(fpr)
        print(f"Fold {f}: ROC-AUC={auc:.4f}, PR-AUC={pr_auc:.4f}, Recall@0.5={rec:.4f}, FPR@0.5={fpr:.4f}")

    print(f"\nROC-AUC across folds: Mean={np.mean(fold_aucs):.4f}, Std={np.std(fold_aucs):.4f}, Min={np.min(fold_aucs):.4f}, Max={np.max(fold_aucs):.4f}")
    print(f"PR-AUC across folds:  Mean={np.mean(fold_pr_aucs):.4f}, Std={np.std(fold_pr_aucs):.4f}, Min={np.min(fold_pr_aucs):.4f}, Max={np.max(fold_pr_aucs):.4f}")

    # 5. Breakdown by Candidate Field and Edit Category across all 3,290 OOF crops
    print("\n--- Performance by Candidate Field (OOF) ---")
    for fld in ["amount", "receiver", "transaction_id", "date"]:
        fld_crops = [r for r in oof_crops if r["field_name"] == fld]
        y_fld = np.array([int(r["crop_label"]) for r in fld_crops])
        s_fld = np.array([float(r["p_crop_tampered"]) for r in fld_crops])
        if len(np.unique(y_fld)) > 1:
            auc = compute_roc_auc(y_fld, s_fld)
            pr = compute_pr_auc(y_fld, s_fld)
            print(f"  {fld} ({len(fld_crops)} crops, {int(np.sum(y_fld))} tampered): ROC-AUC={auc:.4f}, PR-AUC={pr:.4f}")
        else:
            print(f"  {fld} ({len(fld_crops)} crops, {int(np.sum(y_fld))} tampered): No positive/negative class diversity")

    # Save detailed audit report JSON
    audit_data = {
        "fold_provenance": prov,
        "oof_completeness": {
            "expected_eligible_crops": len(expected_ids),
            "actual_oof_crops": len(actual_ids),
            "missing_count": len(expected_ids - set(actual_ids)),
            "duplicate_count": len(actual_ids) - len(set(actual_ids)),
        },
        "fold_0_diagnostic": f0_metrics,
        "cross_fold_summary": {
            "roc_auc_mean": round(float(np.mean(fold_aucs)), 4),
            "roc_auc_std": round(float(np.std(fold_aucs)), 4),
            "roc_auc_min": round(float(np.min(fold_aucs)), 4),
            "roc_auc_max": round(float(np.max(fold_aucs)), 4),
            "pr_auc_mean": round(float(np.mean(fold_pr_aucs)), 4),
            "pr_auc_std": round(float(np.std(fold_pr_aucs)), 4),
            "pr_auc_min": round(float(np.min(fold_pr_aucs)), 4),
            "pr_auc_max": round(float(np.max(fold_pr_aucs)), 4),
        }
    }
    with open(RESULTS_DIR / "detailed_training_oof_audit.json", "w") as f:
        json.dump(audit_data, f, indent=2)


if __name__ == "__main__":
    main()

