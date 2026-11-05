"""
Phase 17 Comprehensive Evaluation Script.
Evaluates the trained lightweight ResNet-18 crop classifier across the 108 validation images:
1. Target-field crop branch
2. OCR-derived Amount crop branch
3. OCR-derived Recipient crop branch
4. OCR-derived Transaction ID crop branch
5. Combined multi-crop branch (max pooling across fields)
Reports ROC-AUC, PR-AUC, FPR, Recall, Precision, Macro-F1, confusion matrices,
breakdown by edit category, and direct comparison to Global CNN baseline.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.models as models
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_BASE = PROJECT_ROOT / "experiments" / "phase17_visual_forensics" / "data"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase17.eval")


def roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    y_scores_sorted = y_scores[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_scores_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if fps[-1] <= 0 or tps[-1] <= 0:
        return 0.5

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    y_scores_sorted = y_scores[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_scores_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    precision = tps / (tps + fps)
    recall = tps / tps[-1] if tps[-1] > 0 else np.zeros_like(tps)

    recall = np.r_[0, recall]
    precision = np.r_[1, precision]
    return float(np.sum((recall[1:] - recall[:-1]) * (precision[1:] + precision[:-1]) / 2.0))


def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.50) -> Dict[str, Any]:
    preds = (y_scores >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))

    n = len(y_true)
    acc = (tp + tn) / n if n > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_edited = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    rec_clean = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec_clean = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1_clean = 2 * prec_clean * rec_clean / (prec_clean + rec_clean) if (prec_clean + rec_clean) > 0 else 0.0
    macro_f1 = (f1_edited + f1_clean) / 2.0

    auc_r = roc_auc(y_true, y_scores)
    auc_p = pr_auc(y_true, y_scores)

    clean_scores = y_scores[y_true == 0]
    edited_scores = y_scores[y_true == 1]

    return {
        "roc_auc": round(auc_r, 4),
        "pr_auc": round(auc_p, 4),
        "accuracy": round(acc, 4),
        "clean_fpr": round(fpr, 4),
        "edited_fnr": round(fnr, 4),
        "edited_recall": round(rec, 4),
        "precision": round(prec, 4),
        "macro_f1": round(macro_f1, 4),
        "clean_mean": round(float(np.mean(clean_scores)), 4) if len(clean_scores) else 0.0,
        "clean_std": round(float(np.std(clean_scores)), 4) if len(clean_scores) else 0.0,
        "edited_mean": round(float(np.mean(edited_scores)), 4) if len(edited_scores) else 0.0,
        "edited_std": round(float(np.std(edited_scores)), 4) if len(edited_scores) else 0.0,
        "mean_diff": round(float(np.mean(edited_scores) - np.mean(clean_scores)), 4) if len(clean_scores) else 0.0,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def main():
    device = torch.device("cpu")  # use CPU for deterministic reproducible evaluation
    ckpt_path = PROJECT_ROOT / "experiments" / "phase17_visual_forensics" / "checkpoints" / "best_crop_resnet18.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_manifest = PROJECT_ROOT / "Hackathon-VIT" / "CBL04-Dataset" / "metadata" / "phase6" / "validation_manifest.csv"
    with open(val_manifest) as fp:
        val_rows = list(csv.DictReader(fp))

    # Load crop dataset manifest to get target crops
    crop_manifest = OUT_BASE / "crop_dataset_manifest.csv"
    with open(crop_manifest) as fp:
        crop_rows = {r["sample_id"]: r for r in csv.DictReader(fp) if r["split"] == "val"}

    val_target_dir = OUT_BASE / "val_target_crops"
    val_ocr_crop_dir = OUT_BASE / "val_crops"

    eval_data = []

    with torch.no_grad():
        for r in val_rows:
            sample_id = r["source_id"] + ("_clean" if r["label"] == "clean" else "_edited")
            pair_id = r["pair_id"]
            label = r["label"]
            y_true_val = 1 if label == "edited" else 0
            cat = r.get("edit_category", "none")
            layout = r["layout_family"]

            # 1. Target Crop Prediction
            c_entry = crop_rows.get(sample_id)
            target_prob = 0.5
            if c_entry:
                c_path = PROJECT_ROOT / c_entry["crop_path"]
                if c_path.exists():
                    img = Image.open(c_path).convert("RGB")
                    tensor = transform(img).unsqueeze(0).to(device)
                    out = model(tensor)
                    target_prob = float(torch.softmax(out, dim=1)[0, 1].item())

            # 2. OCR Amount Crop
            amt_path = val_ocr_crop_dir / f"{sample_id}_amount.png"
            amt_prob = 0.5
            if amt_path.exists():
                img = Image.open(amt_path).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                out = model(tensor)
                amt_prob = float(torch.softmax(out, dim=1)[0, 1].item())

            # 3. OCR Receiver Crop
            rec_path = val_ocr_crop_dir / f"{sample_id}_receiver.png"
            rec_prob = 0.5
            if rec_path.exists():
                img = Image.open(rec_path).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                out = model(tensor)
                rec_prob = float(torch.softmax(out, dim=1)[0, 1].item())

            # 4. OCR Transaction ID Crop
            txn_path = val_ocr_crop_dir / f"{sample_id}_transaction_id.png"
            txn_prob = 0.5
            if txn_path.exists():
                img = Image.open(txn_path).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                out = model(tensor)
                txn_prob = float(torch.softmax(out, dim=1)[0, 1].item())

            # Combined multi-crop: max probability across fields
            comb_max_prob = max(amt_prob, rec_prob, txn_prob)
            comb_mean_prob = (amt_prob + rec_prob + txn_prob) / 3.0

            eval_data.append({
                "sample_id": sample_id,
                "pair_id": pair_id,
                "label": label,
                "y_true": y_true_val,
                "edit_category": cat,
                "layout_family": layout,
                "target_crop_prob": target_prob,
                "ocr_amount_prob": amt_prob,
                "ocr_receiver_prob": rec_prob,
                "ocr_txnid_prob": txn_prob,
                "combined_max_prob": comb_max_prob,
                "combined_mean_prob": comb_mean_prob,
            })

    y_arr = np.array([d["y_true"] for d in eval_data])

    # Evaluate branches
    branches = {
        "target_field_crop": np.array([d["target_crop_prob"] for d in eval_data]),
        "ocr_amount_crop": np.array([d["ocr_amount_prob"] for d in eval_data]),
        "ocr_receiver_crop": np.array([d["ocr_receiver_prob"] for d in eval_data]),
        "ocr_txnid_crop": np.array([d["ocr_txnid_prob"] for d in eval_data]),
        "combined_max_crop": np.array([d["combined_max_prob"] for d in eval_data]),
        "combined_mean_crop": np.array([d["combined_mean_prob"] for d in eval_data]),
    }

    branch_results = {}
    for b_name, scores in branches.items():
        branch_results[b_name] = compute_metrics(y_arr, scores)

    # Category breakdown for target_field_crop and combined_max_crop
    category_results = {}
    for cat in ["amount", "transaction_id", "recipient", "date_time", "other"]:
        sub = [d for d in eval_data if d["edit_category"] == cat or d["y_true"] == 0]
        # Evaluate pairing edited of this cat with clean
        sub_y = np.array([d["y_true"] for d in sub])
        sub_scores = np.array([d["target_crop_prob"] for d in sub])
        category_results[cat] = compute_metrics(sub_y, sub_scores)

    full_report = {
        "total_validation_samples": len(eval_data),
        "clean_samples": int(np.sum(y_arr == 0)),
        "edited_samples": int(np.sum(y_arr == 1)),
        "branch_metrics": branch_results,
        "category_metrics_target_crop": category_results,
        "global_cnn_baseline_comparison": {
            "global_cnn_resnet18": {
                "roc_auc": 0.4938,
                "pr_auc": 0.5028,
                "accuracy": 0.5000,
                "clean_fpr": 1.0000,
                "edited_recall": 1.0000,
                "precision": 0.5000,
                "macro_f1": 0.3333,
                "clean_mean": 0.7000,
                "edited_mean": 0.6995,
                "mean_diff": -0.0005,
            },
            "experimental_crop_resnet18_target": branch_results["target_field_crop"],
            "experimental_crop_resnet18_combined_max": branch_results["combined_max_crop"],
        },
    }

    out_json = OUT_BASE / "phase17_evaluation_results.json"
    with open(out_json, "w") as fp:
        json.dump(full_report, fp, indent=2)

    logger.info("Phase 17 evaluation complete. Report saved to %s", out_json)
    print(json.dumps(full_report, indent=2))
    return full_report


if __name__ == "__main__":
    main()
