"""
Phase 17 Baseline Evaluation Script.
Evaluates simple forensic features and crop-based baselines on the 108-image validation partition.
Compares against the production Global CNN baseline.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.ocr_service import get_ocr_service
from experiments.phase17_visual_forensics.crop_extractor import NativeCropExtractor
from experiments.phase17_visual_forensics.simple_forensics import compute_crop_forensic_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase17.baseline")


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


def run_baseline_evaluation() -> dict:
    val_manifest = PROJECT_ROOT / "Hackathon-VIT" / "CBL04-Dataset" / "metadata" / "phase6" / "validation_manifest.csv"
    with open(val_manifest) as f:
        rows = list(csv.DictReader(f))

    logger.info("Loaded %d validation rows", len(rows))
    crop_out_dir = PROJECT_ROOT / "experiments" / "phase17_visual_forensics" / "data" / "val_crops"
    extractor = NativeCropExtractor(crop_out_dir)
    ocr = get_ocr_service()

    results = []
    y_true = []

    for idx, r in enumerate(rows):
        sample_id = r["source_id"] + ("_clean" if r["label"] == "clean" else "_edited")
        pair_id = r["pair_id"]
        label = r["label"]
        y_val = 1 if label == "edited" else 0
        y_true.append(y_val)
        cat = r.get("edit_category", "none")
        layout = r["layout_family"]
        img_rel_path = r["image_path"]
        img_full_path = PROJECT_ROOT / "Hackathon-VIT" / "CBL04-Dataset" / img_rel_path

        with Image.open(img_full_path) as src:
            orig_img = src.convert("RGB")
        w_orig, h_orig = orig_img.size

        # Run OCR to get bboxes
        with open(img_full_path, "rb") as fp:
            ocr_res = ocr.extract(fp.read())

        detailed = ocr_res.detailed_fields
        field_crops = {}
        forensics = {}

        for f_name, ocr_key in [("amount", "amount"), ("receiver", "receiver"), ("transaction_id", "transaction_id")]:
            bbox = None
            if ocr_key in detailed and detailed[ocr_key].bounding_box is not None:
                bb = detailed[ocr_key].bounding_box
                # Convert normalized to pixel coordinates
                x0 = int(round(bb.xmin * w_orig))
                y0 = int(round(bb.ymin * h_orig))
                x1 = int(round(bb.xmax * w_orig))
                y1 = int(round(bb.ymax * h_orig))
                bbox = (x0, y0, x1, y1)

            crop_img, meta = extractor.extract_field_crop(
                image=orig_img,
                field_name=f_name,
                bbox_xyxy=bbox,
                sample_id=sample_id,
                pair_id=pair_id,
                source_split="val",
                label=label,
                edit_category=cat,
                layout_family=layout,
                extraction_source="ocr" if bbox else "missing",
            )
            field_crops[f_name] = crop_img
            forensics[f_name] = compute_crop_forensic_features(crop_img)

        results.append({
            "sample_id": sample_id,
            "pair_id": pair_id,
            "label": label,
            "y_true": y_val,
            "edit_category": cat,
            "layout_family": layout,
            "forensics": forensics,
        })
        if (idx + 1) % 20 == 0 or idx == len(rows) - 1:
            logger.info("Processed %d / %d validation images...", idx + 1, len(rows))

    y_arr = np.array(y_true)

    # Evaluate feature separation for each forensic metric across each field
    field_metrics = {}
    for f_name in ["amount", "receiver", "transaction_id"]:
        field_metrics[f_name] = {}
        for metric in ["laplacian_energy", "edge_mean", "edge_std", "ela_mean", "bg_noise_std"]:
            scores = np.array([r["forensics"][f_name][metric] for r in results])
            c_scores = scores[y_arr == 0]
            e_scores = scores[y_arr == 1]
            auc_roc = roc_auc(y_arr, scores)
            auc_pr = pr_auc(y_arr, scores)
            field_metrics[f_name][metric] = {
                "clean_mean": float(np.mean(c_scores)),
                "clean_std": float(np.std(c_scores)),
                "edited_mean": float(np.mean(e_scores)),
                "edited_std": float(np.std(e_scores)),
                "roc_auc": round(auc_roc, 4),
                "pr_auc": round(auc_pr, 4),
                "mean_diff": round(float(np.mean(e_scores) - np.mean(c_scores)), 4),
            }

    # Combined simple heuristic score: sum of normalized Laplacian energy across all 3 fields
    combined_scores = []
    for r in results:
        comb = sum(r["forensics"][fn]["laplacian_energy"] for fn in ["amount", "receiver", "transaction_id"])
        combined_scores.append(comb)
    comb_arr = np.array(combined_scores)
    comb_auc_roc = roc_auc(y_arr, comb_arr)
    comb_auc_pr = pr_auc(y_arr, comb_arr)

    summary = {
        "sample_count": len(results),
        "clean_count": int(np.sum(y_arr == 0)),
        "edited_count": int(np.sum(y_arr == 1)),
        "field_metrics": field_metrics,
        "combined_laplacian_roc_auc": round(comb_auc_roc, 4),
        "combined_laplacian_pr_auc": round(comb_auc_pr, 4),
    }

    out_file = PROJECT_ROOT / "experiments" / "phase17_visual_forensics" / "data" / "baseline_evaluation_summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as fp:
        json.dump(summary, fp, indent=2)

    logger.info("Baseline evaluation complete. Summary saved to %s", out_file)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_baseline_evaluation()

