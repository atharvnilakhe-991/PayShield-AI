"""
Phase 8 Feature Extraction Script.
Extracts live OCR and Semantic consistency features on verified OOF training rows (868 samples)
and validation rows (108 samples), pairing them with genuine CNN probabilities.
Preserves test partition untouched.
Saves incremental checkpoints to disk.
"""

import csv
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

DATASET_ROOT = WORKSPACE_ROOT / "Hackathon-VIT" / "CBL04-Dataset"
sys.path.insert(0, str(DATASET_ROOT / "scripts" / "phase7"))

import torch
from torch.utils.data import DataLoader
from app.services.ocr_service import get_ocr_service
from app.services.semantic_feature_service import get_semantic_service
from model import build_model
from train_evaluate import ScreenshotDataset, infer, read_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase8_feature_extraction")


def compute_validation_cnn_predictions() -> Dict[str, Dict[str, Any]]:
    """
    Computes Fold 1 CNN predictions for the 108 validation samples (bank_progress family).
    Fold 1 was trained on enriched and recharge; bank_progress was its exact validation set.
    """
    ckpt_path = DATASET_ROOT / "models" / "phase7" / "fold_1_resnet18_multitask.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing Fold 1 checkpoint at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    val_csv = DATASET_ROOT / "metadata" / "phase7" / "folds" / "fold_1_validation.csv"
    val_rows = read_csv(val_csv)
    loader = DataLoader(ScreenshotDataset(val_rows), batch_size=16, shuffle=False)
    preds, _ = infer(model, loader)

    pred_map = {}
    for p in preds:
        img_p = p["image_path"].replace("\\", "/").strip()
        pred_map[img_p] = p
    logger.info("Computed %d validation CNN predictions with Fold 1 model", len(pred_map))
    return pred_map


def load_existing_sample_ids(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {r["sample_id"] for r in reader}


def extract_features_for_partition(
    manifest_path: Path,
    output_path: Path,
    cnn_predictions_override: Optional[Dict[str, Dict[str, Any]]] = None,
    partition_name: str = "train",
):
    logger.info("Processing partition: %s (manifest: %s)", partition_name, manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    existing_ids = load_existing_sample_ids(output_path)
    logger.info(
        "Found %d manifest rows, %d already processed in %s",
        len(manifest_rows),
        len(existing_ids),
        output_path.name,
    )

    fieldnames = [
        "sample_id",
        "true_label",
        "cnn_score",
        "cnn_pred_class",
        "cnn_status",
        "ocr_overall_confidence",
        "ocr_amount_extracted",
        "ocr_recipient_extracted",
        "ocr_ref_id_extracted",
        "ocr_date_extracted",
        "ocr_status",
        "semantic_manipulation_score",
        "amount_anomaly",
        "recipient_anomaly",
        "date_anomaly",
        "ref_id_anomaly",
        "status_anomaly",
        "upi_handle_anomaly",
        "missing_critical_fields_count",
        "semantic_status",
        "branch_cnn_available",
        "branch_ocr_available",
        "branch_semantic_available",
    ]

    write_header = not output_path.exists() or output_path.stat().st_size == 0
    fp = open(output_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fp, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        fp.flush()

    ocr_service = get_ocr_service()
    semantic_service = get_semantic_service()

    clean_dir = DATASET_ROOT / "data" / "synthetic_gpay_phase4" / "clean"
    edited_dir = DATASET_ROOT / "data" / "synthetic_gpay_phase4" / "edited"

    processed_count = 0
    for idx, row in enumerate(manifest_rows):
        sample_id = row["sample_id"]
        if sample_id in existing_ids:
            continue

        true_label = 1 if row["true_label"] in ["1", "edited"] else 0
        is_clean = true_label == 0
        img_name = f"{sample_id}.png"
        img_path = (clean_dir if is_clean else edited_dir) / img_name
        rel_img_path = f"data/synthetic_gpay_phase4/{'clean' if is_clean else 'edited'}/{img_name}"

        # 1. CNN Branch
        if cnn_predictions_override and rel_img_path in cnn_predictions_override:
            cnn_pred = cnn_predictions_override[rel_img_path]
            cnn_score = float(cnn_pred["edited_probability"])
            cnn_pred_class = int(cnn_pred["predicted_label"])
            cnn_status = "live_val_fold_1"
        elif row.get("cnn_score"):
            cnn_score = float(row["cnn_score"])
            cnn_pred_class = int(row.get("cnn_predicted_class", 1 if cnn_score >= 0.5 else 0))
            cnn_status = row.get("cnn_oof_status", "valid_oof")
        else:
            cnn_score = 0.5
            cnn_pred_class = 0
            cnn_status = "unavailable"

        # 2. OCR Branch
        if not img_path.exists():
            logger.error("Image file missing: %s", img_path)
            continue

        with open(img_path, "rb") as img_fp:
            img_bytes = img_fp.read()

        try:
            ocr_res = ocr_service.extract(img_bytes)
            ocr_conf = ocr_res.overall_confidence
            ocr_status = "extracted"
            has_amt = 1 if (ocr_res.fields.amount is not None or ocr_res.fields.amount_numeric is not None) else 0
            has_rec = 1 if (ocr_res.fields.receiver is not None or ocr_res.fields.receiver_upi is not None) else 0
            has_ref = 1 if (ocr_res.fields.transaction_id is not None or ocr_res.fields.google_txn_id is not None) else 0
            has_date = 1 if (ocr_res.fields.date is not None or ocr_res.fields.time is not None) else 0
            ocr_avail = 1
        except Exception as ocr_exc:
            logger.warning("OCR failed for %s: %s", sample_id, ocr_exc)
            ocr_conf = 0.0
            ocr_status = "failed"
            has_amt, has_rec, has_ref, has_date = 0, 0, 0, 0
            ocr_avail = 0
            ocr_res = None

        # 3. Semantic Branch
        if ocr_res is not None:
            try:
                sem_res = semantic_service.extract_features(ocr_res)
                sem_score = sem_res.semantic_manipulation_score
                anom = sem_res.field_anomalies
                amt_anom = 1 if anom.amount_anomaly else 0
                rec_anom = 1 if anom.recipient_anomaly else 0
                date_anom = 1 if anom.date_anomaly else 0
                ref_anom = 1 if anom.ref_id_anomaly else 0
                stat_anom = 1 if anom.status_anomaly else 0
                upi_anom = 1 if anom.upi_handle_anomaly else 0
                missing_cnt = sem_res.feature_vector.missing_critical_fields_count
                sem_status = "evaluated"
                sem_avail = 1
            except Exception as sem_exc:
                logger.warning("Semantic analysis failed for %s: %s", sample_id, sem_exc)
                sem_score = 0.0
                amt_anom, rec_anom, date_anom, ref_anom, stat_anom, upi_anom = 0, 0, 0, 0, 0, 0
                missing_cnt = 4
                sem_status = "failed"
                sem_avail = 0
        else:
            sem_score = 0.0
            amt_anom, rec_anom, date_anom, ref_anom, stat_anom, upi_anom = 0, 0, 0, 0, 0, 0
            missing_cnt = 4
            sem_status = "not_run"
            sem_avail = 0

        feat_row = {
            "sample_id": sample_id,
            "true_label": true_label,
            "cnn_score": round(cnn_score, 4),
            "cnn_pred_class": cnn_pred_class,
            "cnn_status": cnn_status,
            "ocr_overall_confidence": round(ocr_conf, 4),
            "ocr_amount_extracted": has_amt,
            "ocr_recipient_extracted": has_rec,
            "ocr_ref_id_extracted": has_ref,
            "ocr_date_extracted": has_date,
            "ocr_status": ocr_status,
            "semantic_manipulation_score": round(sem_score, 4),
            "amount_anomaly": amt_anom,
            "recipient_anomaly": rec_anom,
            "date_anomaly": date_anom,
            "ref_id_anomaly": ref_anom,
            "status_anomaly": stat_anom,
            "upi_handle_anomaly": upi_anom,
            "missing_critical_fields_count": missing_cnt,
            "semantic_status": sem_status,
            "branch_cnn_available": 1 if cnn_status != "unavailable" else 0,
            "branch_ocr_available": ocr_avail,
            "branch_semantic_available": sem_avail,
        }

        writer.writerow(feat_row)
        fp.flush()
        existing_ids.add(sample_id)
        processed_count += 1

        if processed_count % 25 == 0 or idx == len(manifest_rows) - 1:
            logger.info("[%s] Processed %d / %d samples...", partition_name, len(existing_ids), len(manifest_rows))

    fp.close()
    logger.info("Completed %s feature extraction: %d samples in %s", partition_name, len(existing_ids), output_path)


def main():
    logger.info("Starting Phase 8 Branch Feature Extraction...")

    # 1. Extract Validation Features (108 samples)
    val_manifest = DATASET_ROOT / "metadata" / "phase7_branch_validation" / "validation_manifest.csv"
    val_output = DATASET_ROOT / "metadata" / "phase8" / "validation_features.csv"
    val_cnn_preds = compute_validation_cnn_predictions()
    extract_features_for_partition(
        manifest_path=val_manifest,
        output_path=val_output,
        cnn_predictions_override=val_cnn_preds,
        partition_name="validation",
    )

    # 2. Extract Training Features (868 samples)
    train_manifest = DATASET_ROOT / "metadata" / "phase7_branch_validation" / "valid_oof_training_manifest.csv"
    train_output = DATASET_ROOT / "metadata" / "phase8" / "oof_stacker_train_features.csv"
    extract_features_for_partition(
        manifest_path=train_manifest,
        output_path=train_output,
        cnn_predictions_override=None,
        partition_name="train",
    )

    logger.info("Phase 8 Feature Extraction successfully finished!")


if __name__ == "__main__":
    main()
