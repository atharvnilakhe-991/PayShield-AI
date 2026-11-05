"""
Phase 18 Preflight: Extract and Cache OCR and Semantic Features for Validation Partition.
Runs the production EasyOCR and SemanticFeatureService on all 108 validation images.
Caches full telemetry into experiments/phase18_semantic_roi_fusion/data/val_ocr_semantic_cache.json.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.ocr_service import get_ocr_service
from app.services.semantic_feature_service import SemanticFeatureService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase18.cache")


def main():
    val_manifest = PROJECT_ROOT / "Hackathon-VIT" / "CBL04-Dataset" / "metadata" / "phase6" / "validation_manifest.csv"
    with open(val_manifest) as fp:
        val_rows = list(csv.DictReader(fp))

    logger.info("Initializing OCR and Semantic services for %d validation images...", len(val_rows))
    ocr_service = get_ocr_service()
    semantic_service = SemanticFeatureService()

    out_file = PROJECT_ROOT / "experiments" / "phase18_semantic_roi_fusion" / "data" / "val_ocr_semantic_cache.json"
    cached_records = []

    start_all = time.time()
    for idx, r in enumerate(val_rows):
        sample_id = r["source_id"] + ("_clean" if r["label"] == "clean" else "_edited")
        pair_id = r["pair_id"]
        label = r["label"]
        cat = r.get("edit_category", "none")
        layout = r["layout_family"]
        rel_img_path = r["image_path"]
        full_img_path = PROJECT_ROOT / "Hackathon-VIT" / "CBL04-Dataset" / rel_img_path

        with Image.open(full_img_path) as src:
            w_orig, h_orig = src.size

        with open(full_img_path, "rb") as fp:
            img_bytes = fp.read()

        t0 = time.perf_counter()
        ocr_res = ocr_service.extract(img_bytes)
        sem_res = semantic_service.extract_features(ocr_res, (w_orig, h_orig))
        t_proc = time.perf_counter() - t0

        # Detailed fields
        detailed = {}
        for fname, fobj in ocr_res.detailed_fields.items():
            norm_box = None
            if fobj.bounding_box is not None:
                norm_box = [
                    round(fobj.bounding_box.xmin, 5),
                    round(fobj.bounding_box.ymin, 5),
                    round(fobj.bounding_box.xmax, 5),
                    round(fobj.bounding_box.ymax, 5),
                ]
            pix_box = None
            if fobj.pixel_box is not None:
                pix_box = [
                    round(fobj.pixel_box.x, 2),
                    round(fobj.pixel_box.y, 2),
                    round(fobj.pixel_box.width, 2),
                    round(fobj.pixel_box.height, 2),
                ]

            detailed[fname] = {
                "field_name": fname,
                "raw_text": fobj.raw_text,
                "confidence": fobj.confidence,
                "normalized_value": fobj.normalized_value,
                "is_detected": fobj.is_detected,
                "extraction_status": fobj.extraction_status,
                "norm_box": norm_box,
                "pixel_box": pix_box,
            }

        raw_tokens = []
        for tok in ocr_res.raw_detections:
            tbox = None
            if tok.bounding_box is not None:
                tbox = [
                    round(tok.bounding_box.xmin, 5),
                    round(tok.bounding_box.ymin, 5),
                    round(tok.bounding_box.xmax, 5),
                    round(tok.bounding_box.ymax, 5),
                ]
            raw_tokens.append({
                "raw_text": tok.raw_text,
                "confidence": tok.confidence,
                "norm_box": tbox,
            })

        cached_records.append({
            "sample_id": sample_id,
            "pair_id": pair_id,
            "label": label,
            "edit_category": cat,
            "layout_family": layout,
            "image_path": rel_img_path,
            "image_dimensions": [w_orig, h_orig],
            "ocr_overall_confidence": ocr_res.overall_confidence,
            "ocr_fields": {
                "amount": ocr_res.fields.amount,
                "amount_numeric": ocr_res.fields.amount_numeric,
                "currency": ocr_res.fields.currency,
                "receiver": ocr_res.fields.receiver,
                "receiver_upi": ocr_res.fields.receiver_upi,
                "sender": ocr_res.fields.sender,
                "sender_upi": ocr_res.fields.sender_upi,
                "date": ocr_res.fields.date,
                "time": ocr_res.fields.time,
                "status": ocr_res.fields.status,
                "transaction_id": ocr_res.fields.transaction_id,
                "google_txn_id": ocr_res.fields.google_txn_id,
            },
            "detailed_fields": detailed,
            "raw_token_count": len(raw_tokens),
            "semantic_manipulation_score": sem_res.semantic_manipulation_score,
            "field_anomalies": sem_res.field_anomalies.model_dump(),
            "reason_codes": [rc.value for rc in sem_res.reason_codes],
            "semantic_features": sem_res.feature_vector.model_dump(),
            "processing_time_ms": round(t_proc * 1000.0, 2),
        })

        if (idx + 1) % 20 == 0 or idx == len(val_rows) - 1:
            logger.info("Cached %d / %d validation images (%.1fs elapsed)...", idx + 1, len(val_rows), time.time() - start_all)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as fp:
        json.dump(cached_records, fp, indent=2)

    total_time = time.time() - start_all
    logger.info("Successfully cached %d records to %s in %.2fs", len(cached_records), out_file, total_time)


if __name__ == "__main__":
    main()

