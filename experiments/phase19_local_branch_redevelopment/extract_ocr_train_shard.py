"""
Phase 19 Sharded OCR Extraction for Training Partition.
Processes a shard of training images using EasyOCR, extracting candidate regions,
detailed fields, token boxes, and geometry features.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from PIL import Image

import torch
torch.set_num_threads(2)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.ocr_service import get_ocr_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase19.ocr_shard")


def extract_shard(shard_id: int, total_shards: int) -> Path:
    train_manifest = PROJECT_ROOT / "Hackathon-VIT" / "CBL04-Dataset" / "metadata" / "phase6" / "train_manifest.csv"
    with open(train_manifest) as fp:
        all_rows = list(csv.DictReader(fp))

    n_total = len(all_rows)
    chunk_size = (n_total + total_shards - 1) // total_shards
    start_idx = shard_id * chunk_size
    end_idx = min(start_idx + chunk_size, n_total)
    shard_rows = all_rows[start_idx:end_idx]

    logger.info("Shard %d/%d: processing rows %d to %d (%d images)...", shard_id, total_shards, start_idx, end_idx, len(shard_rows))

    ocr = get_ocr_service()
    out_dir = PROJECT_ROOT / "experiments" / "phase19_local_branch_redevelopment" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"train_ocr_shard_{shard_id}.json"

    records = []
    t_start = time.time()

    for idx, r in enumerate(shard_rows):
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
        ocr_res = ocr.extract(img_bytes)
        t_proc = time.perf_counter() - t0

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

        records.append({
            "sample_id": sample_id,
            "pair_id": pair_id,
            "label": label,
            "edit_category": cat,
            "layout_family": layout,
            "image_path": rel_img_path,
            "image_dimensions": [w_orig, h_orig],
            "ocr_overall_confidence": ocr_res.overall_confidence,
            "detailed_fields": detailed,
            "raw_tokens": raw_tokens,
            "processing_time_ms": round(t_proc * 1000.0, 2),
        })

        if (idx + 1) % 25 == 0 or idx == len(shard_rows) - 1:
            logger.info("Shard %d: processed %d / %d images (%.1fs elapsed, %.2fs/img)...",
                        shard_id, idx + 1, len(shard_rows), time.time() - t_start, (time.time() - t_start) / (idx + 1))
            # Save intermediate progress
            with open(out_file, "w") as fp:
                json.dump(records, fp, indent=2)

    logger.info("Shard %d finished! Saved %d records to %s in %.2fs", shard_id, len(records), out_file, time.time() - t_start)
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--total-shards", type=int, default=4)
    args = parser.parse_args()
    extract_shard(args.shard, args.total_shards)
