"""
Phase 19: Merge OCR Shards and Build Leakage-Safe Crop Dataset.
1. Merges sharded EasyOCR extractions for all 868 training images into data/train_ocr_cache.json.
2. Uses pre-audited Grouped Stratified 5-Fold split (data/fold_assignments.csv).
3. Extracts inference-realistic candidate crops strictly using EasyOCR bounding boxes.
   Zero ground-truth crop coordinates or fallbacks.
4. Assigns crop labels:
   - Clean screenshots: all candidate crops -> label 0 (clean)
   - Edited screenshots:
     - Clear match (IoU > 0.3 or GT coverage > 0.5) -> label 1 (tampered)
     - Clear non-overlap (IoU < 0.05 and GT coverage < 0.05) -> label 0 (hard negative clean)
     - Ambiguous boundary overlap (0.05 <= IoU <= 0.3 or 0.05 <= coverage <= 0.5) -> marked is_ambiguous=True, counted, and excluded from training.
5. Generates full crop audit manifest and routing diagnostics.
6. Also extracts validation candidate crops (108 bank_progress images) using the exact same pipeline.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase19.build_dataset")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = PROJECT_ROOT / "experiments" / "phase19_local_branch_redevelopment"
DATA_DIR = EXP_DIR / "data"
CROPS_DIR = EXP_DIR / "crops"
DATASET_ROOT = PROJECT_ROOT / "Hackathon-VIT" / "CBL04-Dataset"
FA_DIR = DATASET_ROOT / "annotations" / "phase4" / "field_annotations"

TARGET_CANVAS = (256, 128)  # width=256, height=128
BACKGROUND_RGB = (18, 18, 18)

CONTEXT_MARGINS = {
    "amount": (0.15, 0.25),
    "receiver": (0.10, 0.20),
    "transaction_id": (0.10, 0.25),
    "date": (0.10, 0.20),
}


def compute_box_overlap(box1: List[int], box2: List[int]) -> Tuple[float, float]:
    x0 = max(box1[0], box2[0])
    y0 = max(box1[1], box2[1])
    x1 = min(box1[2], box2[2])
    y1 = min(box1[3], box2[3])

    iw = max(0, x1 - x0)
    ih = max(0, y1 - y0)
    area_inter = iw * ih

    area1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    area_union = area1 + area2 - area_inter

    iou = area_inter / max(1.0, float(area_union))
    cov2 = area_inter / max(1.0, float(area2))
    return round(iou, 4), round(cov2, 4)


def extract_standardized_crop(img: Image.Image, bbox: List[int], margin_x: float, margin_y: float) -> Image.Image:
    w_orig, h_orig = img.size
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w_orig, x1), min(h_orig, y1)
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)

    pad_x = int(round(bw * margin_x))
    pad_y = int(round(bh * margin_y))
    cx0, cy0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    cx1, cy1 = min(w_orig, x1 + pad_x), min(h_orig, y1 + pad_y)

    patch = img.crop((cx0, cy0, cx1, cy1))
    pw, ph = patch.size

    scale = min(TARGET_CANVAS[0] / pw, TARGET_CANVAS[1] / ph)
    new_w = max(1, int(round(pw * scale)))
    new_h = max(1, int(round(ph * scale)))
    resized = patch.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", TARGET_CANVAS, BACKGROUND_RGB)
    canvas.paste(resized, ((TARGET_CANVAS[0] - new_w) // 2, (TARGET_CANVAS[1] - new_h) // 2))
    return canvas


def merge_shards() -> List[Dict[str, Any]]:
    merged = []
    seen = set()
    for shard_idx in range(4):
        shard_path = DATA_DIR / f"train_ocr_shard_{shard_idx}.json"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing {shard_path}")
        with open(shard_path) as fp:
            records = json.load(fp)
        logger.info("Loaded shard %d with %d records", shard_idx, len(records))
        for r in records:
            if r["sample_id"] not in seen:
                seen.add(r["sample_id"])
                merged.append(r)
            else:
                logger.warning("Duplicate record in shard %d: %s", shard_idx, r["sample_id"])

    assert len(merged) == 868, f"Expected 868 records, found {len(merged)}"
    out_cache = DATA_DIR / "train_ocr_cache.json"
    with open(out_cache, "w") as fp:
        json.dump(merged, fp, indent=2)

    h = hashlib.sha256(out_cache.read_bytes()).hexdigest()
    logger.info("Saved merged OCR cache (%d records) to %s (SHA-256: %s)", len(merged), out_cache, h)
    return merged


def load_fold_assignments() -> Dict[str, int]:
    fold_csv = DATA_DIR / "fold_assignments.csv"
    assert fold_csv.exists(), f"Missing {fold_csv}"
    pair_to_fold = {}
    with open(fold_csv) as fp:
        for r in csv.DictReader(fp):
            pair_to_fold[r["pair_id"]] = int(r["fold"])
    assert len(pair_to_fold) == 434
    return pair_to_fold


def build_crop_dataset(train_records: List[Dict[str, Any]], pair_to_fold: Dict[str, int]):
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    # Diagnostics trackers
    total_edited_screenshots = 0
    routing_hits = 0
    routing_misses_by_cat: Dict[str, int] = {}
    routing_hits_by_cat: Dict[str, int] = {}
    ambiguous_crop_count = 0

    crop_counts_by_field = {"amount": 0, "receiver": 0, "transaction_id": 0, "date": 0}
    crop_counts_by_label = {"clean": 0, "tampered": 0, "ambiguous": 0}
    crop_counts_by_layout = {}

    for r in train_records:
        sample_id = r["sample_id"]
        pair_id = r["pair_id"]
        fold = pair_to_fold[pair_id]
        img_label = r["label"]
        edit_cat = r.get("edit_category", "none")
        layout = r["layout_family"]
        rel_img_path = r["image_path"]
        full_img_path = DATASET_ROOT / rel_img_path
        w_img, h_img = r["image_dimensions"]
        detailed = r.get("detailed_fields", {})

        crop_counts_by_layout[layout] = crop_counts_by_layout.get(layout, 0)

        # Load GT annotation if edited
        gt_boxes = []
        if img_label == "edited":
            total_edited_screenshots += 1
            fa_json = FA_DIR / f"{pair_id}.json"
            if fa_json.exists():
                with open(fa_json) as fp:
                    meta = json.load(fp)
                gt_boxes = meta.get("bounding_boxes_xyxy", [])

        img = None
        sample_has_hit = False

        for field_name in ["amount", "receiver", "transaction_id", "date"]:
            finfo = detailed.get(field_name, {})
            norm_box = finfo.get("norm_box")
            if norm_box is None:
                continue

            x0 = int(round(norm_box[0] * w_img))
            y0 = int(round(norm_box[1] * h_img))
            x1 = int(round(norm_box[2] * w_img))
            y1 = int(round(norm_box[3] * h_img))
            raw_pixel_box = [x0, y0, x1, y1]

            max_iou = 0.0
            max_cov = 0.0
            for gt_b in gt_boxes:
                iou, cov = compute_box_overlap(raw_pixel_box, gt_b)
                if iou > max_iou:
                    max_iou = iou
                if cov > max_cov:
                    max_cov = cov

            # Rigorous label assignment
            is_ambiguous = False
            if img_label == "clean":
                crop_label = 0
                is_hit = False
                crop_counts_by_label["clean"] += 1
            else:
                if max_iou > 0.3 or max_cov > 0.5:
                    crop_label = 1
                    is_hit = True
                    sample_has_hit = True
                    crop_counts_by_label["tampered"] += 1
                elif max_iou < 0.05 and max_cov < 0.05:
                    crop_label = 0
                    is_hit = False
                    crop_counts_by_label["clean"] += 1
                else:
                    # Ambiguous boundary overlap
                    crop_label = -1
                    is_hit = False
                    is_ambiguous = True
                    ambiguous_crop_count += 1
                    crop_counts_by_label["ambiguous"] += 1

            if img is None:
                with Image.open(full_img_path) as src:
                    img = src.convert("RGB")

            mx, my = CONTEXT_MARGINS.get(field_name, (0.15, 0.20))
            crop_canvas = extract_standardized_crop(img, raw_pixel_box, mx, my)
            crop_filename = f"{sample_id}_{field_name}.png"
            crop_out_path = CROPS_DIR / crop_filename
            crop_canvas.save(crop_out_path)

            crop_counts_by_field[field_name] += 1
            crop_counts_by_layout[layout] += 1

            manifest_rows.append({
                "crop_id": f"{sample_id}_{field_name}",
                "sample_id": sample_id,
                "pair_id": pair_id,
                "fold": fold,
                "split": "train",
                "image_label": img_label,
                "crop_label": crop_label,
                "is_ambiguous": is_ambiguous,
                "field_name": field_name,
                "edit_category": edit_cat,
                "layout_family": layout,
                "crop_rel_path": f"crops/{crop_filename}",
                "norm_box": json.dumps(norm_box),
                "pixel_box": json.dumps(raw_pixel_box),
                "max_iou": max_iou,
                "max_gt_coverage": max_cov,
                "is_routing_hit": is_hit,
            })

        if img_label == "edited":
            if sample_has_hit:
                routing_hits += 1
                routing_hits_by_cat[edit_cat] = routing_hits_by_cat.get(edit_cat, 0) + 1
            else:
                routing_misses_by_cat[edit_cat] = routing_misses_by_cat.get(edit_cat, 0) + 1

    # Validation records
    val_cache_path = PROJECT_ROOT / "experiments" / "phase18_semantic_roi_fusion" / "data" / "val_ocr_semantic_cache.json"
    with open(val_cache_path) as fp:
        val_records = json.load(fp)

    logger.info("Processing %d validation records...", len(val_records))

    for r in val_records:
        sample_id = r["sample_id"]
        pair_id = r["pair_id"]
        fold = -1
        img_label = r["label"]
        edit_cat = r.get("edit_category", "none")
        layout = r["layout_family"]
        rel_img_path = r["image_path"]
        full_img_path = DATASET_ROOT / rel_img_path
        w_img, h_img = r["image_dimensions"]
        detailed = r.get("detailed_fields", {})

        gt_boxes = []
        if img_label == "edited":
            fa_json = FA_DIR / f"{pair_id}.json"
            if fa_json.exists():
                with open(fa_json) as fp:
                    meta = json.load(fp)
                gt_boxes = meta.get("bounding_boxes_xyxy", [])

        img = None

        for field_name in ["amount", "receiver", "transaction_id", "date"]:
            finfo = detailed.get(field_name, {})
            norm_box = finfo.get("norm_box")
            if norm_box is None:
                continue

            x0 = int(round(norm_box[0] * w_img))
            y0 = int(round(norm_box[1] * h_img))
            x1 = int(round(norm_box[2] * w_img))
            y1 = int(round(norm_box[3] * h_img))
            raw_pixel_box = [x0, y0, x1, y1]

            max_iou = 0.0
            max_cov = 0.0
            for gt_b in gt_boxes:
                iou, cov = compute_box_overlap(raw_pixel_box, gt_b)
                if iou > max_iou:
                    max_iou = iou
                if cov > max_cov:
                    max_cov = cov

            is_ambiguous = False
            if img_label == "clean":
                crop_label = 0
                is_hit = False
            else:
                if max_iou > 0.3 or max_cov > 0.5:
                    crop_label = 1
                    is_hit = True
                elif max_iou < 0.05 and max_cov < 0.05:
                    crop_label = 0
                    is_hit = False
                else:
                    crop_label = -1
                    is_hit = False
                    is_ambiguous = True

            if img is None:
                with Image.open(full_img_path) as src:
                    img = src.convert("RGB")

            mx, my = CONTEXT_MARGINS.get(field_name, (0.15, 0.20))
            crop_canvas = extract_standardized_crop(img, raw_pixel_box, mx, my)
            crop_filename = f"{sample_id}_{field_name}.png"
            crop_out_path = CROPS_DIR / crop_filename
            crop_canvas.save(crop_out_path)

            manifest_rows.append({
                "crop_id": f"{sample_id}_{field_name}",
                "sample_id": sample_id,
                "pair_id": pair_id,
                "fold": fold,
                "split": "val",
                "image_label": img_label,
                "crop_label": crop_label,
                "is_ambiguous": is_ambiguous,
                "field_name": field_name,
                "edit_category": edit_cat,
                "layout_family": layout,
                "crop_rel_path": f"crops/{crop_filename}",
                "norm_box": json.dumps(norm_box),
                "pixel_box": json.dumps(raw_pixel_box),
                "max_iou": max_iou,
                "max_gt_coverage": max_cov,
                "is_routing_hit": is_hit,
            })

    # Save manifest
    manifest_path = DATA_DIR / "crop_dataset_manifest.csv"
    with open(manifest_path, "w", newline="") as fp:
        fields = list(manifest_rows[0].keys())
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    logger.info("Saved crop dataset manifest to %s (%d rows, SHA-256: %s)", manifest_path, len(manifest_rows), manifest_hash)

    # Compile comprehensive diagnostics
    diag = {
        "manifest_path": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": manifest_hash,
        "total_manifest_crops": len(manifest_rows),
        "total_train_crops": sum(1 for m in manifest_rows if m["split"] == "train"),
        "total_val_crops": sum(1 for m in manifest_rows if m["split"] == "val"),
        "train_crop_counts_by_field": crop_counts_by_field,
        "train_crop_counts_by_label": crop_counts_by_label,
        "train_crop_counts_by_layout": crop_counts_by_layout,
        "ambiguous_crops_excluded": ambiguous_crop_count,
        "edited_screenshots_evaluated": total_edited_screenshots,
        "edited_screenshots_routed_hits": routing_hits,
        "edited_screenshots_routed_misses": total_edited_screenshots - routing_hits,
        "routing_recall_overall": round(routing_hits / max(1, total_edited_screenshots), 4),
        "routing_hits_by_category": routing_hits_by_cat,
        "routing_misses_by_category": routing_misses_by_cat,
    }

    diag_path = DATA_DIR / "crop_dataset_audit.json"
    with open(diag_path, "w") as fp:
        json.dump(diag, fp, indent=2)

    logger.info("Crop Dataset Audit:\n%s", json.dumps(diag, indent=2))
    return diag


def main():
    logger.info("Merging OCR shards...")
    train_records = merge_shards()
    pair_to_fold = load_fold_assignments()
    build_crop_dataset(train_records, pair_to_fold)
    logger.info("Crop dataset successfully constructed and audited!")


if __name__ == "__main__":
    main()
