#!/usr/bin/env python3
"""
Phase 21: Sharded CRAFT Text-Region Detection
Extracts OCR-free text candidate regions across complete screenshots:
- Detects word-level bounding boxes using pure CRAFT detector
- Preserves raw coordinates [x1, y1, x2, y2]
- Computes conservative glyph padding margin
- Calculates continuous CRAFT detector confidence from character heatmap
- Completely independent of OCR character recognition
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

import torch
torch.set_num_threads(2)

WORKSPACE_ROOT = Path("/Users/heemanshupjain/Downloads/Member 3")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from app.services.ocr_service import get_ocr_service
import easyocr.detection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase21.craft_shard")

DATASET_ROOT = WORKSPACE_ROOT / "Hackathon-VIT" / "CBL04-Dataset"
OUTPUT_DIR = WORKSPACE_ROOT / "experiments" / "phase21_craft_word_forensics" / "data"
FOLD_ASSIGNMENTS_PATH = WORKSPACE_ROOT / "experiments" / "phase19_local_branch_redevelopment" / "data" / "fold_assignments.csv"


def load_dataset_manifest():
    """Loads all 868 train and 108 val screenshots in a deterministic order."""
    pair_to_fold = {}
    with open(FOLD_ASSIGNMENTS_PATH, "r") as f:
        for r in csv.DictReader(f):
            pair_to_fold[r["pair_id"]] = int(r["fold"])

    items = []
    # Train
    train_manifest = DATASET_ROOT / "metadata" / "phase6" / "train_manifest.csv"
    with open(train_manifest, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            items.append({
                "image_id": r["image_id"],
                "pair_id": r["pair_id"],
                "split": "train",
                "fold": pair_to_fold[r["pair_id"]],
                "image_path": r["image_path"],
                "label": r.get("label", "clean" if "_clean" in r["image_id"] else "edited"),
                "edit_category": r.get("edit_category", ""),
                "layout_family": r.get("layout_family", ""),
                "bounding_boxes_xyxy": r.get("bounding_boxes_xyxy", "[]"),
            })

    # Val
    val_manifest = DATASET_ROOT / "metadata" / "phase6" / "validation_manifest.csv"
    with open(val_manifest, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            items.append({
                "image_id": r["image_id"],
                "pair_id": r["pair_id"],
                "split": "val",
                "fold": -1,
                "image_path": r["image_path"],
                "label": r.get("label", "clean" if "_clean" in r["image_id"] else "edited"),
                "edit_category": r.get("edit_category", ""),
                "layout_family": r.get("layout_family", "bank_progress"),
                "bounding_boxes_xyxy": r.get("bounding_boxes_xyxy", "[]"),
            })

    # Sort deterministically
    items.sort(key=lambda x: (x["split"], x["pair_id"], x["image_id"]))
    return items


def detect_craft_boxes(img_rgb_np, detector, device, text_threshold=0.7, link_threshold=0.4, low_text=0.4):
    """
    Runs CRAFT text detector directly on RGB numpy image.
    Returns list of candidate box dicts with coordinates and confidence.
    """
    img_h, img_w, _ = img_rgb_np.shape

    # Aspect ratio resizing for CRAFT
    img_resized, target_ratio, size_heatmap = easyocr.detection.resize_aspect_ratio(
        img_rgb_np, 2560, interpolation=cv2.INTER_LINEAR, mag_ratio=1.0
    )
    ratio_h = ratio_w = 1.0 / target_ratio

    # Preprocessing
    x = torch.from_numpy(np.array([np.transpose(easyocr.detection.normalizeMeanVariance(img_resized), (2, 0, 1))])).to(device)

    with torch.no_grad():
        y, feature = detector(x)

    score_text = y[0, :, :, 0].cpu().numpy()
    score_link = y[0, :, :, 1].cpu().numpy()

    # Connected component analysis for word boxes
    boxes, polys, mapper = easyocr.detection.getDetBoxes(
        score_text, score_link,
        text_threshold=text_threshold,
        link_threshold=link_threshold,
        low_text=low_text,
        poly=False
    )
    boxes = easyocr.detection.adjustResultCoordinates(boxes, ratio_w, ratio_h)

    candidates = []
    for idx, b in enumerate(boxes):
        xs = [b[0][0], b[1][0], b[2][0], b[3][0]]
        ys = [b[0][1], b[1][1], b[2][1], b[3][1]]
        x1 = max(0, int(min(xs)))
        y1 = max(0, int(min(ys)))
        x2 = min(img_w, int(max(xs)))
        y2 = min(img_h, int(max(ys)))

        w = x2 - x1
        h = y2 - y1
        if w <= 2 or h <= 2:
            continue

        # Compute continuous detector confidence from score_text heatmap
        hx1, hy1 = int(x1 * target_ratio / 2), int(y1 * target_ratio / 2)
        hx2, hy2 = int(x2 * target_ratio / 2), int(y2 * target_ratio / 2)
        sh_h, sh_w = score_text.shape
        clip_hy1, clip_hy2 = max(0, min(sh_h, hy1)), max(0, min(sh_h, hy2))
        clip_hx1, clip_hx2 = max(0, min(sh_w, hx1)), max(0, min(sh_w, hx2))
        if clip_hy2 > clip_hy1 and clip_hx2 > clip_hx1:
            conf = float(np.mean(score_text[clip_hy1:clip_hy2, clip_hx1:clip_hx2]))
        else:
            conf = 0.0

        # Conservative glyph boundary expansion (15% padding, min 4px)
        pad_m = max(4, int(0.15 * min(w, h)))
        exp_x1 = max(0, x1 - pad_m)
        exp_y1 = max(0, y1 - pad_m)
        exp_x2 = min(img_w, x2 + pad_m)
        exp_y2 = min(img_h, y2 + pad_m)

        candidates.append({
            "candidate_idx": idx,
            "original_bbox": [x1, y1, x2, y2],
            "expanded_bbox": [exp_x1, exp_y1, exp_x2, exp_y2],
            "width": w,
            "height": h,
            "aspect_ratio": round(w / h, 4) if h > 0 else 1.0,
            "detector_confidence": round(conf, 4),
            "padding_margin": pad_m,
        })

    return candidates


def run_shard(shard_id, total_shards):
    all_items = load_dataset_manifest()
    n_total = len(all_items)
    chunk_size = (n_total + total_shards - 1) // total_shards
    start_idx = shard_id * chunk_size
    end_idx = min(start_idx + chunk_size, n_total)
    shard_items = all_items[start_idx:end_idx]

    logger.info("Shard %d/%d: processing rows %d to %d (%d screenshots)...",
                shard_id, total_shards, start_idx, end_idx, len(shard_items))

    ocr_service = get_ocr_service()
    ocr_service.load()
    detector = ocr_service.reader.detector
    device = ocr_service.reader.device

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"craft_shard_{shard_id}.json"

    shard_results = []
    t0 = time.time()

    for idx, item in enumerate(shard_items):
        t_img_start = time.time()
        img_full_path = DATASET_ROOT / item["image_path"]
        with Image.open(img_full_path) as im:
            img_rgb_np = np.array(im.convert("RGB"))
            w, h = im.size

        candidates = detect_craft_boxes(img_rgb_np, detector, device)
        elapsed = time.time() - t_img_start

        shard_results.append({
            "image_id": item["image_id"],
            "pair_id": item["pair_id"],
            "split": item["split"],
            "fold": item["fold"],
            "label": item["label"],
            "edit_category": item["edit_category"],
            "layout_family": item["layout_family"],
            "image_path": item["image_path"],
            "image_dimensions": [w, h],
            "bounding_boxes_xyxy": json.loads(item["bounding_boxes_xyxy"]),
            "candidate_count": len(candidates),
            "candidates": candidates,
            "detection_time_ms": round(elapsed * 1000, 2),
        })

        if (idx + 1) % 25 == 0 or (idx + 1) == len(shard_items):
            rate = (idx + 1) / (time.time() - t0)
            logger.info("Shard %d: %d/%d images done (%.2f img/s)", shard_id, idx + 1, len(shard_items), rate)

    with open(out_file, "w") as f:
        json.dump(shard_results, f, indent=2)

    logger.info("Shard %d complete: %d records saved to %s in %.2fs",
                shard_id, len(shard_results), out_file, time.time() - t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_id", type=int, required=True)
    parser.add_argument("--total_shards", type=int, default=4)
    args = parser.parse_args()
    run_shard(args.shard_id, args.total_shards)

