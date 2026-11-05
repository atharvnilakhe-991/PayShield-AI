"""
Phase 17 Dataset Preparation Script.
Extracts native-resolution crops for all training (434 pairs) and validation (54 pairs) images.
Saves crops and metadata manifest to experiments/phase17_visual_forensics/data/.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase17.prep_data")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "Hackathon-VIT" / "CBL04-Dataset"
FA_DIR = DATASET_ROOT / "annotations" / "phase4" / "field_annotations"
CLEAN_DIR = DATASET_ROOT / "data" / "synthetic_gpay_phase4" / "clean"
EDITED_DIR = DATASET_ROOT / "data" / "synthetic_gpay_phase4" / "edited"
OUT_BASE = PROJECT_ROOT / "experiments" / "phase17_visual_forensics" / "data"

TARGET_CANVAS = (256, 128)  # width=256, height=128
BACKGROUND_RGB = (18, 18, 18)


def extract_padded_crop(img: Image.Image, bbox: list[int]) -> Image.Image:
    w_orig, h_orig = img.size
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w_orig, x1), min(h_orig, y1)
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)

    # 15% context margin
    pad_x = int(round(bw * 0.15))
    pad_y = int(round(bh * 0.20))
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


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    train_crop_dir = OUT_BASE / "train_crops"
    val_crop_dir = OUT_BASE / "val_target_crops"
    train_crop_dir.mkdir(parents=True, exist_ok=True)
    val_crop_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for split_name, manifest_name, crop_dir in [
        ("train", "train_manifest.csv", train_crop_dir),
        ("val", "validation_manifest.csv", val_crop_dir),
    ]:
        mpath = DATASET_ROOT / "metadata" / "phase6" / manifest_name
        with open(mpath) as fp:
            rows = list(csv.DictReader(fp))

        pair_ids = sorted(list({r["pair_id"] for r in rows}))
        logger.info("[%s] Processing %d pairs (%d images)...", split_name, len(pair_ids), len(rows))

        for p_id in pair_ids:
            json_file = FA_DIR / f"{p_id}.json"
            if not json_file.exists():
                logger.warning("Missing annotation: %s", json_file)
                continue

            with open(json_file) as fp:
                meta = json.load(fp)

            bbox = meta["bounding_boxes_xyxy"][0]
            cat = meta.get("edited_field", "unknown")
            layout = meta.get("layout_family", "unknown")

            for lbl in ["clean", "edited"]:
                sample_id = f"{p_id}_{lbl}"
                img_dir = CLEAN_DIR if lbl == "clean" else EDITED_DIR
                img_path = img_dir / f"{sample_id}.png"
                if not img_path.exists():
                    logger.warning("Image missing: %s", img_path)
                    continue

                with Image.open(img_path) as src:
                    img = src.convert("RGB")

                crop_img = extract_padded_crop(img, bbox)
                crop_path = crop_dir / f"{sample_id}_target_{cat}.png"
                crop_img.save(crop_path)

                manifest_rows.append({
                    "sample_id": sample_id,
                    "pair_id": p_id,
                    "split": split_name,
                    "label": lbl,
                    "true_label": 0 if lbl == "clean" else 1,
                    "edit_category": cat,
                    "layout_family": layout,
                    "crop_field": cat,
                    "crop_path": str(crop_path.relative_to(PROJECT_ROOT)),
                    "native_bbox": str(bbox),
                })

    out_csv = OUT_BASE / "crop_dataset_manifest.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    logger.info("Saved %d crop entries to %s", len(manifest_rows), out_csv)


if __name__ == "__main__":
    main()

