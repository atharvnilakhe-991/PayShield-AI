#!/usr/bin/env python3
"""
Phase 20: Dense Multi-Scale Patch Extraction
Extracts uniform spatial grid patches across complete screenshots for CBL-04:
- Scale 1: 4x2 uniform grid (8 patches per screenshot)
- Scale 2: 6x3 uniform grid (18 patches per screenshot)
Total patches per screenshot: 26 patches.

Ensures complete spatial coverage with documented mathematical coordinates,
aspect-ratio preservation via square letterboxing (RGB 18, 18, 18),
LANCZOS resizing to 224x224, and full SHA-256 provenance tracking.
"""

import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from PIL import Image

WORKSPACE_ROOT = Path("/Users/heemanshupjain/Downloads/Member 3")
DATASET_ROOT = WORKSPACE_ROOT / "Hackathon-VIT" / "CBL04-Dataset"
OUTPUT_DIR = WORKSPACE_ROOT / "experiments" / "phase20_dense_mil_forensics"
PATCHES_DIR = OUTPUT_DIR / "patches"
MANIFEST_PATH = OUTPUT_DIR / "data" / "patch_manifest.csv"
FOLD_ASSIGNMENTS_PATH = WORKSPACE_ROOT / "experiments" / "phase19_local_branch_redevelopment" / "data" / "fold_assignments.csv"

TARGET_PATCH_SIZE = 224
BACKGROUND_COLOR = (18, 18, 18)


def compute_sha256(path_or_bytes):
    h = hashlib.sha256()
    if isinstance(path_or_bytes, (str, Path)):
        with open(path_or_bytes, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
    else:
        h.update(path_or_bytes)
    return h.hexdigest()


def letterbox_and_resize(patch_img, target_size=TARGET_PATCH_SIZE, bg_color=BACKGROUND_COLOR):
    """
    Letterboxes patch to square using GPay dark background (18, 18, 18),
    then resizes to target_size using high-quality LANCZOS interpolation.
    """
    w, h = patch_img.size
    max_dim = max(w, h)
    canvas = Image.new("RGB", (max_dim, max_dim), bg_color)
    offset_x = (max_dim - w) // 2
    offset_y = (max_dim - h) // 2
    canvas.paste(patch_img, (offset_x, offset_y))
    resized = canvas.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return resized


def get_grid_coordinates(w, h, n_rows, n_cols):
    """
    Calculates exact integer bounding boxes covering the entire image without gaps or border misses.
    """
    coords = []
    for r in range(n_rows):
        y1 = int(r * h / n_rows)
        y2 = int((r + 1) * h / n_rows) if r < n_rows - 1 else h
        for c in range(n_cols):
            x1 = int(c * w / n_cols)
            x2 = int((c + 1) * w / n_cols) if c < n_cols - 1 else w
            coords.append((r, c, [x1, y1, x2, y2]))
    return coords


def process_image(item):
    """
    Worker function to process a single screenshot:
    Extracts Scale 1 (4x2 = 8 patches) and Scale 2 (6x3 = 18 patches).
    """
    image_id = item["image_id"]
    pair_id = item["pair_id"]
    split = item["split"]
    fold = item["fold"]
    rel_path = item["image_path"]
    full_path = DATASET_ROOT / rel_path

    if not full_path.exists():
        raise FileNotFoundError(f"Image not found: {full_path}")

    source_hash = compute_sha256(full_path)

    with Image.open(full_path) as im:
        img_rgb = im.convert("RGB")
        w, h = img_rgb.size

        patch_records = []

        # Scale 1: 4 rows x 2 cols = 8 patches
        s1_coords = get_grid_coordinates(w, h, n_rows=4, n_cols=2)
        for r, c, bbox in s1_coords:
            patch_id = f"{image_id}_s1_r{r}_c{c}"
            patch_crop = img_rgb.crop(bbox)
            proc_patch = letterbox_and_resize(patch_crop)

            patch_filename = f"{patch_id}.png"
            patch_path = PATCHES_DIR / patch_filename
            proc_patch.save(patch_path, format="PNG", optimize=False)
            patch_hash = compute_sha256(patch_path)

            patch_records.append({
                "image_id": image_id,
                "pair_id": pair_id,
                "split": split,
                "fold": fold,
                "patch_id": patch_id,
                "grid_position": f"({r},{c})",
                "original_coordinates": json.dumps(bbox),
                "scale": "scale_1",
                "source_image_hash": source_hash,
                "patch_hash": patch_hash,
            })

        # Scale 2: 6 rows x 3 cols = 18 patches
        s2_coords = get_grid_coordinates(w, h, n_rows=6, n_cols=3)
        for r, c, bbox in s2_coords:
            patch_id = f"{image_id}_s2_r{r}_c{c}"
            patch_crop = img_rgb.crop(bbox)
            proc_patch = letterbox_and_resize(patch_crop)

            patch_filename = f"{patch_id}.png"
            patch_path = PATCHES_DIR / patch_filename
            proc_patch.save(patch_path, format="PNG", optimize=False)
            patch_hash = compute_sha256(patch_path)

            patch_records.append({
                "image_id": image_id,
                "pair_id": pair_id,
                "split": split,
                "fold": fold,
                "patch_id": patch_id,
                "grid_position": f"({r},{c})",
                "original_coordinates": json.dumps(bbox),
                "scale": "scale_2",
                "source_image_hash": source_hash,
                "patch_hash": patch_hash,
            })

    return patch_records


def main():
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.joinpath("data").mkdir(parents=True, exist_ok=True)

    # Load fold assignments from Phase 19
    pair_to_fold = {}
    with open(FOLD_ASSIGNMENTS_PATH, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pair_to_fold[r["pair_id"]] = int(r["fold"])

    # Load training manifest (434 pairs / 868 images)
    train_manifest = DATASET_ROOT / "metadata" / "phase6" / "train_manifest.csv"
    items = []
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
            })

    # Load validation manifest (54 pairs / 108 images)
    val_manifest = DATASET_ROOT / "metadata" / "phase6" / "validation_manifest.csv"
    with open(val_manifest, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            items.append({
                "image_id": r["image_id"],
                "pair_id": r["pair_id"],
                "split": "val",
                "fold": -1,  # Strictly held out from all folds
                "image_path": r["image_path"],
                "label": r.get("label", "clean" if "_clean" in r["image_id"] else "edited"),
                "edit_category": r.get("edit_category", ""),
                "layout_family": r.get("layout_family", "bank_progress"),
            })

    print(f"Total screenshots to extract: {len(items)} (Train: 868, Val: 108)")
    print(f"Expected patches: {len(items) * 26} ({len(items) * 8} Scale-1, {len(items) * 18} Scale-2)")

    num_workers = min(8, mp.cpu_count())
    print(f"Launching {num_workers} parallel extraction workers...")

    all_patch_records = []
    with mp.Pool(num_workers) as pool:
        results = pool.map(process_image, items)
        for res in results:
            all_patch_records.extend(res)

    print(f"Extracted {len(all_patch_records)} patches successfully.")

    # Write patch manifest
    fieldnames = [
        "image_id", "pair_id", "split", "fold", "patch_id",
        "grid_position", "original_coordinates", "scale",
        "source_image_hash", "patch_hash"
    ]
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_patch_records)

    print(f"Patch manifest written to {MANIFEST_PATH}")
    print(f"Manifest SHA-256: {compute_sha256(MANIFEST_PATH)}")


if __name__ == "__main__":
    main()
