"""
Phase 19: Comprehensive Crop Dataset Audit Script.
Verifies all constraints mandated in Part 2 of User Instructions.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "experiments" / "phase19_local_branch_redevelopment" / "data"


def main():
    manifest_p = DATA_DIR / "crop_dataset_manifest.csv"
    with open(manifest_p) as f:
        rows = list(csv.DictReader(f))

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]

    print("=== PART 2: CROP DATASET AUDIT ===")
    print("Total crops in manifest:", len(rows))
    print("Total train crops:", len(train_rows))
    print("Total val crops:", len(val_rows))

    train_samples = set(r["sample_id"] for r in train_rows)
    train_pairs = set(r["pair_id"] for r in train_rows)
    print("Unique source images in train crops:", len(train_samples))
    print("Unique matched pairs in train crops:", len(train_pairs))

    print("\n--- Counts by candidate type (train) ---")
    for k, v in sorted(Counter(r["field_name"] for r in train_rows).items()):
        print(f"  {k}: {v}")

    print("\n--- Counts by label (train) ---")
    for k, v in sorted(Counter(r["crop_label"] for r in train_rows).items()):
        lbl_str = "clean" if k == "0" else ("tampered" if k == "1" else "ambiguous (excluded)")
        print(f"  label {k} ({lbl_str}): {v}")

    print("\n--- Counts by edit category (train) ---")
    for k, v in sorted(Counter(r["edit_category"] for r in train_rows).items()):
        print(f"  {k}: {v}")

    print("\n--- Counts by fold (train) ---")
    for k, v in sorted(Counter(r["fold"] for r in train_rows).items()):
        print(f"  Fold {k}: {v} crops")

    ambiguous = [r for r in train_rows if r.get("is_ambiguous") == "True" or r["crop_label"] == "-1"]
    print("\n--- Ambiguous / Excluded Crops ---")
    print(f"Total ambiguous crops: {len(ambiguous)}")
    for a in ambiguous:
        print(f"  Crop ID: {a['crop_id']}, Field: {a['field_name']}, IoU: {a['max_iou']}, Cov: {a['max_gt_coverage']}")

    # Verify no pair crosses folds
    pair_folds = {}
    for r in train_rows:
        pair_folds.setdefault(r["pair_id"], set()).add(r["fold"])
    crossing_pairs = {p: f for p, f in pair_folds.items() if len(f) > 1}
    print("\nPairs crossing folds (must be 0):", len(crossing_pairs))

    # Verify clean screenshots have zero tampered crops
    clean_tampered = [r for r in train_rows if r["image_label"] == "clean" and int(r["crop_label"]) == 1]
    print("Clean screenshots with tampered crops (must be 0):", len(clean_tampered))

    # Verify edited screenshots: crops that do not overlap GT are labeled 0
    edited_nonoverlap_labeled_tampered = [
        r for r in train_rows 
        if r["image_label"] == "edited" and float(r["max_iou"]) < 0.05 and float(r["max_gt_coverage"]) < 0.05 and int(r["crop_label"]) == 1
    ]
    print("Non-overlapping crops on edited screenshots labeled tampered (must be 0):", len(edited_nonoverlap_labeled_tampered))

    # Inspect 5 reproducible samples across different categories
    print("\n=== REPRODUCIBLE SAMPLE INSPECTION (5 crops) ===")
    sample_indices = [0, 500, 1000, 2000, 3000]
    for idx in sample_indices:
        r = train_rows[idx]
        print(f"Sample #{idx}:")
        print(f"  Crop ID: {r['crop_id']}")
        print(f"  Pair ID: {r['pair_id']}, Fold: {r['fold']}")
        print(f"  Image Label: {r['image_label']}, Edit Cat: {r['edit_category']}, Layout: {r['layout_family']}")
        print(f"  Field: {r['field_name']}, Crop Label: {r['crop_label']}")
        print(f"  OCR Pixel Box: {r['pixel_box']}")
        print(f"  Max IoU: {r['max_iou']}, Max GT Coverage: {r['max_gt_coverage']}")
        print(f"  Routing Hit: {r['is_routing_hit']}")


if __name__ == "__main__":
    main()

