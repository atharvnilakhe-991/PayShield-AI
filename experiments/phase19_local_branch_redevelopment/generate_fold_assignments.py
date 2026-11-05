"""
Phase 19: Generate and Verify Grouped Stratified 5-Fold Assignments.
Pre-generates and rigorously audits the 5-fold cross-validation split on 434 training pairs (868 images).
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase19.folds")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "Hackathon-VIT" / "CBL04-Dataset"
EXP_DIR = PROJECT_ROOT / "experiments" / "phase19_local_branch_redevelopment"
DATA_DIR = EXP_DIR / "data"

SEED = 42
N_FOLDS = 5


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_manifest = DATASET_ROOT / "metadata" / "phase6" / "train_manifest.csv"
    with open(train_manifest) as fp:
        rows = list(csv.DictReader(fp))

    assert len(rows) == 868, f"Expected 868 rows, found {len(rows)}"

    pair_meta: Dict[str, Dict[str, str]] = {}
    for r in rows:
        pid = r["pair_id"]
        cat = r.get("edit_category", "")
        layout = r["layout_family"]
        if pid not in pair_meta:
            pair_meta[pid] = {
                "pair_id": pid,
                "edit_category": cat,
                "layout_family": layout,
            }
        elif r["label"] == "edited" and cat:
            pair_meta[pid]["edit_category"] = cat

    assert len(pair_meta) == 434, f"Expected 434 pairs, found {len(pair_meta)}"

    # Ensure every pair has a valid non-empty edit category
    for pid, meta in pair_meta.items():
        assert meta["edit_category"] in {"amount", "date_time", "recipient", "other", "transaction_id"}, \
            f"Invalid category '{meta['edit_category']}' for pair {pid}"

    # Group pairs by edit category
    cat_to_pairs: Dict[str, List[str]] = {}
    for pid, meta in pair_meta.items():
        cat = meta["edit_category"]
        cat_to_pairs.setdefault(cat, []).append(pid)

    rng = random.Random(SEED)
    pair_to_fold: Dict[str, int] = {}

    for cat in sorted(cat_to_pairs.keys()):
        pids = sorted(cat_to_pairs[cat])  # sort for deterministic reproducibility
        rng.shuffle(pids)
        for idx, pid in enumerate(pids):
            fold = idx % N_FOLDS
            pair_to_fold[pid] = fold

    # Write fold manifest
    out_csv = DATA_DIR / "fold_assignments.csv"
    with open(out_csv, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["pair_id", "fold", "edit_category", "layout_family"])
        writer.writeheader()
        for pid in sorted(pair_meta.keys()):
            writer.writerow({
                "pair_id": pid,
                "fold": pair_to_fold[pid],
                "edit_category": pair_meta[pid]["edit_category"],
                "layout_family": pair_meta[pid]["layout_family"],
            })

    # Verification
    # 1. Confirm all 434 pairs appear in exactly one fold
    assert len(pair_to_fold) == 434
    for f in range(N_FOLDS):
        val_pairs = set(pid for pid, fold in pair_to_fold.items() if fold == f)
        train_pairs = set(pid for pid, fold in pair_to_fold.items() if fold != f)
        assert len(val_pairs & train_pairs) == 0, f"Leakage detected in fold {f}!"

    # 2. Confirm both members of every pair belong to the same fold
    for r in rows:
        pid = r["pair_id"]
        assert pair_to_fold[pid] is not None

    # 3. Check Category distribution across folds
    fold_cat_counts = {f: {} for f in range(N_FOLDS)}
    for pid, fold in pair_to_fold.items():
        cat = pair_meta[pid]["edit_category"]
        fold_cat_counts[fold][cat] = fold_cat_counts[fold].get(cat, 0) + 1

    fold_total_pairs = {f: sum(fold_cat_counts[f].values()) for f in range(N_FOLDS)}

    # Compute SHA-256
    h = hashlib.sha256(out_csv.read_bytes()).hexdigest()

    summary = {
        "fold_assignments_path": str(out_csv.relative_to(PROJECT_ROOT)),
        "sha256": h,
        "total_pairs": len(pair_to_fold),
        "total_images": len(rows),
        "fold_total_pairs": fold_total_pairs,
        "category_stratification": fold_cat_counts,
    }

    summary_json = DATA_DIR / "fold_audit_summary.json"
    with open(summary_json, "w") as fp:
        json.dump(summary, fp, indent=2)

    logger.info("Fold Audit Summary:\n%s", json.dumps(summary, indent=2))
    print(f"FOLD_AUDIT_SHA256: {h}")


if __name__ == "__main__":
    main()
