from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261207
FOLDS = [
    {"fold": 1, "test": "compact", "validation": "bank_progress"},
    {"fold": 2, "test": "enriched", "validation": "recharge"},
    {"fold": 3, "test": "bank_progress", "validation": "compact"},
    {"fold": 4, "test": "recharge", "validation": "enriched"},
]


def load_rows() -> list[dict[str, str]]:
    with (ROOT / "metadata/phase6_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Empty fold manifest: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    source = load_rows()
    synthetic = [r for r in source if r["sample_type"] == "synthetic"]
    root = ROOT / "metadata/phase7/folds"
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Fold manifests already exist: {root}; refusing to overwrite")
    output_rows: list[dict[str, Any]] = []
    summary = []
    rng = random.Random(args.seed)
    for spec in FOLDS:
        test_family, val_family = spec["test"], spec["validation"]
        train_families = sorted({r["layout_family"] for r in synthetic} - {test_family, val_family})
        if len(train_families) != 2:
            raise RuntimeError(f"Fold {spec['fold']} does not leave two training families")
        assignment = {}
        for row in synthetic:
            family = row["layout_family"]
            assignment[row["pair_id"]] = "test" if family == test_family else "validation" if family == val_family else "train"
        fold_rows = []
        for row in synthetic:
            x = dict(row)
            x["fold"] = spec["fold"]
            x["fold_role"] = assignment[row["pair_id"]]
            fold_rows.append(x)
            output_rows.append(x)
        counts = {}
        for role in ("train", "validation", "test"):
            members = [r for r in fold_rows if r["fold_role"] == role and r["label"] == "clean"]
            counts[role] = {"families": sorted({r["layout_family"] for r in members}),
                            "pairs": len(members), "clean": len(members), "edited": len(members)}
        for role in ("train", "validation", "test"):
            rows = [r for r in fold_rows if r["fold_role"] == role]
            rng.shuffle(rows)
            write(root / f"fold_{spec['fold']}_{role}.csv", rows)
        summary.append({**spec, "train_families": train_families, "counts": counts})
    rng.shuffle(output_rows)
    write(ROOT / "metadata/phase7/fold_manifest.csv", output_rows)
    cfg = {"seed": args.seed, "strategy": "leave-one-layout-family-out with distinct whole-family validation group",
           "phash_threshold_hamming": 6, "family_rotation": FOLDS,
           "fold_pair_source_id_unique": "pair_id remains intact across all samples in any fold; every pair is test exactly once",
           "phase6_manifests_modified": False}
    (ROOT / "metadata/phase7/fold_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
