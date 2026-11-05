from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261206
PHASH_THRESHOLD = 6
SPLIT_BY_FAMILY = {
    "compact": "train",
    "enriched": "train",
    "bank_progress": "validation",
    "recharge": "test",
}
FAMILY_LABELS = {
    "Compact transaction details": "compact",
    "Merchant/context-enriched details": "enriched",
    "Bank payment-progress timeline": "bank_progress",
    "Recharge/utility detail summary": "recharge",
}
sys.path.insert(0, str(ROOT / "scripts/phase5"))
from audit_dataset import p_hash  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty split manifest: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def components(clean_rows: list[dict[str, str]]) -> dict[str, str]:
    by_family: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for row in clean_rows:
        by_family[row["layout_family"]].append((row["source_id"], p_hash(ROOT / row["image_path"])))
    result: dict[str, str] = {}
    for family, items in by_family.items():
        parent = list(range(len(items)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if (items[i][1] ^ items[j][1]).bit_count() <= PHASH_THRESHOLD:
                    a, b = find(i), find(j)
                    if a != b:
                        parent[b] = a
        groups: dict[int, list[str]] = defaultdict(list)
        for i, (sid, _) in enumerate(items):
            groups[find(i)].append(sid)
        ordered = sorted((sorted(ids) for ids in groups.values()), key=lambda ids: ids[0])
        for number, ids in enumerate(ordered, 1):
            gid = f"ND-{family}-{number:03d}"
            for sid in ids:
                result[sid] = gid
    return result


def main() -> None:
    out_dir = ROOT / "metadata/phase6"
    if (ROOT / "metadata/phase6_manifest.csv").exists():
        raise FileExistsError("Phase 6 manifest exists; refusing to overwrite split assignments")
    bulk = read_csv(ROOT / "metadata/phase4_manifest.csv")
    clean_rows = [r for r in bulk if r["label"] == "0"]
    pgroup = components(clean_rows)
    real_inventory = read_csv(ROOT / "reports/layout_family_inventory.csv")
    real_family = {r["filename"]: FAMILY_LABELS.get(r["family"], "uncertain") for r in real_inventory}

    manifest: list[dict[str, Any]] = []
    clean_info = {r["source_id"]: {"phash_group": pgroup[r["source_id"]],
                                    "template_group": f"TPL-{r['layout_family']}"} for r in clean_rows}
    for row in bulk:
        sid = row["source_id"]
        family = row["layout_family"]
        split = SPLIT_BY_FAMILY[family]
        image_path = ROOT / row["image_path"]
        ann_path = ROOT / "annotations/phase4/field_annotations" / f"{sid}.json"
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        with Image.open(image_path) as im:
            im.load(); width, height, image_format = im.width, im.height, im.format
        manifest.append({
            "image_id": row["image_id"], "pair_id": sid, "source_id": sid,
            "source_dataset": "phase4_synthetic", "sample_type": "synthetic",
            "split": split, "label": "clean" if row["label"] == "0" else "edited",
            "label_known": True, "layout_family": family,
            "edit_category": row.get("edited_field", ""), "is_synthetic": True,
            "image_path": row["image_path"], "mask_path": row["mask_path"],
            "annotation_path": ann_path.relative_to(ROOT).as_posix(),
            "bounding_boxes_xyxy": row["bounding_box"], "width": width, "height": height,
            "image_format": image_format, "template_group_id": clean_info[sid]["template_group"],
            "near_duplicate_group_id": clean_info[sid]["phash_group"],
            "perceptual_hash_64": f"{p_hash(image_path):016x}", "image_sha256": sha256(image_path),
            "random_seed": row["random_seed"], "template_version": row["template_version"],
            "original_fictional_value": ann["original_fictional_value"] if row["label"] == "1" else "",
            "replacement_fictional_value": ann["replacement_fictional_value"] if row["label"] == "1" else "",
        })

    # Canonical one-per-content real inventory; exact duplicate copies remain in raw storage.
    raw_paths = sorted(p for p in (ROOT / "data/real_gpay/raw").rglob("*") if p.is_file())
    seen_real: set[str] = set()
    for path in raw_paths:
        digest = sha256(path)
        if digest in seen_real:
            continue
        seen_real.add(digest)
        relative = path.relative_to(ROOT).as_posix()
        with Image.open(path) as im:
            im.load(); width, height, image_format = im.width, im.height, im.format
        manifest.append({
            "image_id": f"REAL-{path.stem}", "pair_id": "", "source_id": f"REAL-{path.stem}",
            "source_dataset": "real_gpay_archive", "sample_type": "real_sensitive",
            "split": "restricted_real_holdout", "label": "unlabeled", "label_known": False,
            "layout_family": real_family.get(path.name, "uncertain"), "edit_category": "",
            "is_synthetic": False, "image_path": relative, "mask_path": "", "annotation_path": "",
            "bounding_boxes_xyxy": "", "width": width, "height": height, "image_format": image_format,
            "template_group_id": "", "near_duplicate_group_id": "", "perceptual_hash_64": "",
            "image_sha256": digest, "random_seed": "", "template_version": "",
            "original_fictional_value": "", "replacement_fictional_value": "",
        })
    if len(seen_real) != 387:
        raise RuntimeError(f"Expected 387 unique real screenshots; found {len(seen_real)}")

    # Seed controls row order only; group-to-split assignments follow the fixed leakage-safe family map.
    rng = random.Random(SEED)
    for split in ("train", "validation", "test", "restricted_real_holdout"):
        subset = [r for r in manifest if r["split"] == split]
        rng.shuffle(subset)
        write_csv(out_dir / f"{split}_manifest.csv", subset)
    rng.shuffle(manifest)
    write_csv(ROOT / "metadata/phase6_manifest.csv", manifest)

    group_rows = []
    for family, split in SPLIT_BY_FAMILY.items():
        sources = sorted(r["source_id"] for r in clean_rows if r["layout_family"] == family)
        phash_ids = sorted({pgroup[s] for s in sources})
        group_rows.append({"layout_family": family, "template_group_id": f"TPL-{family}",
                           "near_duplicate_components_at_hamming_le_6": len(phash_ids),
                           "source_pair_count": len(sources), "assigned_split": split,
                           "component_ids_merged_by_family_group": json.dumps(phash_ids)})
    write_csv(out_dir / "group_assignments.csv", group_rows)
    config = {"seed": SEED, "target_ratios": {"train": .70, "validation": .15, "test": .15},
              "actual_split_policy": "whole layout-family template groups; pHash connected components at distance <= 6 are never separated",
              "perceptual_hash_method": "Phase 5 64-bit grayscale pHash; connected components within family at Hamming <= 6",
              "family_split_assignment": SPLIT_BY_FAMILY, "real_split": "restricted_real_holdout",
              "real_labels": "unlabeled; excluded from supervised splits pending verified ground truth and privacy review",
              "real_duplicate_selection": "first lexicographic path for each unique SHA-256; duplicate copies excluded from phase6 manifest",
              "previous_pilots": "kept separate; excluded from Phase 6 model splits"}
    (out_dir / "phase6_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} Phase 6 manifest rows ({len(seen_real)} unique real, {len(bulk)} synthetic)")


if __name__ == "__main__":
    main()
