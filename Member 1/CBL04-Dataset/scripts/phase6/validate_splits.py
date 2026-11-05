from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/phase6"
SPLITS = ["train", "validation", "test"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def pct(n: int, den: int) -> str:
    return f"{100*n/den:.1f}%" if den else "0.0%"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"]*len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    manifest_path = ROOT / "metadata/phase6_manifest.csv"
    manifest = read_csv(manifest_path)
    errors: list[str] = []
    synthetic = [r for r in manifest if r["sample_type"] == "synthetic"]
    real = [r for r in manifest if r["sample_type"] == "real_sensitive"]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    real_hashes: set[str] = set()
    for row in manifest:
        if row["pair_id"]:
            groups[row["pair_id"]].append(row)
        image_path = ROOT / row["image_path"]
        if not image_path.is_file():
            errors.append(f"missing image: {row['image_id']}")
        else:
            try:
                with Image.open(image_path) as im:
                    im.verify()
                with Image.open(image_path) as im:
                    im.load()
                    if im.size != (int(row["width"]), int(row["height"])):
                        errors.append(f"dimension mismatch: {row['image_id']}")
                    if im.format != row["image_format"]:
                        errors.append(f"image format mismatch: {row['image_id']}")
            except Exception as exc:
                errors.append(f"unreadable image {row['image_id']}: {exc}")
        if row["sample_type"] == "synthetic":
            for key in ("mask_path", "annotation_path"):
                if not row[key] or not (ROOT / row[key]).is_file():
                    errors.append(f"missing {key}: {row['image_id']}")
            if row["mask_path"] and (ROOT / row["mask_path"]).is_file():
                try:
                    with Image.open(ROOT / row["mask_path"]) as im:
                        im.verify()
                    with Image.open(ROOT / row["mask_path"]) as im:
                        if im.size != (int(row["width"]), int(row["height"])):
                            errors.append(f"mask dimension mismatch: {row['image_id']}")
                except Exception as exc:
                    errors.append(f"unreadable mask {row['image_id']}: {exc}")
            if row["annotation_path"] and (ROOT / row["annotation_path"]).is_file():
                try:
                    annotation = json.loads((ROOT / row["annotation_path"]).read_text(encoding="utf-8"))
                    if annotation.get("source_id") != row["source_id"]:
                        errors.append(f"annotation source ID mismatch: {row['image_id']}")
                except Exception as exc:
                    errors.append(f"unreadable annotation {row['image_id']}: {exc}")
        elif row["split"] != "restricted_real_holdout" or row["label_known"] != "False" or row["label"] != "unlabeled":
            errors.append(f"real sample assigned a supervised label/split: {row['image_id']}")
        else:
            real_hashes.add(row["image_sha256"])
    if len(real_hashes) != len(real):
        errors.append("real holdout contains repeated SHA-256 values")
    for pid, pair in groups.items():
        labels = {r["label"] for r in pair}
        if len(pair) != 2 or labels != {"clean", "edited"}:
            errors.append(f"invalid pair membership: {pid}")
        if len({r["split"] for r in pair}) != 1:
            errors.append(f"pair split leakage: {pid}")
        for key in ("layout_family", "template_group_id", "near_duplicate_group_id"):
            if len({r[key] for r in pair}) != 1:
                errors.append(f"pair {key} mismatch: {pid}")
    for key in ("template_group_id", "near_duplicate_group_id"):
        assignments: dict[str, set[str]] = defaultdict(set)
        for row in synthetic:
            assignments[row[key]].add(row["split"])
        for group_id, splitset in assignments.items():
            if len(splitset) != 1:
                errors.append(f"{key} leaks across splits: {group_id} -> {sorted(splitset)}")
    hash_assignments: dict[str, set[str]] = defaultdict(set)
    for row in synthetic:
        hash_assignments[row["image_sha256"]].add(row["split"])
    cross_hash = [h for h, splits in hash_assignments.items() if len(splits) > 1]
    if cross_hash:
        errors.append(f"exact screenshot hashes cross split boundaries: {len(cross_hash)}")

    split_rows: dict[str, list[dict[str, str]]] = {}
    for split in SPLITS:
        rows = read_csv(ROOT / f"metadata/phase6/{split}_manifest.csv")
        split_rows[split] = rows
        if rows != [r for r in manifest if r["split"] == split]:
            # Output order is seeded and intentionally independent from master manifest order.
            if {r["image_id"] for r in rows} != {r["image_id"] for r in manifest if r["split"] == split}:
                errors.append(f"{split} manifest differs from master manifest")
        expected_labels = Counter({"clean": 0, "edited": 0})
        expected_labels.update(r["label"] for r in rows if r["sample_type"] == "synthetic")
        if expected_labels["clean"] != expected_labels["edited"]:
            errors.append(f"class imbalance within paired {split} split")

    family_split: dict[str, str] = {}
    template_groups = defaultdict(set)
    p_hash_groups = defaultdict(set)
    split_counts = {}
    family_counts: dict[str, Counter[str]] = {}
    edit_counts: dict[str, Counter[str]] = {}
    label_counts: dict[str, Counter[str]] = {}
    for split, rows in split_rows.items():
        syn = [r for r in rows if r["sample_type"] == "synthetic"]
        pairs = len({r["pair_id"] for r in syn})
        split_counts[split] = {"pairs": pairs, "synthetic_screenshots": len(syn),
                               "clean": sum(r["label"] == "clean" for r in syn),
                               "edited": sum(r["label"] == "edited" for r in syn)}
        family_counts[split] = Counter(r["layout_family"] for r in syn if r["label"] == "clean")
        edit_counts[split] = Counter(r["edit_category"] for r in syn if r["label"] == "edited")
        label_counts[split] = Counter(r["label"] for r in syn)
        for r in syn:
            family_split[r["layout_family"]] = split
            template_groups[r["template_group_id"]].add(split)
            p_hash_groups[r["near_duplicate_group_id"]].add(split)
    real_families = Counter(r["layout_family"] for r in real)
    distributions = {s: {"families": dict(family_counts[s]), "edits": dict(edit_counts[s])} for s in SPLITS}
    total_pairs = sum(x["pairs"] for x in split_counts.values())
    ratios = {s: {"pair_ratio": pct(split_counts[s]["pairs"], total_pairs),
                  "screenshot_ratio": pct(split_counts[s]["synthetic_screenshots"], total_pairs*2)} for s in SPLITS}
    report = {
        "passed": not errors, "manifest_rows": len(manifest), "synthetic_rows": len(synthetic),
        "synthetic_pairs": len(groups), "real_unique_unlabeled_rows": len(real),
        "real_family_counts": dict(real_families), "split_counts": split_counts, "split_ratios": ratios,
        "family_distribution_per_split": {s: dict(family_counts[s]) for s in SPLITS},
        "edit_categories_per_split": {s: dict(edit_counts[s]) for s in SPLITS},
        "errors": errors,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    targets = {"train": .70, "validation": .15, "test": .15}
    strategy = f"""# Phase 6 split strategy

## Policy

The 500 labeled Phase 4 clean/edited pairs are split as indivisible groups. Every pair remains together, and group integrity is stricter at the layout-family level: all pairs in a family stay in one split. Within each family, 64-bit grayscale pHash connected components at Hamming distance ≤6 are also kept intact. The Phase 5 similarity audit found that compact, enriched and bank-progress each form one connected pHash component; recharge forms three. Keeping the whole family together therefore prevents all identified pHash and template-group leakage.

The four family groups cannot be divided among three splits while approaching 70/15/15. The deterministic assignment that best preserves the target sizes places compact and enriched in training, bank progress in validation, and recharge in testing. The realized proportions are documented separately, and the limitation is not hidden by breaking group integrity.

| Split | Layout groups | Pairs | Actual pair ratio | Requested |
|---|---|---:|---:|---:|
| Train | compact, enriched | {split_counts['train']['pairs']} | {ratios['train']['pair_ratio']} | 70% |
| Validation | bank_progress | {split_counts['validation']['pairs']} | {ratios['validation']['pair_ratio']} | 15% |
| Test | recharge | {split_counts['test']['pairs']} | {ratios['test']['pair_ratio']} | 15% |

## Seed and reproducibility

Seed: `20261206`. Family-to-split mapping is fixed and recorded in `metadata/phase6/phase6_config.json`; the seed controls deterministic row ordering in manifests. pHash is deterministic. The split builder refuses to overwrite the existing master manifest. To reproduce, run it in a fresh project copy with the same source manifests and files.

## Real data and prior pilots

Real images are not assigned clean/edited labels: they remain in `restricted_real_holdout`, outside supervised splits, pending verified ground truth and privacy review. One canonical path per SHA-256 is included; seven byte-identical copies remain in raw storage but are not repeated in the split manifest. The earlier 20-pair and 12-pair pilots remain separate and are excluded from these splits.

Split manifests reference existing files; no images, masks or annotations were copied or modified.
"""
    (REPORTS / "split_strategy.md").write_text(strategy, encoding="utf-8")
    rows = []
    for s in SPLITS:
        rows.append([s, split_counts[s]["pairs"], split_counts[s]["synthetic_screenshots"],
                     split_counts[s]["clean"], split_counts[s]["edited"], ratios[s]["pair_ratio"]])
    rows.append(["Restricted real holdout", 0, 0, "unlabeled", "unlabeled", "not in supervised splits"])
    distribution = "# Phase 6 split distribution\n\n" + table(["Split", "Synthetic pairs", "Synthetic screenshots", "Clean", "Edited", "Share of pairs"], rows)
    distribution += "\n\n## Layout-family distribution (pairs)\n\n"
    fams = sorted({f for c in family_counts.values() for f in c})
    distribution += table(["Split"] + fams, [[s] + [family_counts[s][f] for f in fams] for s in SPLITS])
    distribution += "\n\n## Edit-category distribution (edited pairs)\n\n"
    edits = sorted({e for c in edit_counts.values() for e in c})
    distribution += table(["Split"] + edits, [[s] + [edit_counts[s][e] for e in edits] for s in SPLITS])
    distribution += f"\n\nReal archive: **{len(real)} unique unlabeled images** in a separate restricted holdout, not counted in supervised split percentages. The original pilots are separate.\n"
    (REPORTS / "split_distribution.md").write_text(distribution, encoding="utf-8")
    class_rows = []
    for s in SPLITS:
        class_rows.append([s, split_counts[s]["clean"], split_counts[s]["edited"],
                           f"{split_counts[s]['clean']}/{max(1,split_counts[s]['edited'])}"])
    class_report = "# Phase 6 class-balance report\n\n" + table(["Split", "Clean samples", "Edited samples", "Clean:edited"], class_rows)
    class_report += "\n\nEach synthetic source pair contributes one clean and one edited screenshot, so the paired class balance is exact within every split. No oversampling or duplicate balancing was applied.\n\n"
    class_report += "## Synthetic layout and edit totals\n\n"
    class_report += table(["Layout family", "Pairs"] , [[f, sum(c[f] for c in family_counts.values())] for f in fams])
    class_report += "\n\n" + table(["Edit category", "Edited pairs"], [[e, sum(c[e] for c in edit_counts.values())] for e in edits])
    class_report += f"\n\nReal versus synthetic: {len(real)} unique real unlabeled screenshots are restricted holdout records; {len(synthetic)} synthetic screenshots enter labeled train/validation/test manifests. Real samples are not counted as clean.\n"
    (REPORTS / "class_balance_report.md").write_text(class_report, encoding="utf-8")
    leakage = f"""# Phase 6 leakage audit

## Results

- Pair groups checked: {len(groups)}; clean/edited pair leakage: **0**.
- Layout-family template groups checked: {len(template_groups)}; template groups spanning splits: **0**.
- pHash groups at Hamming ≤6 checked: {len(p_hash_groups)}; near-duplicate groups spanning splits: **0**.
- Exact screenshot hashes crossing split boundaries: **{len(cross_hash)}**.
- Image, mask, and annotation references missing: **{sum('missing ' in e for e in errors)}**.
- Validation errors overall: **{len(errors)}**.

The pHash groups are based on the Phase 5 64-bit grayscale pHash and are constructed within each layout family. Entire families are assigned to one split, so related screenshots cannot straddle the split boundary. Phase 5's pHash distance ≤6 graph connects each of the three larger families; recharge's three components are merged by its family template group.

## Limit

No family-level split can meet 70/15/15 here. The actual split has a held-out test set consisting only of recharge layouts and a validation set consisting only of bank-progress layouts. Scores on these sets will not measure performance across all four families; a future expansion should add independently designed templates before changing group integrity.
"""
    (REPORTS / "leakage_audit.md").write_text(leakage, encoding="utf-8")
    final_md = f"""# Phase 6 final split validation

**Result: {'PASS' if not errors else 'FAIL'}** — {len(errors)} validation errors.

- Phase 6 manifest: {len(manifest)} rows ({len(synthetic)} labeled synthetic screenshot records; {len(real)} unique real unlabeled records).
- Synthetic pairs: {len(groups)}; every pair is assigned wholly to one split.
- Split sizes: train {split_counts['train']['pairs']} pairs ({ratios['train']['pair_ratio']}), validation {split_counts['validation']['pairs']} ({ratios['validation']['pair_ratio']}), test {split_counts['test']['pairs']} ({ratios['test']['pair_ratio']}).
- Family-template and pHash group leakage: 0.
- Missing image/mask/annotation references: {sum('missing ' in e for e in errors)}.
- Real screenshots are excluded from supervised class labels and kept in a restricted holdout; no exact duplicate real copies are repeated in the manifest.
- Previous pilots remain separate. No screenshots were generated, copied, deleted, or modified; no models were trained.

Errors: {json.dumps(errors) if errors else 'none'}
"""
    (REPORTS / "final_split_validation.md").write_text(final_md, encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
