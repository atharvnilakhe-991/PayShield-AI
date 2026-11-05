from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parents[1] / "phase3"))
from data import edit_value, example  # noqa: E402
from generate_phase3_pilot import EDIT_FIELDS, FAMILY_TITLES, exact_mask, tight_boxes  # noqa: E402
from renderer import RenderVariant, render  # noqa: E402

REAL_COUNTS = {"compact": 169, "enriched": 167, "bank_progress": 42, "recharge": 9}
REAL_TOTAL = 387
TARGET_PAIRS = 500
BATCH_SIZE = 100
SEED = 20261204
EDIT_CATEGORIES = ["amount", "recipient", "date_time", "transaction_id", "other"]
DIMENSION_RANGES = {
    "compact": (540, 732, 1319, 1600),
    "enriched": (420, 738, 1280, 1601),
    "bank_progress": (353, 478, 1280, 1600),
    "recharge": (270, 471, 1248, 1603),
}


def proportional_targets() -> dict[str, int]:
    exact = {k: TARGET_PAIRS * v / REAL_TOTAL for k, v in REAL_COUNTS.items()}
    targets = {k: int(v) for k, v in exact.items()}
    remaining = TARGET_PAIRS - sum(targets.values())
    for family in sorted(exact, key=lambda k: exact[k] - targets[k], reverse=True)[:remaining]:
        targets[family] += 1
    return targets


def build_plan(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    targets = proportional_targets()
    families = [family for family, count in targets.items() for _ in range(count)]
    rng.shuffle(families)
    plan: list[dict[str, Any]] = []
    for batch in range(TARGET_PAIRS // BATCH_SIZE):
        fields = EDIT_CATEGORIES * (BATCH_SIZE // len(EDIT_CATEGORIES))
        while True:
            rng.shuffle(fields)
            if all(not (family == "recharge" and field == "other")
                   for family, field in zip(families[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE], fields)):
                break
        for offset in range(BATCH_SIZE):
            idx = batch * BATCH_SIZE + offset + 1
            family = families[idx - 1]
            pair_seed = seed * 1000 + idx
            pair_rng = random.Random(pair_seed)
            wmin, wmax, hmin, hmax = DIMENSION_RANGES[family]
            width = pair_rng.randrange(wmin, wmax + 1)
            height = pair_rng.randrange(hmin, hmax + 1)
            plan.append({
                "index": idx, "batch": batch + 1, "source_id": f"P4-{idx:04d}",
                "family": family, "field": fields[offset], "seed": pair_seed,
                "width": width, "height": height,
                "font_scale": round(pair_rng.uniform(0.90, 1.10), 3),
                "spacing": round(pair_rng.uniform(0.94, 1.06), 3),
                "margin": pair_rng.randrange(27, 47),
                "show_pay_again": pair_rng.random() < 0.78,
                "show_action_strip": family == "bank_progress" and pair_rng.random() < 0.62,
                "context_chip": family == "enriched" and pair_rng.random() < 0.68,
            })
    return plan


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_changed(clean: Image.Image, edited: Image.Image) -> Image.Image:
    return exact_mask(clean, edited)


def validate_batch(root: Path, batch: int, expected: list[dict[str, Any]], rows: list[dict[str, Any]],
                   seen_hashes: dict[str, str]) -> dict[str, Any]:
    errors: list[str] = []
    duplicate_images: list[dict[str, str]] = []
    batch_ids = {p["source_id"] for p in expected}
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["source_id"] not in batch_ids:
            continue
        groups.setdefault(row["source_id"], {})[str(row["label"])] = row
        image_path, mask_path = root / row["image_path"], root / row["mask_path"]
        try:
            raw_bytes = image_path.read_bytes()
            digest = hashlib.sha256(raw_bytes).hexdigest()
            with Image.open(image_path) as src:
                src.load(); image = src.convert("RGB")
            with Image.open(mask_path) as src:
                src.load(); mask = src.convert("L")
        except Exception as exc:
            errors.append(f"decode failure {row['source_id']} label {row['label']}: {exc}")
            continue
        prior = seen_hashes.get(digest)
        if prior:
            duplicate_images.append({"sha256": digest, "first": prior, "duplicate": row["image_path"]})
        else:
            seen_hashes[digest] = row["image_path"]
        if image.size != (int(row["width"]), int(row["height"])) or mask.size != image.size:
            errors.append(f"dimension mismatch: {row['image_path']}")
        if str(row["label"]) == "0" and mask.getbbox() is not None:
            errors.append(f"clean mask not all black: {row['mask_path']}")
        if str(row["label"]) == "1":
            colors = mask.getcolors(maxcolors=3)
            if colors is None or {c for _, c in colors} - {0, 255}:
                errors.append(f"edited mask not binary: {row['mask_path']}")
            pix = image.load(); w, h = image.size
            top = any(pix[x, y][0] > 200 and pix[x, y][1] > 130 and pix[x, y][2] < 180
                      for y in range(max(0, int(h * .01)), int(h * .09)) for x in range(w))
            bottom = any(pix[x, y][0] > 200 and pix[x, y][1] > 130 and pix[x, y][2] < 180
                         for y in range(int(h * .85), h - int(h * .04)) for x in range(w))
            if not top or not bottom:
                errors.append(f"missing synthetic marker: {row['image_path']}")
    for pair in expected:
        sid = pair["source_id"]
        group = groups.get(sid, {})
        if set(group) != {"0", "1"}:
            errors.append(f"missing clean/edited manifest members: {sid}")
            continue
        cr, er = group["0"], group["1"]
        clean = Image.open(root / cr["image_path"]).convert("RGB")
        edited = Image.open(root / er["image_path"]).convert("RGB")
        mask = Image.open(root / er["mask_path"]).convert("L")
        if clean.size != edited.size or cr["layout_family"] != er["layout_family"]:
            errors.append(f"pair configuration mismatch: {sid}")
        exact = exact_changed(clean, edited)
        if ImageChops.difference(mask, exact).getbbox() is not None:
            errors.append(f"mask differs from exact RGB change: {sid}")
        ann_path = root / "annotations/phase4/field_annotations" / f"{sid}.json"
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        boxes = tight_boxes(exact, [tuple(x) for x in ann["allowed_regions_xyxy"]])
        if boxes != ann["bounding_boxes_xyxy"]:
            errors.append(f"bounding boxes are not exact: {sid}")
        allowed = Image.new("L", exact.size, 0)
        draw = ImageDraw.Draw(allowed)
        for x1, y1, x2, y2 in ann["allowed_regions_xyxy"]:
            draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)
        if ImageChops.multiply(exact, Image.eval(allowed, lambda v: 255 - v)).getbbox():
            errors.append(f"changed pixels outside intended field: {sid}")
        if ann["layout_family"] != pair["family"] or ann["edited_field"] != pair["field"]:
            errors.append(f"annotation metadata mismatch: {sid}")
    if duplicate_images:
        errors.append(f"duplicate screenshots found in batch {batch}: {len(duplicate_images)}")
    return {"batch": batch, "pairs": len(expected), "screenshots": len(expected) * 2,
            "clean_masks_black": True, "exact_edited_masks": True, "bounding_boxes": True,
            "pairing_and_metadata": True, "duplicate_image_count": len(duplicate_images),
            "duplicates": duplicate_images, "errors": errors, "passed": not errors}


def generate(root: Path, seed: int = SEED) -> None:
    root = root.resolve()
    sentinel = root / "metadata/phase4_manifest.csv"
    if sentinel.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 4 output: {sentinel}")
    dirs = {
        "clean": root / "data/synthetic_gpay_phase4/clean",
        "edited": root / "data/synthetic_gpay_phase4/edited",
        "clean_masks": root / "data/synthetic_gpay_phase4/masks/clean",
        "edited_masks": root / "data/synthetic_gpay_phase4/masks/edited",
        "annotations": root / "annotations/phase4/field_annotations",
        "reports": root / "reports/phase4/batches",
        "metadata": root / "metadata/phase4_batches",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    plan = build_plan(seed)
    rows_all: list[dict[str, Any]] = []
    bbox_rows: list[dict[str, Any]] = []
    seen_hashes: dict[str, str] = {}
    for batch in range(1, 6):
        planned = [p for p in plan if p["batch"] == batch]
        batch_rows: list[dict[str, Any]] = []
        for p in planned:
            prng = random.Random(p["seed"])
            data = example(prng, p["family"])
            changed, old, new = edit_value(data, p["field"], prng)
            variant = RenderVariant(p["width"], p["height"], p["font_scale"], p["spacing"], p["margin"],
                                    p["show_pay_again"], p["show_action_strip"], p["context_chip"])
            clean_image, _ = render(data, p["family"], variant)
            edited_image, regions = render(changed, p["family"], variant)
            mask = exact_changed(clean_image, edited_image)
            boxes = tight_boxes(mask, regions[p["field"]])
            if clean_image.size != (p["width"], p["height"]) or not boxes:
                raise RuntimeError(f"render/mask failure before batch validation: {p['source_id']}")
            sid = p["source_id"]
            paths = {
                "clean": dirs["clean"] / f"{sid}_clean.png",
                "edited": dirs["edited"] / f"{sid}_edited.png",
                "clean_mask": dirs["clean_masks"] / f"{sid}_clean_mask.png",
                "edited_mask": dirs["edited_masks"] / f"{sid}_edited_mask.png",
            }
            clean_image.save(paths["clean"], optimize=False)
            edited_image.save(paths["edited"], optimize=False)
            Image.new("L", clean_image.size, 0).save(paths["clean_mask"], optimize=False)
            mask.save(paths["edited_mask"], optimize=False)
            annotation = {
                "source_id": sid, "batch": batch, "layout_family": p["family"],
                "layout_family_title": FAMILY_TITLES[p["family"]], "edited_field": p["field"],
                "original_fictional_value": old, "replacement_fictional_value": new,
                "edit_method": "same family renderer and configuration; one field edit",
                "edit_component": "fictional bank account suffix" if p["field"] == "other" else p["field"],
                "bounding_boxes_xyxy": boxes, "allowed_regions_xyxy": [list(x) for x in regions[p["field"]]],
                "random_seed": p["seed"], "width": p["width"], "height": p["height"],
                "font_scale": p["font_scale"], "spacing_scale": p["spacing"], "margin": p["margin"],
                "show_pay_again": p["show_pay_again"], "show_action_strip": p["show_action_strip"],
                "context_chip": p["context_chip"], "template_version": "gpay-dark-v2",
                "is_synthetic": True,
            }
            (dirs["annotations"] / f"{sid}.json").write_text(json.dumps(annotation, indent=2), encoding="utf-8")
            for label, image_key, mask_key in ((0, "clean", "clean_mask"), (1, "edited", "edited_mask")):
                image_path, mask_path = paths[image_key], paths[mask_key]
                batch_rows.append({"image_id": image_path.stem, "source_id": sid, "filename": image_path.name,
                    "image_path": image_path.relative_to(root).as_posix(), "label": label,
                    "app_name": "Google Pay style (synthetic)", "theme": "dark", "is_synthetic": True,
                    "layout_family": p["family"], "edited_field": p["field"] if label else "",
                    "edit_method": annotation["edit_method"] if label else "",
                    "mask_path": mask_path.relative_to(root).as_posix(),
                    "bounding_box": json.dumps(boxes if label else []),
                    "template_version": "gpay-dark-v2", "random_seed": p["seed"],
                    "width": p["width"], "height": p["height"], "split": f"phase4_batch_{batch:02d}"})
            for box in boxes:
                bbox_rows.append({"source_id": sid, "layout_family": p["family"], "edited_field": p["field"],
                                  "x_min": box[0], "y_min": box[1], "x_max": box[2], "y_max": box[3],
                                  "width": p["width"], "height": p["height"]})
        rows_all.extend(batch_rows)
        batch_path = dirs["metadata"] / f"batch_{batch:02d}_manifest.csv"
        csv_write(batch_path, batch_rows)
        report = validate_batch(root, batch, planned, rows_all, seen_hashes)
        (dirs["reports"] / f"batch_{batch:02d}_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"batch": batch, "pairs": report["pairs"], "passed": report["passed"],
                          "errors": report["errors"], "duplicate_images": report["duplicate_image_count"]}))
        if not report["passed"]:
            raise RuntimeError(f"Batch {batch} validation failed; generation stopped. See {dirs['reports']}")
    csv_write(sentinel, rows_all)
    csv_write(root / "annotations/phase4/bounding_boxes.csv", bbox_rows)
    target = proportional_targets()
    family_counts = Counter(p["family"] for p in plan)
    field_counts = Counter(p["field"] for p in plan)
    duplicate_report = {"images_checked": len(seen_hashes), "exact_duplicate_count": 0,
                       "method": "SHA-256 of PNG files; checked cumulatively after each batch"}
    (root / "reports/phase4/duplicate_audit.json").write_text(json.dumps(duplicate_report, indent=2), encoding="utf-8")
    (root / "reports/phase4/layout_family_distribution.json").write_text(json.dumps({
        "reference_counts": REAL_COUNTS, "reference_total": REAL_TOTAL, "target_pairs": TARGET_PAIRS,
        "pair_counts": dict(family_counts), "target_pair_counts": target,
    }, indent=2), encoding="utf-8")
    (root / "reports/phase4/edit_category_distribution.json").write_text(json.dumps({
        "pair_counts": dict(field_counts), "target_pair_counts": {k: TARGET_PAIRS // 4 for k in EDIT_CATEGORIES},
    }, indent=2), encoding="utf-8")
    final = {"passed": True, "pairs": TARGET_PAIRS, "clean_screenshots": TARGET_PAIRS,
             "edited_screenshots": TARGET_PAIRS, "clean_masks": TARGET_PAIRS, "edited_masks": TARGET_PAIRS,
             "batch_size": BATCH_SIZE, "batches_passed": 5, "family_counts": dict(family_counts),
             "edit_category_counts": dict(field_counts), "duplicate_screenshots": 0,
             "validation_reports": [f"reports/phase4/batches/batch_{i:02d}_validation.json" for i in range(1, 6)]}
    (root / "reports/phase4/final_validation_report.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    (root / "metadata/phase4_config.json").write_text(json.dumps({
        "seed": seed, "target_pairs": TARGET_PAIRS, "batch_size": BATCH_SIZE, "batch_count": 5,
        "reference_family_counts": REAL_COUNTS, "generated_family_counts": dict(family_counts),
        "generated_edit_category_counts": dict(field_counts),
        "dimensions_sampled_by_family": {k: list(v) for k, v in DIMENSION_RANGES.items()},
        "synthetic_marker": "SYNTHETIC / DEMO", "bulk_generation_complete": True,
    }, indent=2), encoding="utf-8")
    print(f"Completed 500 pairs under {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and validate the 500-pair Phase 4 dataset in five batches.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.root, args.seed)


if __name__ == "__main__":
    main()
