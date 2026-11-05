from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import edit_value, example  # noqa: E402
from renderer import RenderVariant, render  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FAMILY_PRESETS = {
    "compact": [(540, 1324), (540, 1600), (637, 1600)],
    "enriched": [(478, 1599), (600, 1600), (540, 1478)],
    "bank_progress": [(478, 1599), (432, 1440), (540, 1600)],
    "recharge": [(432, 1440), (540, 1600), (600, 1600)],
}
FAMILY_TITLES = {
    "compact": "Compact transaction details",
    "enriched": "Merchant/context-enriched details",
    "bank_progress": "Bank payment-progress timeline",
    "recharge": "Recharge/utility detail summary",
}
EDIT_FIELDS = ["amount", "recipient", "date_time", "transaction_id"]


def exact_mask(clean: Image.Image, edited: Image.Image) -> Image.Image:
    raw = ImageChops.difference(clean.convert("RGB"), edited.convert("RGB")).tobytes()
    mask = Image.new("L", clean.size, 0)
    mask.putdata([255 if raw[i] or raw[i + 1] or raw[i + 2] else 0 for i in range(0, len(raw), 3)])
    return mask


def tight_boxes(mask: Image.Image, regions: list[tuple[int, int, int, int]]) -> list[list[int]]:
    boxes: list[list[int]] = []
    for x1, y1, x2, y2 in regions:
        hit = mask.crop((x1, y1, x2, y2)).getbbox()
        if hit:
            boxes.append([x1 + hit[0], y1 + hit[1], x1 + hit[2], y1 + hit[3]])
    return boxes


def generate(root: Path, per_family: int, seed: int, selected_families: list[str] | None = None) -> None:
    manifest_target = root / "metadata/phase3_manifest.csv"
    if manifest_target.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 3 manifest: {manifest_target}")
    dirs = {
        "clean": root / "data/synthetic_gpay_phase3/clean",
        "edited": root / "data/synthetic_gpay_phase3/edited",
        "clean_masks": root / "data/synthetic_gpay_phase3/masks/clean",
        "edited_masks": root / "data/synthetic_gpay_phase3/masks/edited",
        "field_annotations": root / "annotations/phase3/field_annotations",
        "metadata": root / "metadata",
        "bbox": root / "annotations/phase3",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, str]] = []
    global_index = 0
    families = selected_families or list(FAMILY_PRESETS)
    unknown = set(families) - set(FAMILY_PRESETS)
    if unknown:
        raise ValueError(f"Unknown layout family: {', '.join(sorted(unknown))}")
    if not families:
        raise ValueError("Select at least one layout family")
    for family_index, family in enumerate(families):
        if per_family < 1:
            raise ValueError("per_family must be positive")
        for item in range(1, per_family + 1):
            global_index += 1
            pair_seed = seed + family_index * 1000 + item
            rng = random.Random(pair_seed)
            data = example(rng, family)
            field = EDIT_FIELDS[(global_index - 1) % len(EDIT_FIELDS)]
            changed, original, replacement = edit_value(data, field, rng)
            width, height = FAMILY_PRESETS[family][(item - 1) % len(FAMILY_PRESETS[family])]
            variant = RenderVariant(width=width, height=height,
                                    font_scale=[0.94, 1.0, 1.06][(item - 1) % 3],
                                    spacing=[0.96, 1.0, 1.04][(item - 1) % 3],
                                    margin=[30, 35, 39][(item - 1) % 3],
                                    show_pay_again=[True, False, True][(item - 1) % 3],
                                    show_action_strip=(family == "bank_progress" and item != 2),
                                    context_chip=(family == "enriched" and item != 2))
            source_id = f"P3-{family}-{seed}-{item:02d}"
            paths = {
                "clean": dirs["clean"] / f"{source_id}_clean.png",
                "edited": dirs["edited"] / f"{source_id}_edited.png",
                "clean_mask": dirs["clean_masks"] / f"{source_id}_clean_mask.png",
                "edited_mask": dirs["edited_masks"] / f"{source_id}_edited_mask.png",
                "annotation": dirs["field_annotations"] / f"{source_id}.json",
            }
            exists = [path.exists() for path in paths.values()]
            if any(exists) and not all(exists):
                raise FileExistsError(f"Incomplete existing Phase 3 pair; refusing to overwrite: {source_id}")
            clean_image, clean_regions = render(data, family, variant)
            edited_image, edited_regions = render(changed, family, variant)
            if clean_image.size != (width, height) or edited_image.size != clean_image.size:
                raise RuntimeError(f"Renderer returned unexpected dimensions for {source_id}")
            mask = exact_mask(clean_image, edited_image)
            boxes = tight_boxes(mask, edited_regions[field])
            if not boxes or mask.getbbox() is None:
                raise RuntimeError(f"Empty edit mask for {source_id}")
            clean_mask = Image.new("L", clean_image.size, 0)
            annotation = {
                "source_id": source_id, "layout_family": family, "layout_family_title": FAMILY_TITLES[family],
                "edited_field": field, "original_fictional_value": original,
                "replacement_fictional_value": replacement,
                "edit_method": "same family renderer and variant; field replacement only",
                "bounding_boxes_xyxy": boxes, "random_seed": pair_seed,
                "allowed_regions_xyxy": [list(r) for r in edited_regions[field]],
                "template_version": "gpay-dark-v2", "width": width, "height": height,
                "font_scale": variant.font_scale, "spacing_scale": variant.spacing,
                "margin": variant.margin,
            }
            if all(exists):
                for path, expected in ((paths["clean"], clean_image), (paths["edited"], edited_image),
                                       (paths["clean_mask"], clean_mask), (paths["edited_mask"], mask)):
                    with Image.open(path) as existing:
                        if ImageChops.difference(existing.convert(expected.mode), expected).getbbox() is not None:
                            raise FileExistsError(f"Existing Phase 3 pair differs from deterministic rerun: {source_id}")
                if json.loads(paths["annotation"].read_text(encoding="utf-8")) != annotation:
                    raise FileExistsError(f"Existing Phase 3 annotation differs from deterministic rerun: {source_id}")
            else:
                clean_image.save(paths["clean"], format="PNG", optimize=False)
                edited_image.save(paths["edited"], format="PNG", optimize=False)
                clean_mask.save(paths["clean_mask"], format="PNG", optimize=False)
                mask.save(paths["edited_mask"], format="PNG", optimize=False)
                paths["annotation"].write_text(json.dumps(annotation, indent=2, ensure_ascii=False), encoding="utf-8")
            for label, image_path, mask_path in ((0, paths["clean"], paths["clean_mask"]),
                                                 (1, paths["edited"], paths["edited_mask"])):
                rows.append({"image_id": image_path.stem, "source_id": source_id,
                             "filename": image_path.name, "image_path": image_path.relative_to(root).as_posix(),
                             "label": label, "app_name": "Google Pay style (synthetic)", "theme": "dark",
                             "is_synthetic": True, "layout_family": family,
                             "edited_field": field if label else "", "edit_method": annotation["edit_method"] if label else "",
                             "mask_path": mask_path.relative_to(root).as_posix(),
                             "bounding_box": json.dumps(boxes if label else []),
                             "template_version": annotation["template_version"], "random_seed": pair_seed,
                             "width": width, "height": height, "split": "phase3_visual_review"})
            for box in boxes:
                box_rows.append({"source_id": source_id, "layout_family": family, "edited_field": field,
                                 "x_min": box[0], "y_min": box[1], "x_max": box[2], "y_max": box[3],
                                 "width": width, "height": height})
            source_rows.append({"source_id": source_id, "layout_family": family, "split": "phase3_visual_review"})

    def write_csv(path: Path, items: list[dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(items[0].keys()))
            writer.writeheader(); writer.writerows(items)

    write_csv(manifest_target, rows)
    write_csv(dirs["bbox"] / "phase3_bounding_boxes.csv", box_rows)
    write_csv(dirs["metadata"] / "phase3_source_group_splits.csv", source_rows)
    (dirs["metadata"] / "phase3_config.json").write_text(json.dumps({
        "seed": seed, "pairs_per_family": per_family, "families": list(FAMILY_PRESETS),
        "pair_count": len(source_rows), "bulk_generation": False,
    }, indent=2), encoding="utf-8")
    print(f"Generated {len(source_rows)} matched pairs ({len(rows)} screenshots) in {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the Phase 3 visual pilot; not a bulk generator run.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--per-family", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261130)
    parser.add_argument("--families", default=",".join(FAMILY_PRESETS),
                        help="comma-separated family names: compact,enriched,bank_progress,recharge")
    args = parser.parse_args()
    generate(args.root.resolve(), args.per_family, args.seed, [x.strip() for x in args.families.split(",") if x.strip()])


if __name__ == "__main__":
    main()
