from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw


def tight_boxes(mask: Image.Image, zones: list[list[int]]) -> list[list[int]]:
    out = []
    for x1, y1, x2, y2 in zones:
        box = mask.crop((x1, y1, x2, y2)).getbbox()
        if box:
            out.append([x1 + box[0], y1 + box[1], x1 + box[2], y1 + box[3]])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = list(csv.DictReader((root / "metadata/phase3_manifest.csv").open(encoding="utf-8-sig", newline="")))
    errors: list[str] = []
    ids: set[str] = set()
    sources: dict[str, dict[str, dict[str, str]]] = {}
    family_counts: Counter[str] = Counter()
    for row in manifest:
        if row["image_id"] in ids:
            errors.append(f"duplicate image_id: {row['image_id']}")
        ids.add(row["image_id"])
        sources.setdefault(row["source_id"], {})[row["label"]] = row
        if row["label"] == "0":
            family_counts[row["layout_family"]] += 1
        img_path, mask_path = root / row["image_path"], root / row["mask_path"]
        try:
            with Image.open(img_path) as im:
                im.load(); image = im.convert("RGB")
            with Image.open(mask_path) as m:
                m.load(); mask = m.convert("L")
        except Exception as exc:
            errors.append(f"decode failure for {row['image_path']} / {row['mask_path']}: {exc}")
            continue
        expected_size = (int(row["width"]), int(row["height"]))
        if image.size != expected_size or mask.size != expected_size:
            errors.append(f"dimension mismatch: {row['image_path']}")
        if row["label"] == "0" and mask.getbbox() is not None:
            errors.append(f"clean mask is not all black: {row['mask_path']}")
        if row["label"] == "1":
            colors = mask.getcolors(maxcolors=3)
            if colors is None or {v for _, v in colors} - {0, 255}:
                errors.append(f"edited mask is not binary: {row['mask_path']}")
            ann_path = root / "annotations/phase3/field_annotations" / f"{row['source_id']}.json"
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
            for x1, y1, x2, y2 in ann["bounding_boxes_xyxy"]:
                if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
                    errors.append(f"invalid bounding box for {row['source_id']}: {[x1, y1, x2, y2]}")
            # Verify a visible synthetic indicator by its amber color in top and bottom regions.
            pixels = image.load()
            has_top = any(pixels[x, y][0] > 200 and pixels[x, y][1] > 130 and pixels[x, y][2] < 180
                          for y in range(max(0, int(image.height * .01)), int(image.height * .09))
                          for x in range(image.width))
            has_bottom = any(pixels[x, y][0] > 200 and pixels[x, y][1] > 130 and pixels[x, y][2] < 180
                             for y in range(int(image.height * .85), image.height - int(image.height * .04))
                             for x in range(image.width))
            if not has_top or not has_bottom:
                errors.append(f"SYNTHETIC / DEMO marker not found at both ends: {row['image_path']}")
    for source_id, group in sources.items():
        if set(group) != {"0", "1"}:
            errors.append(f"source {source_id} is missing clean/edited counterpart")
            continue
        clean_row, edited_row = group["0"], group["1"]
        with Image.open(root / clean_row["image_path"]) as im:
            clean = im.convert("RGB")
        with Image.open(root / edited_row["image_path"]) as im:
            edited = im.convert("RGB")
        with Image.open(root / edited_row["mask_path"]) as m:
            mask = m.convert("L")
        diff = ImageChops.difference(clean, edited).convert("RGB")
        raw = diff.tobytes()
        exact = Image.new("L", diff.size, 0)
        exact.putdata([255 if raw[i] or raw[i + 1] or raw[i + 2] else 0 for i in range(0, len(raw), 3)])
        if ImageChops.difference(mask, exact).getbbox() is not None:
            errors.append(f"mask does not match exact changed pixels: {source_id}")
        ann = json.loads((root / "annotations/phase3/field_annotations" / f"{source_id}.json").read_text(encoding="utf-8"))
        zones = ann["allowed_regions_xyxy"]
        expected_boxes = tight_boxes(exact, zones)
        if expected_boxes != ann["bounding_boxes_xyxy"]:
            errors.append(f"bounding boxes do not tightly match edited regions: {source_id}")
        allowed = Image.new("L", exact.size, 0)
        draw = ImageDraw.Draw(allowed)
        for x1, y1, x2, y2 in zones:
            draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)
        outside = ImageChops.multiply(exact, Image.eval(allowed, lambda v: 255 - v))
        if outside.getbbox() is not None:
            errors.append(f"unrelated UI pixels changed outside target field: {source_id}")
        if clean.size != edited.size or clean_row["layout_family"] != edited_row["layout_family"]:
            errors.append(f"matched pair variant mismatch: {source_id}")
    pairs = len(sources)
    family_pairs = Counter(row["layout_family"] for row in manifest if row["label"] == "0")
    if any(count < 3 for count in family_pairs.values()):
        errors.append("one or more identified layout families has fewer than three pairs")
    configs = [json.loads((root / "annotations/phase3/field_annotations" / f"{source}.json").read_text(encoding="utf-8")) for source in sources]
    field_counts = Counter(config["edited_field"] for config in configs)
    summary = {
        "screenshots": len(manifest), "matched_pairs": pairs,
        "pairs_per_family": dict(family_counts), "edited_field_counts": dict(field_counts),
        "unique_dimensions": sorted({f"{x['width']}x{x['height']}" for x in configs}),
        "font_scales": sorted({x["font_scale"] for x in configs}),
        "spacing_scales": sorted({x["spacing_scale"] for x in configs}),
        "errors": errors, "passed": not errors,
    }
    (root / "reports/phase3_validation_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
