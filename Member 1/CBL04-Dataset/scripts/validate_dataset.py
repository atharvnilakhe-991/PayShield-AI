from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageChops


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate pilot screenshots, masks, annotations, and pairs.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    manifest_path = root / "metadata/manifest.csv"
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8-sig", newline="")))
    seen_ids: set[str] = set()
    by_source: dict[str, dict[int, dict[str, str]]] = {}
    for row in rows:
        if row["image_id"] in seen_ids:
            errors.append(f"duplicate image_id {row['image_id']}")
        seen_ids.add(row["image_id"])
        by_source.setdefault(row["source_id"], {})[int(row["label"])] = row
        img_path, mask_path = root / row["image_path"], root / row["mask_path"]
        try:
            with Image.open(img_path) as im:
                im.load(); image = im.convert("RGB")
            with Image.open(mask_path) as m:
                m.load(); mask = m.convert("L")
        except Exception as exc:
            errors.append(f"cannot decode {row['image_path']} or mask: {exc}")
            continue
        if image.size != (540, 1600):
            errors.append(f"unexpected screenshot size {image.size}: {row['image_path']}")
        if mask.size != image.size:
            errors.append(f"mask dimension mismatch: {row['mask_path']}")
        if int(row["label"]) == 0 and mask.getbbox() is not None:
            errors.append(f"clean mask not black: {row['mask_path']}")
        if int(row["label"]) == 1:
            if mask.getbbox() is None:
                errors.append(f"edited mask empty: {row['mask_path']}")
            colors = mask.getcolors(maxcolors=3)
            if colors is None or set(value for _, value in colors) - {0, 255}:
                errors.append(f"edited mask is not binary: {row['mask_path']}")
            ann_path = root / "annotations/field_annotations" / f"{row['source_id']}.json"
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
            for box in ann["bounding_boxes_xyxy"]:
                x1, y1, x2, y2 = box
                if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
                    errors.append(f"invalid bbox {box} in {ann_path}")
    for source, members in by_source.items():
        if set(members) != {0, 1}:
            errors.append(f"source {source} does not have exactly one clean and edited image")
            continue
        clean_row, edit_row = members[0], members[1]
        with Image.open(root / clean_row["image_path"]) as im:
            clean = im.convert("RGB")
        with Image.open(root / edit_row["image_path"]) as im:
            edited = im.convert("RGB")
        with Image.open(root / edit_row["mask_path"]) as m:
            mask = m.convert("L")
        diff = ImageChops.difference(clean, edited).convert("RGB")
        exact = Image.new("L", diff.size, 0)
        raw_diff = diff.tobytes()
        exact.putdata([255 if raw_diff[offset] or raw_diff[offset + 1] or raw_diff[offset + 2] else 0
                       for offset in range(0, len(raw_diff), 3)])
        if ImageChops.difference(mask, exact).getbbox() is not None:
            errors.append(f"mask does not equal the exact changed pixels: {source}")
        ann = json.loads((root / "annotations/field_annotations" / f"{source}.json").read_text(encoding="utf-8"))
        covered = Image.new("L", mask.size, 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(covered)
        for x1, y1, x2, y2 in ann["bounding_boxes_xyxy"]:
            draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)
        uncovered = ImageChops.multiply(exact, Image.eval(covered, lambda px: 255 - px))
        if uncovered.getbbox() is not None:
            errors.append(f"bounding boxes do not cover every changed pixel: {source}")
        regions = {
            "amount": [(65, 270, 475, 402)],
            "recipient": [(225, 98, 315, 185), (40, 190, 500, 237), (48, 844, 492, 881)],
            "date_time": [(65, 575, 475, 611)],
            "transaction_id": [(48, 794, 492, 829)],
        }
        allowed = Image.new("L", exact.size, 0)
        allowed_draw = ImageDraw.Draw(allowed)
        expected_boxes: list[list[int]] = []
        for x1, y1, x2, y2 in regions[ann["edited_field"]]:
            allowed_draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)
            changed_box = exact.crop((x1, y1, x2, y2)).getbbox()
            if changed_box:
                expected_boxes.append([x1 + changed_box[0], y1 + changed_box[1],
                                       x1 + changed_box[2], y1 + changed_box[3]])
        outside_field = ImageChops.multiply(exact, Image.eval(allowed, lambda px: 255 - px))
        if outside_field.getbbox() is not None:
            errors.append(f"unrelated UI pixels changed outside field {ann['edited_field']}: {source}")
        if ann["bounding_boxes_xyxy"] != expected_boxes:
            errors.append(f"bounding boxes are not tight around changed pixels: {source}")
    summary = {"images": len(rows), "pairs": len(by_source), "errors": errors, "passed": not errors}
    report = root / "reports/validation_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
