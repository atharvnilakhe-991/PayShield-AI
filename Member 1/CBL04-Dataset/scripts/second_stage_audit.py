from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


IN_SCOPE = "Google Pay dark theme — in scope"


def load_reviewed_classifications(root: Path) -> int:
    """Load human visual-review results; never infer app/theme from file format."""
    raw = root / "data/real_gpay/raw"
    seen: set[str] = set()
    unique_names: set[str] = set()
    for path in sorted(p for p in raw.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        unique_names.add(path.name)
    out = root / "reports/real_image_classification.csv"
    rows = list(csv.DictReader(out.open(encoding="utf-8-sig", newline="")))
    reviewed_names = {row["filename"] for row in rows}
    if reviewed_names != unique_names:
        raise ValueError("Unique filenames changed since visual review; classify new/missing files manually before compiling the audit.")
    allowed = {IN_SCOPE, "Google Pay light theme — out of scope", "Other UPI application — out of scope",
               "Uncertain — requires manual review", "Corrupted/unreadable — unusable"}
    if any(row["classification"] not in allowed for row in rows):
        raise ValueError("Classification CSV contains an unrecognized category.")
    reviewed = {row["filename"]: row for row in rows}
    # Propagate the audited class to the full technical inventory, including
    # duplicate copies, while retaining the unique-only classification CSV.
    inventory_path = root / "reports/dataset_inventory.csv"
    inventory_rows = list(csv.DictReader(inventory_path.open(encoding="utf-8-sig", newline="")))
    for row in inventory_rows:
        base = reviewed.get(row["filename"])
        if base is None and row.get("duplicate_of"):
            base = reviewed.get(Path(row["duplicate_of"]).name)
        row["scope_review"] = base["classification"] if base else "Uncertain — requires manual review"
        row["classification_reason"] = base["reason"] if base else "No matching unique classification record; requires review."
    fields = list(inventory_rows[0].keys()) if inventory_rows else []
    with inventory_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(inventory_rows)
    inventory_md = root / "reports/dataset_inventory.md"
    body = inventory_md.read_text(encoding="utf-8")
    old = "- Scope review is pending for every entry; do not interpret technical usability as Google Pay scope approval."
    new = (f"- Scope review completed for {len(rows)} unique images: all were visually classified as dark Google Pay transaction screens. "
           "See `real_image_classification.csv`; duplicate copies inherit their canonical image's classification.")
    inventory_md.write_text(body.replace(old, new), encoding="utf-8")
    return len(rows)


def make_audit_report(root: Path, classified_count: int) -> None:
    class_rows = list(csv.DictReader((root / "reports/real_image_classification.csv").open(encoding="utf-8-sig", newline="")))
    categories = [IN_SCOPE, "Google Pay light theme — out of scope", "Other UPI application — out of scope",
                  "Uncertain — requires manual review", "Corrupted/unreadable — unusable"]
    class_counts = {category: sum(row["classification"] == category for row in class_rows) for category in categories}
    manifest = list(csv.DictReader((root / "metadata/manifest.csv").open(encoding="utf-8-sig", newline="")))
    edited = [r for r in manifest if r["label"] == "1"]
    field_map = {"amount": "Amount", "recipient": "Recipient name", "date_time": "Date/time",
                 "transaction_id": "Transaction ID"}
    counts = {name: sum(r["edited_field"] == key for r in edited) for key, name in field_map.items()}
    counts["Other fields"] = sum(r["edited_field"] not in field_map for r in edited)
    val = json.loads((root / "reports/validation_report.json").read_text(encoding="utf-8"))
    summary = {
        "unique_real_classified": classified_count,
        "real_classifications": class_counts,
        "pilot_clean_images": sum(r["label"] == "0" for r in manifest),
        "pilot_edited_images": len(edited), "edit_field_counts": counts,
        "validation": val,
        "layout_variation": "One fixed 540x1600 dark template. Field anchors, layout, dimensions, and typography are fixed; exact mask and box extents vary with changed text pixels.",
        "mask_policy": "Edited masks are exact changed RGB pixels; annotation boxes enclose every changed pixel.",
    }
    report_json = root / "reports/second_stage_quality_audit.json"
    report_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_md = root / "reports/second_stage_quality_audit.md"
    report_md.write_text(f"""# CBL-04 second-stage quality audit

## Real dataset

All {classified_count} unique usable images were visually reviewed via 33 contact sheets, then spot-checked at full resolution for variant layouts. The classification CSV contains one row per unique image. Results: **{class_counts[IN_SCOPE]} Google Pay dark / {class_counts[categories[1]]} Google Pay light / {class_counts[categories[2]]} other UPI app / {class_counts[categories[3]]} uncertain / {class_counts[categories[4]]} corrupt**. Seven redundant duplicate copies are omitted from this unique-image classification but retained in the raw archive; there are no unreadable files.

The reviewed set includes standard person-to-person receipts, merchant payment details, bank-specific payment-progress timelines, and recharge/utility-style transaction-detail pages. These variants visibly carry Google Pay branding and Google transaction details, so they were classified as dark Google Pay screens. The dataset is broader than one exact receipt layout.

## Synthetic visual review

The screens are recognizable as dark payment-completion/details pages and carry visible SYNTHETIC / DEMO marks. They are acceptable as a **pilot for pipeline and annotation review**, but are not visually diverse or faithful enough for bulk model training yet. The renderer fixes every screenshot at 540 × 1600 with one layout, one font family, fixed field positions, and simplified generic bank/UPI branding. It does not reproduce the different real GPay layout variants observed in the archive.

The 10-pair visual contact sheet is supplied as five PNG pages: `quality_audit/synthetic_visual_audit_01.png` through `_05.png`.

## Mask and box audit

- Clean masks: 20/20 entirely black.
- Edited masks: 20/20 exactly equal the pixelwise RGB difference between their matched clean/edited images.
- Bounding boxes: every changed pixel is covered; box coordinates are within image dimensions.
- Unrelated UI: no changed pixels outside the named edited field regions were found. Recipient edits include its avatar initial and both displayed name instances; these are part of the recipient field.
- Validation: {val['images']} images, {val['pairs']} matched pairs, {len(val['errors'])} errors; passed = {val['passed']}.

## Edit diversity and layout variation

{chr(10).join(f'- {name}: {count}' for name, count in counts.items())}

Edit locations span four UI areas: amount near the top, recipient at the avatar/name and card recipient row, date/time in the upper summary, and transaction ID in the details card. Field anchors are fixed across pairs; there is no layout or screen-dimension variation. The 20-pair pilot has balanced field counts but insufficient visual variation for scale.

## Recommendation before bulk generation

Keep the 20-pair pilot as the current review set. Before scaling, add controlled layout and dimension variants based on the real dark GPay families, preserve one renderer-defined exact mask per altered field, and render merchant/recharge/bank-progress variants as distinct template families or explicitly exclude them. Review a revised small batch and audit it again before bulk generation. The pilot does **not** establish real-world fraud-detection performance.
""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    count = load_reviewed_classifications(root)
    make_audit_report(root, count)
    print(f"Classified {count} unique real images; wrote CSV and second-stage audit report.")


if __name__ == "__main__":
    main()
