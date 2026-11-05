from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory extracted real image files without modifying them.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path, default=None, help="Defaults to data/real_gpay/raw")
    args = parser.parse_args()
    root = args.root.resolve()
    source = (args.source or root / "data/real_gpay/raw").resolve()
    out_csv = root / "reports/dataset_inventory.csv"
    out_md = root / "reports/dataset_inventory.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    hashes: dict[str, str] = {}
    counts = Counter()
    dimensions = Counter()
    formats = Counter()
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        fmt, width, height, error = "", "", "", ""
        try:
            with Image.open(path) as im:
                fmt = str(im.format or "")
                im.verify()
            with Image.open(path) as im:
                im.load()
                width, height = im.size
            decode_status = "usable" if digest not in hashes else "duplicate_copy"
        except Exception as exc:
            decode_status, error = "corrupt", str(exc)
        duplicate_of = hashes.get(digest, "") if decode_status == "duplicate_copy" else ""
        if decode_status == "usable":
            hashes[digest] = path.relative_to(root).as_posix()
        rows.append({"filename": path.name, "relative_path": path.relative_to(root).as_posix(),
                     "bytes": path.stat().st_size, "extension": path.suffix.lower(), "format": fmt,
                     "width": width, "height": height, "sha256": digest, "decode_status": decode_status,
                     "duplicate_of": duplicate_of, "scope_review": "pending_manual_visual_review", "error": error})
        counts[decode_status] += 1
        if width and height:
            dimensions[f"{width}x{height}"] += 1
        if fmt:
            formats[fmt] += 1
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["filename"])
        writer.writeheader(); writer.writerows(rows)
    unique_usable = counts["usable"]
    dupcopies = counts["duplicate_copy"]
    common_dims = "\n".join(f"- {key}: {value}" for key, value in dimensions.most_common(12)) or "- No readable images"
    report = (
        "# Real dataset inventory\n\n"
        f"- Extracted image files: {len(rows)}\n"
        f"- Technically readable files: {len(rows) - counts['corrupt']}\n"
        f"- Unique usable image content: {unique_usable}\n"
        f"- Exact duplicate copies: {dupcopies}\n"
        f"- Corrupt/unreadable files: {counts['corrupt']}\n"
        f"- Decoded formats: {dict(formats)}\n"
        "- Original files are retained; this audit is read-only.\n"
        "- Scope review is pending for every entry; do not interpret technical usability as Google Pay scope approval.\n\n"
        "## Common dimensions\n\n" + common_dims + "\n\n"
        "## Scope notes\n\n"
        "The full collection needs visual scope labeling before using it as a Google Pay-only set. Spot checks found mostly dark transaction screens, including at least one visually distinct PhonePe completion presentation; app branding alone is not reliable enough to classify all entries automatically. No entries were discarded.\n"
    )
    out_md.write_text(report, encoding="utf-8")
    summary = {"files": len(rows), "unique_usable": unique_usable, "duplicate_copies": dupcopies,
               "corrupt": counts["corrupt"], "formats": dict(formats), "dimensions": dict(dimensions)}
    (root / "reports/dataset_inventory_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
