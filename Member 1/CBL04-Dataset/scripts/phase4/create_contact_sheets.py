from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = list(csv.DictReader((root / "metadata/phase4_manifest.csv").open(encoding="utf-8-sig", newline="")))
    families = sorted({row["layout_family"] for row in rows})
    output = root / "reports/phase4/visual_audit"
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    row_h, panel_w, panel_h, header_h = 570, 260, 540, 26
    for family in families:
        pairs = sorted({r["source_id"] for r in rows if r["layout_family"] == family})[:5]
        sheet = Image.new("RGB", (panel_w * 3, len(pairs) * row_h), (34, 34, 34))
        draw = ImageDraw.Draw(sheet)
        for index, sid in enumerate(pairs):
            clean = next(r for r in rows if r["source_id"] == sid and r["label"] == "0")
            edited = next(r for r in rows if r["source_id"] == sid and r["label"] == "1")
            panels = [("CLEAN", root / clean["image_path"], "RGB"),
                      ("EDITED", root / edited["image_path"], "RGB"),
                      ("EDIT MASK", root / edited["mask_path"], "L")]
            for col, (label, path, mode) in enumerate(panels):
                x, y = col * panel_w, index * row_h
                draw.text((x + 6, y + 5), f"{sid} · {label}", font=font, fill=(245, 245, 245))
                with Image.open(path) as source:
                    im = source.convert(mode)
                im.thumbnail((panel_w - 12, panel_h), Image.Resampling.LANCZOS)
                if mode == "L":
                    im = im.convert("RGB")
                tile = Image.new("RGB", (panel_w - 12, panel_h), (0, 0, 0))
                tile.paste(im, ((tile.width - im.width) // 2, (tile.height - im.height) // 2))
                sheet.paste(tile, (x + 6, y + header_h))
        sheet.save(output / f"{family}.png", optimize=False)
    print(f"Wrote {len(families)} contact sheets to {output}")


if __name__ == "__main__":
    main()
