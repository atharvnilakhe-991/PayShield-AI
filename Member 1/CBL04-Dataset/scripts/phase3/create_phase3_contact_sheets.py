from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    rows = list(csv.DictReader((root / "metadata/phase3_manifest.csv").open(encoding="utf-8-sig", newline="")))
    families = sorted({row["layout_family"] for row in rows})
    out = root / "reports/phase3_visual_pilot"
    out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    width, row_height, header_h = 810, 800, 28
    for family in families:
        clean_rows = [row for row in rows if row["layout_family"] == family and row["label"] == "0"]
        sheet = Image.new("RGB", (width, len(clean_rows) * (row_height + header_h)), (34, 34, 34))
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(clean_rows):
            sid = row["source_id"]
            files = [
                ("CLEAN", root / row["image_path"], "RGB"),
                ("EDITED", root / next(x["image_path"] for x in rows if x["source_id"] == sid and x["label"] == "1"), "RGB"),
                ("EDIT MASK", root / next(x["mask_path"] for x in rows if x["source_id"] == sid and x["label"] == "1"), "L"),
            ]
            for col, (label, path, mode) in enumerate(files):
                x, y = col * 270, index * (row_height + header_h)
                title = sid if col == 0 else label
                draw.text((x + 5, y + 5), title, font=font, fill=(245, 245, 245))
                with Image.open(path) as source:
                    im = source.convert(mode)
                im.thumbnail((260, row_height - 8), Image.Resampling.LANCZOS)
                if mode == "L":
                    im = im.convert("RGB")
                panel = Image.new("RGB", (260, row_height - 8), (0, 0, 0))
                panel.paste(im, ((panel.width - im.width) // 2, (panel.height - im.height) // 2))
                sheet.paste(panel, (x + 5, y + header_h))
        sheet.save(out / f"{family}.png", optimize=False)
    print(f"Wrote {len(families)} family contact sheets to {out}")


if __name__ == "__main__":
    main()
