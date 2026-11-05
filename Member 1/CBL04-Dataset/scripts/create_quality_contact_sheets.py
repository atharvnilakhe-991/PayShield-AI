"""Create non-destructive review sheets for real-image and pilot visual audits."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def thumbnail(path: Path, size: tuple[int, int], mode: str = "RGB") -> Image.Image:
    with Image.open(path) as src:
        im = src.convert(mode)
    bg = (0, 0, 0) if mode == "RGB" else 0
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new(mode, size, bg)
    canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return canvas


def make_real_sheets(source: Path, destination: Path) -> int:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(path)
    cols, rows_per_sheet, cw, ch, label_h = 3, 4, 180, 450, 26
    destination.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
    for page, start in enumerate(range(0, len(unique), cols * rows_per_sheet), 1):
        group = unique[start:start + cols * rows_per_sheet]
        sheet = Image.new("RGB", (cols * cw, rows_per_sheet * (ch + label_h)), (38, 38, 38))
        d = ImageDraw.Draw(sheet)
        for offset, path in enumerate(group):
            x, y = (offset % cols) * cw, (offset // cols) * (ch + label_h)
            d.text((x + 4, y + 4), f"{start + offset + 1:03d}  {path.name}", font=font, fill=(245, 245, 245))
            sheet.paste(thumbnail(path, (cw - 8, ch)), (x + 4, y + label_h))
        sheet.save(destination / f"real_unique_{page:02d}.jpg", quality=90)
    return len(unique)


def make_synthetic_sheets(root: Path, destination: Path) -> None:
    clean_dir = root / "data/synthetic_gpay/clean"
    edited_dir = root / "data/synthetic_gpay/edited"
    mask_dir = root / "data/synthetic_gpay/masks/edited"
    clean_files = sorted(clean_dir.glob("*_clean.png"))[:10]
    destination.mkdir(parents=True, exist_ok=True)
    cols, rows_per_page, cw, ch = 3, 2, 270, 800
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
    for page in range(5):
        group = clean_files[page * rows_per_page:(page + 1) * rows_per_page]
        sheet = Image.new("RGB", (cols * cw, rows_per_page * ch), (35, 35, 35))
        d = ImageDraw.Draw(sheet)
        for row, clean_path in enumerate(group):
            source = clean_path.name.removesuffix("_clean.png")
            items = [
                ("CLEAN", clean_path, "RGB"),
                ("EDITED", edited_dir / f"{source}_edited.png", "RGB"),
                ("EDIT MASK", mask_dir / f"{source}_edited_mask.png", "L"),
            ]
            for col, (label, path, mode) in enumerate(items):
                x, y = col * cw, row * ch
                if col == 0:
                    d.text((x + 4, y + 4), source, font=font, fill=(245, 245, 245))
                else:
                    d.text((x + 4, y + 4), label, font=font, fill=(245, 245, 245))
                im = thumbnail(path, (cw - 8, ch - 26), mode)
                if mode == "L":
                    im = im.convert("RGB")
                sheet.paste(im, (x + 4, y + 24))
        sheet.save(destination / f"synthetic_visual_audit_{page + 1:02d}.png", optimize=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "reports/quality_audit").resolve()
    count = make_real_sheets(root / "data/real_gpay/raw", output / "real_review_work")
    make_synthetic_sheets(root, output)
    print(f"Created temporary review sheets for {count} unique real images and contact sheets for 10 synthetic pairs: {output}")


if __name__ == "__main__":
    main()
