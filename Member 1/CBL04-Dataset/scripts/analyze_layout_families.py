"""Unsupervised visual grouping to support manual layout-family review."""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def vector(path: Path) -> tuple[list[float], tuple[int, int]]:
    with Image.open(path) as src:
        size = src.size
        im = src.convert("L")
        im.thumbnail((18, 54), Image.Resampling.BILINEAR)
        canvas = Image.new("L", (18, 54), 0)
        canvas.paste(im, ((18 - im.width) // 2, (54 - im.height) // 2))
        # Downsample to 12 x 36. Mild contrast compression reduces text-content noise.
        small = canvas.resize((12, 36), Image.Resampling.BOX)
        vals = [v / 255.0 for v in small.getdata()]
        vals.extend([size[0] / 800.0, size[1] / 1800.0])
        return vals, size


def dist2(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--clusters", type=int, default=6)
    args = parser.parse_args()
    root = args.root.resolve()
    files: list[tuple[Path, list[float], tuple[int, int]]] = []
    seen: set[str] = set()
    for path in sorted(p for p in (root / "data/real_gpay/raw").rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        feat, size = vector(path)
        files.append((path, feat, size))
    rng = random.Random(17)
    centers = [files[rng.randrange(len(files))][1]]
    while len(centers) < args.clusters:
        idx = max(range(len(files)), key=lambda i: min(dist2(files[i][1], c) for c in centers))
        centers.append(files[idx][1])
    labels = [0] * len(files)
    for _ in range(9):
        labels = [min(range(len(centers)), key=lambda c: dist2(feat, centers[c])) for _, feat, _ in files]
        for ci in range(len(centers)):
            members = [feat for (_, feat, _), label in zip(files, labels) if label == ci]
            if members:
                centers[ci] = [sum(row[j] for row in members) / len(members) for j in range(len(centers[ci]))]
    out = root / "reports/layout_visual_clusters.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "width", "height", "visual_cluster"])
        writer.writeheader()
        for (path, _, size), label in zip(files, labels):
            writer.writerow({"filename": path.name, "width": size[0], "height": size[1], "visual_cluster": label + 1})
    print(dict(Counter(labels)))
    print(f"Wrote {len(files)} unique images to {out}")
    groups: dict[int, list[Path]] = {}
    for (path, _, _), label in zip(files, labels):
        groups.setdefault(label + 1, []).append(path)
    sheet_root = root / "reports/layout_cluster_sheets"
    sheet_root.mkdir(parents=True, exist_ok=True)
    label_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
    cols, rows_per_sheet, cw, ch, label_h = 3, 4, 180, 450, 26
    for cluster, paths in groups.items():
        for page, start in enumerate(range(0, len(paths), cols * rows_per_sheet), 1):
            subset = paths[start:start + cols * rows_per_sheet]
            sheet = Image.new("RGB", (cols * cw, rows_per_sheet * (ch + label_h)), (38, 38, 38))
            draw = ImageDraw.Draw(sheet)
            for offset, path in enumerate(subset):
                x, y = (offset % cols) * cw, (offset // cols) * (ch + label_h)
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    im.thumbnail((cw - 8, ch), Image.Resampling.LANCZOS)
                    panel = Image.new("RGB", (cw - 8, ch), (0, 0, 0))
                    panel.paste(im, ((panel.width - im.width) // 2, (panel.height - im.height) // 2))
                draw.text((x + 4, y + 4), f"C{cluster} {path.name}", font=label_font, fill=(245, 245, 245))
                sheet.paste(panel, (x + 4, y + label_h))
            sheet.save(sheet_root / f"cluster_{cluster:02d}_{page:02d}.jpg", quality=90)


if __name__ == "__main__":
    main()
