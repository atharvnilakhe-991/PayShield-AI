from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/phase5"
FAMILIES = ["compact", "enriched", "bank_progress", "recharge"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def exact_mask(a: Image.Image, b: Image.Image) -> Image.Image:
    diff = ImageChops.difference(a.convert("RGB"), b.convert("RGB"))
    r, g, bl = diff.split()
    return ImageChops.lighter(ImageChops.lighter(r, g), bl).point(lambda v: 255 if v else 0)


def tight_boxes(mask: Image.Image, zones: list[list[int]]) -> list[list[int]]:
    boxes = []
    for x1, y1, x2, y2 in zones:
        if not (0 <= x1 < x2 <= mask.width and 0 <= y1 < y2 <= mask.height):
            raise ValueError(f"invalid allowed region {x1,y1,x2,y2}")
        found = mask.crop((x1, y1, x2, y2)).getbbox()
        if found:
            boxes.append([x1 + found[0], y1 + found[1], x1 + found[2], y1 + found[3]])
    return boxes


def p_hash(path: Path) -> int:
    n = 32
    im = Image.open(path).convert("L").resize((n, n), Image.Resampling.LANCZOS)
    pix = list(im.getdata())
    cos = [[math.cos(math.pi * (2 * x + 1) * u / (2 * n)) for x in range(n)] for u in range(8)]
    alpha = [math.sqrt(1 / n) if u == 0 else math.sqrt(2 / n) for u in range(8)]
    horizontal = [[sum(pix[y*n+x] * cos[u][x] for x in range(n)) * alpha[u]
                   for u in range(8)] for y in range(n)]
    coeffs = []
    for v in range(8):
        for u in range(8):
            if u == v == 0:
                continue
            coeffs.append(sum(horizontal[y][u] * cos[v][y] for y in range(n)) * alpha[v])
    med = median(coeffs)
    result = 0
    for bit in coeffs:
        result = (result << 1) | int(bit > med)
    return result


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def audit_image_file(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as im:
            fmt = im.format
            mode = im.mode
            size = im.size
            im.verify()
        with Image.open(path) as im:
            im.load()
        return {"path": path, "readable": True, "format": fmt, "mode": mode, "size": size}
    except Exception as exc:
        return {"path": path, "readable": False, "error": str(exc)}


def marker_in_region(im: Image.Image, y1: float, y2: float) -> bool:
    left, right = int(im.width * .2), int(im.width * .8)
    top, bottom = int(im.height * y1), min(im.height, int(im.height * y2))
    px = im.convert("RGB").crop((left, top, right, bottom)).load()
    for y in range(bottom - top):
        for x in range(right - left):
            r, g, b = px[x, y]
            if r > 200 and g > 130 and b < 180:
                return True
    return False


def audit_real() -> dict[str, Any]:
    paths = sorted(p for p in (ROOT / "data/real_gpay/raw").rglob("*") if p.is_file())
    file_records = [audit_image_file(p) for p in paths]
    decoded = [x for x in file_records if x.get("readable")]
    exact: dict[str, list[str]] = defaultdict(list)
    for p, rec in zip(paths, file_records):
        if rec.get("readable"):
            exact[sha(p)].append(p.relative_to(ROOT).as_posix())
    dup_groups = [v for v in exact.values() if len(v) > 1]
    unique = len(exact)
    return {
        "files": len(paths), "readable": len(decoded), "unreadable": len(paths) - len(decoded),
        "unique_by_sha256": unique, "exact_duplicate_copies": sum(len(x)-1 for x in dup_groups),
        "duplicate_groups": len(dup_groups), "formats": dict(Counter(x["format"] for x in decoded)),
        "dimension_counts": {f"{w}x{h}": n for (w, h), n in Counter(x["size"] for x in decoded).most_common()},
        "min_dimensions": [min(x["size"][0] for x in decoded), min(x["size"][1] for x in decoded)] if decoded else None,
        "max_dimensions": [max(x["size"][0] for x in decoded), max(x["size"][1] for x in decoded)] if decoded else None,
        "unreadable_files": [x["path"].relative_to(ROOT).as_posix() for x in file_records if not x.get("readable")],
        "duplicate_examples": dup_groups[:12],
    }


def audit_manifest(root: Path, manifest: Path, annotation_dir: Path | None, exact_annotations: bool) -> dict[str, Any]:
    rows = read_csv(manifest)
    errors: list[str] = []
    malformed: list[str] = []
    marker_errors: list[str] = []
    groups: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    image_records: list[dict[str, Any]] = []
    sha_map: dict[str, list[str]] = defaultdict(list)
    family_role_paths: dict[tuple[str, str], list[tuple[str, Path]]] = defaultdict(list)
    dimensions = Counter()
    formats = Counter()
    for row in rows:
        sid = row.get("source_id", "")
        label = str(row.get("label", ""))
        if label in groups[sid]:
            errors.append(f"duplicate label {label} for {sid}")
        groups[sid][label] = row
        image_path, mask_path = ROOT / row.get("image_path", ""), ROOT / row.get("mask_path", "")
        if not image_path.is_file() or not mask_path.is_file():
            errors.append(f"missing image or mask: {sid} label={label}")
            continue
        if row.get("filename") and row["filename"] != image_path.name:
            errors.append(f"manifest filename/path mismatch: {sid}")
        if row.get("image_id") and row["image_id"] != image_path.stem:
            errors.append(f"manifest image_id/path mismatch: {sid}")
        rec = audit_image_file(image_path)
        mrec = audit_image_file(mask_path)
        image_records.append(rec)
        if not rec["readable"] or not mrec["readable"]:
            errors.append(f"unreadable image or mask: {sid} label={label}")
            continue
        dimensions[rec["size"]] += 1
        formats[rec["format"]] += 1
        digest = sha(image_path)
        sha_map[digest].append(image_path.relative_to(ROOT).as_posix())
        if row.get("width") and int(row["width"]) != rec["size"][0]:
            errors.append(f"manifest width mismatch: {sid}")
        if row.get("height") and int(row["height"]) != rec["size"][1]:
            errors.append(f"manifest height mismatch: {sid}")
        if mrec["size"] != rec["size"]:
            errors.append(f"image/mask dimension mismatch: {sid}")
        if row.get("is_synthetic", "").lower() in ("true", "1"):
            with Image.open(image_path) as src:
                marker_image = src.convert("RGB")
            if not marker_in_region(marker_image, .01, .10) or not marker_in_region(marker_image, .84, .99):
                marker_errors.append(f"{image_path.relative_to(ROOT).as_posix()}: missing top/bottom synthetic marker")
        if row.get("theme") and row["theme"].lower() != "dark":
            errors.append(f"non-dark synthetic row: {sid}")
        family = row.get("layout_family", "unknown")
        family_role_paths[(family, "clean" if label == "0" else "edited")].append((sid, image_path))
        try:
            with Image.open(mask_path) as msrc:
                msrc.load(); mask = msrc.convert("L")
            if label == "0" and mask.getbbox() is not None:
                errors.append(f"clean mask not black: {sid}")
        except Exception as exc:
            errors.append(f"mask load error {sid}: {exc}")
    dup_paths = [v for v in sha_map.values() if len(v) > 1]
    for sid, pair in groups.items():
        if set(pair) != {"0", "1"}:
            errors.append(f"unpaired source_id: {sid} labels={sorted(pair)}")
            continue
        clean_row, edit_row = pair["0"], pair["1"]
        if clean_row.get("layout_family") != edit_row.get("layout_family"):
            errors.append(f"family mismatch in pair: {sid}")
        if clean_row.get("width") and edit_row.get("width") and clean_row["width"] != edit_row["width"]:
            errors.append(f"width mismatch in pair: {sid}")
        if clean_row.get("height") and edit_row.get("height") and clean_row["height"] != edit_row["height"]:
            errors.append(f"height mismatch in pair: {sid}")
        if not annotation_dir:
            continue
        ann_path = annotation_dir / f"{sid}.json"
        if not ann_path.is_file():
            errors.append(f"missing annotation: {sid}")
            continue
        try:
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
            required = ["source_id", "layout_family", "edited_field", "edit_method", "bounding_boxes_xyxy", "allowed_regions_xyxy",
                        "random_seed", "width", "height", "template_version", "original_fictional_value", "replacement_fictional_value"]
            missing = [x for x in required if x not in ann]
            if missing:
                malformed.append(f"{sid}: missing {missing}")
                continue
            if ann["source_id"] != sid or ann["layout_family"] != edit_row.get("layout_family"):
                errors.append(f"annotation points to wrong pair/family: {sid}")
            with Image.open(ROOT / clean_row["image_path"]) as im:
                clean = im.convert("RGB")
            with Image.open(ROOT / edit_row["image_path"]) as im:
                edited = im.convert("RGB")
            with Image.open(ROOT / edit_row["mask_path"]) as im:
                mask = im.convert("L")
            expected = exact_mask(clean, edited)
            if ImageChops.difference(expected, mask).getbbox() is not None:
                errors.append(f"edited mask differs from pixel diff: {sid}")
            boxes = ann["bounding_boxes_xyxy"]
            if not isinstance(boxes, list) or not boxes:
                malformed.append(f"{sid}: empty or malformed bounding_boxes_xyxy")
                continue
            for b in boxes:
                if not isinstance(b, list) or len(b) != 4 or not all(isinstance(v, int) for v in b):
                    malformed.append(f"{sid}: malformed box {b!r}")
                    continue
                x1, y1, x2, y2 = b
                if not (0 <= x1 < x2 <= edited.width and 0 <= y1 < y2 <= edited.height):
                    errors.append(f"out-of-bounds bbox {sid}: {b}")
            if exact_annotations:
                expected_boxes = tight_boxes(expected, ann["allowed_regions_xyxy"])
                if expected_boxes != boxes:
                    errors.append(f"bbox not exact tight edited region: {sid}")
                allowed = Image.new("L", expected.size, 0)
                draw = ImageDraw.Draw(allowed)
                for x1, y1, x2, y2 in ann["allowed_regions_xyxy"]:
                    draw.rectangle((x1, y1, x2-1, y2-1), fill=255)
                if ImageChops.multiply(expected, Image.eval(allowed, lambda v: 255-v)).getbbox():
                    errors.append(f"diff pixels outside target field: {sid}")
            if ann["edited_field"] != edit_row.get("edited_field"):
                errors.append(f"edited field metadata mismatch: {sid}")
            if ann["original_fictional_value"] == ann["replacement_fictional_value"]:
                errors.append(f"annotation edit values are identical: {sid}")
            if str(ann["width"]) != str(edit_row.get("width", ann["width"])) or str(ann["height"]) != str(edit_row.get("height", ann["height"])):
                errors.append(f"annotation dimension metadata mismatch: {sid}")
            if str(ann["random_seed"]) != str(edit_row.get("random_seed", ann["random_seed"])):
                errors.append(f"annotation seed metadata mismatch: {sid}")
            if ann["template_version"] != edit_row.get("template_version", ann["template_version"]):
                errors.append(f"annotation template version mismatch: {sid}")
            if not ann["edit_method"] or not edit_row.get("edit_method"):
                malformed.append(f"{sid}: missing edit method")
            if ann["edited_field"] == "other" and ann.get("edit_component") != "fictional bank account suffix":
                malformed.append(f"{sid}: unsupported or missing other edit component")
        except Exception as exc:
            malformed.append(f"{sid}: {type(exc).__name__}: {exc}")
    return {
        "rows": len(rows), "pairs": len(groups), "images_readable": len(image_records),
        "formats": dict(formats), "dimension_counts": {f"{w}x{h}": n for (w,h),n in dimensions.most_common()},
        "exact_duplicate_images": sum(len(v)-1 for v in dup_paths), "duplicate_groups": dup_paths[:20],
        "errors": errors, "malformed_annotations": malformed,
        "synthetic_marker_errors": marker_errors,
        "annotations_checked": len(groups) if annotation_dir else 0,
    }


def similarity_scan(role_paths: dict[tuple[str, str], list[tuple[str, Path]]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for (family, role), items in sorted(role_paths.items()):
        hashes = [(sid, p_hash(path)) for sid, path in items]
        nearest: dict[str, int] = {sid: 64 for sid, _ in hashes}
        near3 = near6 = phash_equal = 0
        close_examples = []
        for i, (sid_a, hash_a) in enumerate(hashes):
            for sid_b, hash_b in hashes[i+1:]:
                d = hamming(hash_a, hash_b)
                nearest[sid_a] = min(nearest[sid_a], d)
                nearest[sid_b] = min(nearest[sid_b], d)
                if d <= 3:
                    near3 += 1
                if d == 0:
                    phash_equal += 1
                if d <= 6:
                    near6 += 1
                    close_examples.append({"a": sid_a, "b": sid_b, "hamming": d})
        vals = list(nearest.values())
        records.append({"family": family, "role": role, "images": len(items),
                        "pairs_compared": len(items)*(len(items)-1)//2,
                        "nearest_hamming_median": median(vals) if vals else None,
                        "nearest_hamming_min": min(vals) if vals else None,
                        "pair_count_hamming_eq_0": phash_equal,
                        "pair_count_hamming_le_3": near3, "pair_count_hamming_le_6": near6,
                        "closest_pairs_le_6": sorted(close_examples, key=lambda x: (x["hamming"], x["a"], x["b"]))[:12]})
    return {"method": "64-bit grayscale pHash (32x32 normalized resize, 8x8 low-frequency DCT); Hamming distance",
            "warning": "Low pHash distance flags visual similarity, not duplicate identity; all pairs are generated from shared family templates.",
            "by_family_role": records}


def make_sheets(manifest: list[dict[str, str]], similarity: dict[str, Any]) -> None:
    out = OUT / "contact_sheets"
    out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    panel_w, panel_h, header, row_h = 260, 540, 28, 570
    for family in FAMILIES:
        pairs = sorted({r["source_id"] for r in manifest if r.get("layout_family") == family})[:8]
        sheet = Image.new("RGB", (panel_w * 3, row_h * len(pairs)), (35, 35, 35))
        draw = ImageDraw.Draw(sheet)
        for i, sid in enumerate(pairs):
            group = {r["label"]: r for r in manifest if r["source_id"] == sid}
            edit_field = group["1"].get("edited_field", "edit")
            items = [("CLEAN", ROOT / group["0"]["image_path"], "RGB"),
                     ("EDITED", ROOT / group["1"]["image_path"], "RGB"),
                     ("EDIT MASK", ROOT / group["1"]["mask_path"], "L")]
            for col, (name, path, mode) in enumerate(items):
                x, y = col*panel_w, i*row_h
                draw.text((x+5, y+5), f"{sid} · {edit_field} · {name}", font=font, fill="white")
                with Image.open(path) as src:
                    im = src.convert(mode)
                im.thumbnail((panel_w-10, panel_h), Image.Resampling.LANCZOS)
                if mode == "L": im = im.convert("RGB")
                tile = Image.new("RGB", (panel_w-10, panel_h), "black")
                tile.paste(im, ((tile.width-im.width)//2,(tile.height-im.height)//2))
                sheet.paste(tile, (x+5,y+header))
        sheet.save(out / f"{family}.png", optimize=False)
    rows = []
    for family in FAMILIES:
        group = next((x for x in similarity["by_family_role"] if x["family"] == family and x["role"] == "clean"), None)
        if group and group["closest_pairs_le_6"]:
            rows.append((family, group["closest_pairs_le_6"][0]))
    near = Image.new("RGB", (540, 4 * 430), (35, 35, 35))
    draw = ImageDraw.Draw(near)
    for i, (family, pair) in enumerate(rows):
        y = i * 430
        draw.text((6, y + 4), f"{family}: {pair['a']} / {pair['b']} · pHash Hamming {pair['hamming']}", font=font, fill="white")
        for col, sid in enumerate((pair["a"], pair["b"])):
            row = next(r for r in manifest if r["source_id"] == sid and r["label"] == "0")
            with Image.open(ROOT / row["image_path"]) as src:
                im = src.convert("RGB")
            im.thumbnail((260, 390), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (260, 390), "black")
            tile.paste(im, ((260-im.width)//2,(390-im.height)//2))
            near.paste(tile, (col*270+5, y+30))
    near.save(out / "near_similarity_examples.png", optimize=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    real = audit_real()
    manifests = {
        "phase2_original_pilot": (ROOT / "metadata/manifest.csv", None, False),
        "phase3_pilot": (ROOT / "metadata/phase3_manifest.csv", ROOT / "annotations/phase3/field_annotations", True),
        "phase4_bulk": (ROOT / "metadata/phase4_manifest.csv", ROOT / "annotations/phase4/field_annotations", True),
    }
    all_audits = {"real_dataset": real}
    phase4_roles: dict[tuple[str, str], list[tuple[str, Path]]] = {}
    for name, (manifest, annotation_dir, exact) in manifests.items():
        audit = audit_manifest(ROOT, manifest, annotation_dir, exact)
        all_audits[name] = audit
        if name == "phase4_bulk":
            rows = read_csv(manifest)
            phase4_roles = defaultdict(list)
            for row in rows:
                phase4_roles[(row["layout_family"], "clean" if row["label"] == "0" else "edited")].append((row["source_id"], ROOT / row["image_path"]))
    sim = similarity_scan(phase4_roles)
    all_audits["phase4_similarity"] = sim
    batch = []
    for i in range(1, 6):
        path = ROOT / f"reports/phase4/batches/batch_{i:02d}_validation.json"
        batch.append(json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"batch":i,"missing":True})
    all_audits["phase4_batch_reports"] = batch
    (OUT / "audit_results.json").write_text(json.dumps(all_audits, indent=2), encoding="utf-8")
    phase4_rows = read_csv(ROOT / "metadata/phase4_manifest.csv")
    make_sheets(phase4_rows, sim)
    print(json.dumps({"real": {k: real[k] for k in ["files","readable","unreadable","unique_by_sha256","exact_duplicate_copies"]},
                      "datasets": {k:{"rows":v.get("rows"),"pairs":v.get("pairs"),"errors":len(v.get("errors",[])),"malformed":len(v.get("malformed_annotations",[]))} for k,v in all_audits.items() if isinstance(v,dict) and "rows" in v},
                      "phase4_batches_passed":sum(bool(x.get("passed")) for x in batch),
                      "similarity_groups":len(sim["by_family_role"]),"contact_sheets":4},indent=2))


if __name__ == "__main__":
    main()
