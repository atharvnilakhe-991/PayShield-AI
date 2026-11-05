"""Reproducible fictional dark payment-screen pair generator (Pillow)."""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFont

W, H = 540, 1600
BG = (17, 17, 17)
CARD = (20, 20, 20)
TEXT = (232, 232, 232)
MUTED = (185, 185, 185)
ACCENT = (166, 196, 255)
GREEN = (52, 168, 83)
TEMPLATE_VERSION = "gpay-dark-demo-v1"


@dataclass
class Example:
    recipient: str
    sender: str
    amount: int
    date_text: str
    bank: str
    account: str
    phone: str
    recipient_upi: str
    sender_upi: str
    upi_id: str
    google_id: str


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def centered(draw: ImageDraw.ImageDraw, text: str, y: int, f: ImageFont.ImageFont,
             fill: tuple[int, int, int], x: int = W // 2) -> None:
    box = draw.textbbox((0, 0), text, font=f)
    draw.text((x - (box[2] - box[0]) / 2, y), text, font=f, fill=fill)


def fitted(text: str, max_width: int, start: int = 25, bold: bool = False) -> ImageFont.ImageFont:
    for size in range(start, 15, -1):
        f = font(size, bold)
        if f.getlength(text) <= max_width:
            return f
    return font(15, bold)


def rounded(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], radius: int,
            fill: tuple[int, int, int] | None = None,
            outline: tuple[int, int, int] | None = None, width: int = 1) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def render(e: Example) -> tuple[Image.Image, dict[str, list[tuple[int, int, int, int]]]]:
    """Render one full screenshot and return fixed field boxes in pixel coordinates."""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    fields: dict[str, list[tuple[int, int, int, int]]] = {}
    # Persistent synthetic warning, identical in both members of every pair.
    rounded(d, (154, 18, 386, 52), 13, fill=(64, 42, 12))
    centered(d, "SYNTHETIC / DEMO", 24, font(16, True), (255, 208, 105))

    d.ellipse((231, 102, 309, 180), fill=(0, 121, 107))
    centered(d, e.recipient[0].upper(), 113, font(42), (255, 255, 255))
    f_rec = fitted("To " + e.recipient, 470, 25)
    centered(d, "To " + e.recipient, 199, f_rec, TEXT)
    fields["recipient"] = [(225, 98, 315, 185), (40, 190, 500, 237), (61, 896, 479, 934)]
    centered(d, f"+91 ••••• ••{e.phone[-2:]}", 235, font(20), MUTED)

    f_amt = fitted(f"₹{e.amount:,}", 380, 76)
    centered(d, f"₹{e.amount:,}", 299, f_amt, TEXT)
    fields["amount"] = [(65, 270, 475, 402)]
    rounded(d, (192, 418, 348, 473), 28, fill=ACCENT)
    centered(d, "Pay again", 432, font(21), (16, 44, 82))
    d.ellipse((203, 509, 222, 528), fill=GREEN)
    d.line((208, 518, 212, 522, 218, 514), fill=(17, 65, 34), width=2)
    d.text((234, 507), "Completed", font=font(21), fill=TEXT)
    d.line((121, 560, 419, 560), fill=(65, 65, 65), width=1)
    centered(d, e.date_text, 582, font(20), TEXT)
    fields["date_time"] = [(65, 575, 475, 611)]

    # Transaction card and fixed rows. Field value boxes are the exact regions
    # repainted by the corresponding edited pair and therefore the mask boxes.
    rounded(d, (33, 652, 507, 1160), 22, fill=CARD, outline=(70, 70, 70), width=1)
    d.ellipse((57, 675, 118, 718), fill=(255, 255, 255))
    d.ellipse((68, 679, 108, 715), fill=(34, 157, 190))
    d.ellipse((85, 685, 91, 691), fill=(255, 255, 255))
    d.rectangle((87, 690, 89, 712), fill=(255, 255, 255))
    d.text((143, 682), f"{e.bank} {e.account}", font=fitted(f"{e.bank} {e.account}", 320, 23, True), fill=TEXT)
    d.line((34, 737, 506, 737), fill=(70, 70, 70), width=1)

    d.text((55, 766), "UPI transaction ID", font=font(20), fill=TEXT)
    d.text((55, 800), e.upi_id, font=font(20), fill=MUTED)
    fields["transaction_id"] = [(48, 794, 492, 829)]

    d.text((55, 850), "To: " + e.recipient, font=fitted("To: " + e.recipient, 430, 20, True), fill=TEXT)
    d.text((55, 888), e.recipient_upi, font=font(19), fill=MUTED)
    # Include both the To-name and the upper recipient field in recipient edits.
    fields["recipient"] = [(225, 98, 315, 185), (40, 190, 500, 237), (48, 844, 492, 881)]

    sender_line = "From: " + e.sender
    d.text((55, 944), sender_line, font=fitted(sender_line, 430, 19, True), fill=TEXT)
    d.text((55, 982), f"••••{e.sender_upi}", font=font(19), fill=MUTED)
    d.text((55, 1050), "Google transaction ID", font=font(20), fill=TEXT)
    d.text((55, 1085), e.google_id, font=font(20), fill=MUTED)

    centered(d, "SYNTHETIC / DEMO", 1210, font(15, True), (255, 208, 105))
    centered(d, "UPI", 1255, font(25, True), (235, 235, 235))
    # A small multicolor mark stands in for app branding without copying assets.
    d.rounded_rectangle((191, 1511, 204, 1531), radius=7, fill=(66, 133, 244))
    d.rounded_rectangle((200, 1509, 213, 1529), radius=7, fill=(52, 168, 83))
    d.rounded_rectangle((196, 1516, 209, 1536), radius=7, fill=(251, 188, 5))
    d.text((218, 1508), "Google Pay", font=font(23, True), fill=TEXT)
    return im, fields


def create_example(rng: random.Random, index: int) -> Example:
    names = ["Aarav Mehta", "Isha Kulkarni", "Kabir Sharma", "Mira Nair", "Rohan Desai",
             "Anaya Joshi", "Dev Malhotra", "Tara Iyer", "Neel Patil", "Sara Rao"]
    banks = ["Demo Bank", "Sample National Bank", "Fictional Co-op Bank", "Test Payments Bank"]
    date = datetime(2025, 1, 1) + timedelta(days=rng.randrange(650), minutes=rng.randrange(1440))
    def display_date(value: datetime) -> str:
        hour = value.hour % 12 or 12
        return f"{value.day:02d} {value.strftime('%b %Y')}, {hour}:{value.minute:02d}{'am' if value.hour < 12 else 'pm'}"
    return Example(
        recipient=rng.choice(names), sender=rng.choice(names), amount=rng.choice([125, 280, 450, 999, 1250, 2400]),
        date_text=display_date(date), bank=rng.choice(banks),
        account=f"{rng.randrange(1000, 9999)}", phone=f"{rng.randrange(10000000, 99999999)}",
        recipient_upi=f"••••{rng.randrange(1000,9999)}@demo on Google Pay",
        sender_upi=f"@{rng.choice(['demo','sample','fiction'])}",
        upi_id=f"DEMO{rng.randrange(10**9, 10**10)}", google_id=f"DEMO-{rng.randrange(10**8,10**9)}-{rng.randrange(10**4,10**5)}",
    )


def replacement(e: Example, field: str, rng: random.Random) -> tuple[Example, str, str]:
    before = asdict(e)
    value_key = {"date_time": "date_text", "transaction_id": "upi_id"}.get(field, field)
    if field == "amount":
        e.amount = e.amount + rng.choice([37, 61, 105])
    elif field == "recipient":
        choices = ["Nina Kapoor", "Vikram Sen", "Leela Menon", "Om Prakash"]
        e.recipient = rng.choice([n for n in choices if n != e.recipient])
    elif field == "date_time":
        parsed = datetime.strptime(e.date_text, "%d %b %Y, %I:%M%p") + timedelta(hours=2)
        hour = parsed.hour % 12 or 12
        e.date_text = f"{parsed.day:02d} {parsed.strftime('%b %Y')}, {hour}:{parsed.minute:02d}{'am' if parsed.hour < 12 else 'pm'}"
    else:
        e.upi_id = "DEMO" + str(rng.randrange(10**9, 10**10))
    return e, str(before[value_key]), str(getattr(e, value_key))


def paint_mask(size: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    for x1, y1, x2, y2 in boxes:
        d.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)
    return mask


def generate(root: Path, count: int = 20, seed: int = 20261103) -> None:
    manifest_target = root / "metadata/manifest.csv"
    if manifest_target.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {manifest_target}")
    dirs = {
        "clean": root / "data/synthetic_gpay/clean",
        "edited": root / "data/synthetic_gpay/edited",
        "clean_masks": root / "data/synthetic_gpay/masks/clean",
        "edited_masks": root / "data/synthetic_gpay/masks/edited",
        "annotations": root / "annotations/field_annotations",
        "bounding_boxes": root / "annotations/bounding_boxes",
        "metadata": root / "metadata",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    fields_cycle = ["amount", "recipient", "date_time", "transaction_id"]
    manifest: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    for i in range(1, count + 1):
        pair_seed = seed + i
        rng = random.Random(pair_seed)
        source_id = f"SYN-{seed}-{i:03d}"
        clean_data = create_example(rng, i)
        edited_field = fields_cycle[(i - 1) % len(fields_cycle)]
        clean_image, boxes = render(clean_data)
        clean_path = dirs["clean"] / f"{source_id}_clean.png"
        edited_path = dirs["edited"] / f"{source_id}_edited.png"
        cm_path = dirs["clean_masks"] / f"{source_id}_clean_mask.png"
        em_path = dirs["edited_masks"] / f"{source_id}_edited_mask.png"
        if any(p.exists() for p in (clean_path, edited_path, cm_path, em_path)):
            raise FileExistsError(f"Refusing to overwrite existing pilot output for {source_id}")
        clean_image.save(clean_path, format="PNG", optimize=False)
        clean_mask = Image.new("L", clean_image.size, 0)
        clean_mask.save(cm_path, format="PNG", optimize=False)

        edited_data = Example(**asdict(clean_data))
        edited_data, old_value, new_value = replacement(edited_data, edited_field, rng)
        edited_image, edited_boxes = render(edited_data)
        edited_image.save(edited_path, format="PNG", optimize=False)
        field_boxes = edited_boxes[edited_field]
        diff = ImageChops.difference(clean_image, edited_image).convert("RGB")
        raw_diff = diff.tobytes()
        exact_mask = Image.new("L", edited_image.size, 0)
        exact_mask.putdata([255 if raw_diff[offset] or raw_diff[offset + 1] or raw_diff[offset + 2] else 0
                            for offset in range(0, len(raw_diff), 3)])
        tight_boxes: list[tuple[int, int, int, int]] = []
        for x1, y1, x2, y2 in field_boxes:
            crop_box = exact_mask.crop((x1, y1, x2, y2))
            changed = crop_box.getbbox()
            if changed:
                tight_boxes.append((x1 + changed[0], y1 + changed[1], x1 + changed[2], y1 + changed[3]))
        exact_mask.save(em_path, format="PNG", optimize=False)
        ann = {
            "source_id": source_id, "edited_field": edited_field,
            "original_synthetic_value": old_value, "replacement_synthetic_value": new_value,
            "edit_method": "localized field replacement; same deterministic renderer and fixed layout",
            "bounding_boxes_xyxy": [list(b) for b in tight_boxes], "seed": pair_seed,
            "template_version": TEMPLATE_VERSION,
        }
        (dirs["annotations"] / f"{source_id}.json").write_text(__import__("json").dumps(ann, indent=2), encoding="utf-8")
        for label, image_path, mask_path in ((0, clean_path, cm_path), (1, edited_path, em_path)):
            manifest.append({"image_id": image_path.stem, "source_id": source_id, "filename": image_path.name,
                             "image_path": image_path.relative_to(root).as_posix(), "label": label,
                             "app_name": "Google Pay style (synthetic)", "theme": "dark", "is_synthetic": True,
                             "edited_field": edited_field if label else "", "edit_method": ann["edit_method"] if label else "",
                             "mask_path": mask_path.relative_to(root).as_posix(),
                             "bounding_box": __import__("json").dumps(ann["bounding_boxes_xyxy"] if label else []),
                             "template_version": TEMPLATE_VERSION, "random_seed": pair_seed, "split": "pilot_review"})
        for box in tight_boxes:
            box_rows.append({"source_id": source_id, "edited_field": edited_field,
                             "x_min": box[0], "y_min": box[1], "x_max": box[2], "y_max": box[3]})
    import csv
    import json
    with (root / "metadata/manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys())); writer.writeheader(); writer.writerows(manifest)
    with (root / "metadata/source_group_splits.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["source_id", "split"]); writer.writeheader()
        writer.writerows({"source_id": f"SYN-{seed}-{i:03d}", "split": "pilot_review"} for i in range(1, count + 1))
    with (dirs["bounding_boxes"] / "bounding_boxes.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(box_rows[0].keys())); writer.writeheader(); writer.writerows(box_rows)
    (root / "metadata/pilot_config.json").write_text(json.dumps({"count": count, "seed": seed, "template_version": TEMPLATE_VERSION}, indent=2), encoding="utf-8")
