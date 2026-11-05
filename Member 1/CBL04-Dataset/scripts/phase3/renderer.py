"""Family-based GPay dark renderer. Clean and edited variants share exact geometry."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parents[2]
SPEC = json.loads((BASE / "templates/phase3/layout_families.json").read_text(encoding="utf-8"))
BG = (18, 18, 18)
CARD = (20, 20, 20)
TEXT = (238, 238, 238)
MUTED = (187, 187, 187)
ACCENT = (166, 196, 255)
GREEN = (52, 168, 83)
BLUE = (125, 178, 245)


@dataclass(frozen=True)
class RenderVariant:
    width: int
    height: int
    font_scale: float
    spacing: float
    margin: int
    show_pay_again: bool
    show_action_strip: bool
    context_chip: bool


def _font(size: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf")
    if not path.exists():
        path = Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), max(10, round(size))) if path.exists() else ImageFont.load_default()


def _text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, size: float,
          fill: tuple[int, int, int] = TEXT, bold: bool = False) -> None:
    draw.text((round(xy[0]), round(xy[1])), text, font=_font(size, bold), fill=fill)


def _center(draw: ImageDraw.ImageDraw, y: float, text: str, size: float,
            fill: tuple[int, int, int] = TEXT, bold: bool = False, x: float | None = None) -> None:
    f = _font(size, bold)
    x = W_CURRENT / 2 if x is None else x
    draw.text((round(x - f.getlength(text) / 2), round(y)), text, font=f, fill=fill)


def _fit(text: str, available: float, size: float, bold: bool = False) -> ImageFont.ImageFont:
    while size > 13 and _font(size, bold).getlength(text) > available:
        size -= 1
    return _font(size, bold)


def _draw_gpay_footer(draw: ImageDraw.ImageDraw, w: int, h: int, scale: float) -> None:
    y = h - round(57 * scale)
    _center(draw, y, "Google Pay", 22 * scale, TEXT, True, x=w * 0.57)
    x, yy = round(w * 0.36), y + round(5 * scale)
    draw.rounded_rectangle((x, yy, x + round(12 * scale), yy + round(21 * scale)), radius=round(6 * scale), fill=(66, 133, 244))
    draw.rounded_rectangle((x + round(8 * scale), yy - round(2 * scale), x + round(20 * scale), yy + round(18 * scale)), radius=round(6 * scale), fill=(52, 168, 83))
    draw.rounded_rectangle((x + round(5 * scale), yy + round(6 * scale), x + round(17 * scale), yy + round(27 * scale)), radius=round(6 * scale), fill=(251, 188, 5))


def _base_canvas(variant: RenderVariant, family: str) -> tuple[Image.Image, ImageDraw.ImageDraw, dict[str, list[tuple[int, int, int, int]]], float, float]:
    global W_CURRENT
    W_CURRENT = variant.width
    sx, sy = variant.width / 540.0, variant.height / 1600.0
    scale = math.sqrt(sx * sy) * variant.font_scale
    spacing = variant.spacing
    im = Image.new("RGB", (variant.width, variant.height), BG)
    d = ImageDraw.Draw(im)
    regions: dict[str, list[tuple[int, int, int, int]]] = {"amount": [], "recipient": [], "date_time": [], "transaction_id": [], "other": []}
    # A protected synthetic marker is present at both ends of every family.
    mx1, mx2 = round(variant.width * 0.29), round(variant.width * 0.71)
    d.rounded_rectangle((mx1, round(18 * sy), mx2, round(51 * sy)), radius=round(15 * scale), fill=(67, 44, 13))
    _center(d, 25 * sy, "SYNTHETIC / DEMO", 15 * scale, (255, 210, 115), True)
    return im, d, regions, scale, spacing


def _box(x1: float, y1: float, x2: float, y2: float, sx: float, sy: float,
         top: float, spacing: float) -> tuple[int, int, int, int]:
    return (max(0, round(x1 * sx)), max(0, round(y1 * sy)), round(x2 * sx), round(y2 * sy))


def _card(draw: ImageDraw.ImageDraw, variant: RenderVariant, y: float, bottom: float,
          sx: float, sy: float, scale: float) -> tuple[float, float, float]:
    left = variant.margin * sx
    right = variant.width - variant.margin * sx
    top = y * sy
    end = bottom * sy
    draw.rounded_rectangle((round(left), round(top), round(right), round(end)), radius=round(22 * scale),
                           fill=CARD, outline=(70, 70, 70), width=max(1, round(scale)))
    return left, right, top


def _profile_and_summary(draw: ImageDraw.ImageDraw, value: dict[str, Any], variant: RenderVariant,
                         regions: dict[str, list[tuple[int, int, int, int]]], scale: float,
                         spacing: float, family: str) -> tuple[float, float, float, float]:
    w, h = variant.width, variant.height
    sx, sy = w / 540.0, h / 1600.0
    top = 68.0
    cy = (150.0 if family != "recharge" else 142.0)
    rad = 38 * scale
    draw.ellipse((round(w / 2 - rad), round(cy * sy - rad), round(w / 2 + rad), round(cy * sy + rad)), fill=(0, 121, 107))
    initials = value["recipient"][:1].upper()
    _center(draw, (cy - 27) * sy, initials, 36 * scale, (255, 255, 255), False)
    name = ("To " if family != "incoming" else "From ") + value["recipient"]
    name_y = 204.0 if family != "recharge" else 192.0
    name_font = _fit(name, w - 50 * sx, 24 * scale)
    f = _font(name_font.size, False)
    draw.text((round(w / 2 - f.getlength(name) / 2), round((top + (name_y - top) * spacing) * sy)), name, font=f, fill=TEXT)
    regions["recipient"].append(_box(205, 106, 335, 190, sx, sy, top, spacing))
    regions["recipient"].append(_box(30, 188, 510, 250, sx, sy, top, spacing))
    if family != "recharge":
        _center(draw, 239 * sy, f"+91 ••••• ••{value['phone_last']}", 19 * scale, MUTED)

    amount_y = 293.0 if family != "recharge" else 278.0
    amount_text = f"₹{value['amount']:,}"
    amt_f = _fit(amount_text, w - 110 * sx, 72 * scale)
    f = _font(amt_f.size, False)
    draw.text((round(w / 2 - f.getlength(amount_text) / 2), round(amount_y * sy)), amount_text, font=f, fill=TEXT)
    regions["amount"].append(_box(65, 278, 475, 388, sx, sy, top, spacing))
    if family == "recharge":
        button_text, button_y = "Repeat recharge", 399.0
    else:
        button_text, button_y = "Pay again", 418.0
    if variant.show_pay_again or family == "recharge":
        btn_w = min(w * 0.46, 190 * scale)
        by = button_y * sy
        draw.rounded_rectangle((round(w / 2 - btn_w / 2), round(by), round(w / 2 + btn_w / 2), round(by + 54 * scale)),
                               radius=round(28 * scale), fill=ACCENT)
        _center(draw, by + 13 * scale, button_text, 20 * scale, (18, 49, 88))
    status_y = 511.0 if family != "recharge" else 485.0
    d = draw
    d.ellipse((round(w / 2 - 67 * sx), round(status_y * sy), round(w / 2 - 48 * sx), round((status_y + 19) * sy)), fill=GREEN)
    d.line((round(w / 2 - 62 * sx), round((status_y + 9) * sy), round(w / 2 - 58 * sx), round((status_y + 13) * sy),
            round(w / 2 - 52 * sx), round((status_y + 5) * sy)), fill=(15, 70, 30), width=max(1, round(2 * scale)))
    _text(d, (w / 2 - 39 * sx, (status_y - 3) * sy), "Completed", 20 * scale)
    div_y = (559 if family != "recharge" else 532) * sy
    d.line((round(w * .22), round(div_y), round(w * .78), round(div_y)), fill=(65, 65, 65), width=max(1, round(scale)))
    date_y = 581 if family != "recharge" else 553
    _center(d, date_y * sy, value["date_time"], 19 * scale, TEXT)
    regions["date_time"].append(_box(65, date_y - 5, 475, date_y + 35, sx, sy, top, spacing))
    return 650.0 if family != "recharge" else 625.0, sx, sy, scale


def _draw_bank_icon(draw: ImageDraw.ImageDraw, x: float, y: float, scale: float) -> None:
    draw.rounded_rectangle((round(x), round(y), round(x + 58 * scale), round(y + 39 * scale)),
                           radius=round(8 * scale), fill=(241, 248, 249))
    draw.ellipse((round(x + 8 * scale), round(y + 3 * scale), round(x + 50 * scale), round(y + 36 * scale)), fill=(34, 157, 190))
    draw.ellipse((round(x + 26 * scale), round(y + 8 * scale), round(x + 32 * scale), round(y + 14 * scale)), fill=(255, 255, 255))
    draw.rectangle((round(x + 28 * scale), round(y + 14 * scale), round(x + 30 * scale), round(y + 31 * scale)), fill=(255, 255, 255))


def _draw_footer_actions(draw: ImageDraw.ImageDraw, variant: RenderVariant, scale: float) -> None:
    if not variant.show_action_strip:
        return
    y = variant.height - round(205 * scale)
    labels = [("ⓘ", "Having issues?"), ("↗", "Share"), ("◎", "Split expense")]
    x = round(variant.width * .045)
    for icon, label in labels:
        width = round(variant.width * .29)
        draw.rounded_rectangle((x, y, x + width, y + round(45 * scale)), radius=round(12 * scale),
                               fill=(32, 32, 32), outline=(92, 92, 92), width=max(1, round(scale)))
        _text(draw, (x + round(8 * scale), y + round(11 * scale)), f"{icon} {label}", 12 * scale, TEXT)
        x += width + round(7 * scale)


def render(value: dict[str, Any], family: str, variant: RenderVariant) -> tuple[Image.Image, dict[str, list[tuple[int, int, int, int]]]]:
    if family not in SPEC["families"]:
        raise ValueError(f"Unsupported family: {family}")
    im, d, regions, scale, spacing = _base_canvas(variant, family)
    card_y, sx, sy, scale = _profile_and_summary(d, value, variant, regions, scale, spacing, family)
    w, h = variant.width, variant.height
    left = variant.margin * sx
    right = w - variant.margin * sx
    xtext = left + 22 * scale

    if family == "recharge":
        # Fictional operator/service identity and billing summary.
        top = 622 * sy
        _card(d, variant, 622, 882, sx, sy, scale)
        _text(d, (xtext, top + 18 * scale), "Plan price", 16 * scale, TEXT)
        _text(d, (right - 86 * scale, top + 18 * scale), f"₹{value['plan_price']:.2f}", 15 * scale, TEXT)
        regions["amount"].append(_box(380, 628, 510, 677, sx, sy, 68, spacing))
        _text(d, (xtext, top + 55 * scale), "Service fee", 16 * scale, TEXT)
        _text(d, (right - 86 * scale, top + 55 * scale), f"₹{value['service_fee']:.2f}", 15 * scale, TEXT)
        d.line((round(xtext), round(top + 91 * scale), round(right - 18 * scale), round(top + 91 * scale)), fill=(70, 70, 70))
        _text(d, (xtext, top + 108 * scale), "Total amount", 16 * scale, TEXT, True)
        total_text = f"₹{value['amount']:.2f}"
        f = _fit(total_text, 125 * scale, 15 * scale, True)
        d.text((round(right - f.getlength(total_text) - 18 * scale), round(top + 108 * scale)), total_text, font=f, fill=TEXT)
        regions["amount"].append(_box(380, 718, 510, 774, sx, sy, 68, spacing))
        progress_top = 900 * sy
        _card(d, variant, 900, 1355, sx, sy, scale)
        _text(d, (xtext, progress_top + 18 * scale), "Payment completed", 16 * scale, (126, 220, 152), True)
        for j, label in enumerate(("Payment started", "Service processed", "Recharge completed")):
            yy = progress_top + (62 + j * 40) * scale
            d.ellipse((round(xtext), round(yy + 4 * scale), round(xtext + 13 * scale), round(yy + 17 * scale)), fill=GREEN)
            _text(d, (xtext + 23 * scale, yy), label, 14 * scale, TEXT)
        _text(d, (xtext, progress_top + 203 * scale), "From: Sample User", 15 * scale, TEXT, True)
        _text(d, (xtext, progress_top + 232 * scale), "UPI transaction ID", 14 * scale, TEXT)
        _text(d, (xtext, progress_top + 256 * scale), value["upi_id"], 15 * scale, MUTED)
        transaction_y_base = 900 + 256 * scale / sy
        regions["transaction_id"].append(_box(45, transaction_y_base - 48, 495, transaction_y_base + 70, sx, sy, 68, spacing))
        _text(d, (xtext, progress_top + 302 * scale), "Google transaction ID", 14 * scale, TEXT)
        _text(d, (xtext, progress_top + 326 * scale), value["google_id"], 15 * scale, MUTED)
    else:
        card_start = 650 * sy
        card_bottom = (1215 if family in ("enriched", "bank_progress") else 1142) * sy
        d.rounded_rectangle((round(left), round(card_start), round(right), round(card_bottom)), radius=round(22 * scale),
                            fill=CARD, outline=(70, 70, 70), width=max(1, round(scale)))
        _draw_bank_icon(d, xtext, card_start + 18 * scale, scale)
        bank = value["bank"]
        bank_f = _fit(f"{bank} ····{value['account_last']}", right - xtext - 91 * scale, 20 * scale, True)
        _text(d, (xtext + 72 * scale, card_start + 23 * scale), f"{bank} ····{value['account_last']}", bank_f.size, TEXT, True)
        header_x_base = (xtext + 68 * scale) / sx
        header_y_base = 650 + 8 * scale / sy
        regions["other"].append(_box(header_x_base, header_y_base, (right - 5 * scale) / sx,
                                     650 + 60 * scale / sy, sx, sy, 68, spacing))
        header_bottom = card_start + 74 * scale
        d.line((round(left), round(header_bottom), round(right), round(header_bottom)), fill=(70, 70, 70))

        if family == "enriched":
            msg_y = header_bottom + 17 * scale
            _text(d, (xtext, msg_y), "Payment completed", 17 * scale, TEXT, True)
            _text(d, (xtext, msg_y + 28 * scale), "The recipient account confirmed this", 14 * scale, MUTED)
            _text(d, (xtext, msg_y + 49 * scale), "fictional demo payment.", 14 * scale, MUTED)
            id_y = header_bottom + 93 * scale
        elif family == "bank_progress":
            timeline_y = header_bottom + 18 * scale
            events = ["Payment started", "Amount debited", "Recipient credited", "Payment completed"]
            for j, label in enumerate(events):
                yy = timeline_y + j * 37 * scale
                d.ellipse((round(xtext), round(yy + 4 * scale), round(xtext + 13 * scale), round(yy + 17 * scale)), fill=GREEN if j == 3 else BLUE)
                if j < len(events) - 1:
                    d.line((round(xtext + 6 * scale), round(yy + 18 * scale), round(xtext + 6 * scale), round(yy + 39 * scale)), fill=(95, 126, 156), width=max(1, round(2 * scale)))
                _text(d, (xtext + 24 * scale, yy), label, 14 * scale, TEXT if j < 3 else (126, 220, 152), j == 3)
            d.line((round(left), round(timeline_y + 4 * 37 * scale), round(right), round(timeline_y + 4 * 37 * scale)), fill=(70, 70, 70))
            id_y = timeline_y + 4 * 37 * scale + 15 * scale
        else:
            id_y = header_bottom + 15 * scale

        _text(d, (xtext, id_y), "UPI transaction ID", 15 * scale, TEXT)
        _text(d, (xtext, id_y + 24 * scale), value["upi_id"], 16 * scale, MUTED)
        regions["transaction_id"].append(_box(48, id_y / sy + 18, 492, id_y / sy + 52, sx, sy, 68, spacing))
        row_y = id_y + 69 * scale
        name_line = f"To: {value['recipient']}"
        _text(d, (xtext, row_y), name_line, 15 * scale, TEXT, True)
        f = _font(_fit(name_line, right - xtext - 15 * scale, 15 * scale, True).size, True)
        regions["recipient"].append(_box(48, row_y / sy - 5, min(492, 48 + f.getlength(name_line) / sx + 15), row_y / sy + 28, sx, sy, 68, spacing))
        upi_y = row_y + 29 * scale
        _text(d, (xtext, upi_y), value["recipient_upi"], 14 * scale, MUTED)
        sender_y = upi_y + 48 * scale
        _text(d, (xtext, sender_y), f"From: {value['sender']}", 14 * scale, TEXT, True)
        _text(d, (xtext, sender_y + 26 * scale), "••••@demo · Google Pay", 14 * scale, MUTED)
        google_y = sender_y + 68 * scale
        _text(d, (xtext, google_y), "Google transaction ID", 15 * scale, TEXT)
        _text(d, (xtext, google_y + 24 * scale), value["google_id"], 16 * scale, MUTED)

        if family == "enriched" and variant.context_chip:
            chip = value["context"]
            f = _font(13 * scale)
            chip_w = f.getlength(chip) + 28 * scale
            y = 382 * sy
            d.rounded_rectangle((round(w / 2 - chip_w / 2), round(y), round(w / 2 + chip_w / 2), round(y + 31 * scale)),
                                radius=round(15 * scale), fill=(38, 38, 38))
            d.text((round(w / 2 - f.getlength(chip) / 2), round(y + 7 * scale)), chip, font=f, fill=MUTED)
        if family == "bank_progress":
            _draw_footer_actions(d, variant, scale)
    target_h = max(1, round(h * spacing))
    if target_h != h:
        scaled = im.resize((w, target_h), Image.Resampling.BICUBIC)
        final = Image.new("RGB", (w, h), BG)
        final.paste(scaled.crop((0, 0, w, min(h, target_h))), (0, 0))
        im = final
        for field_boxes in regions.values():
            for i, (x1, y1, x2, y2) in enumerate(field_boxes):
                field_boxes[i] = (x1, min(h, round(y1 * spacing)), x2, min(h, round(y2 * spacing)))
    final_draw = ImageDraw.Draw(im)
    _draw_gpay_footer(final_draw, w, h, scale)
    marker_y = h - round((135 if family == "bank_progress" and variant.show_action_strip else 95) * scale)
    _center(final_draw, marker_y, "SYNTHETIC / DEMO", 13 * scale, (255, 210, 115), True)
    # Include antialiased glyph edge pixels at unusual screenshot scales.
    for field_boxes in regions.values():
        for i, (x1, y1, x2, y2) in enumerate(field_boxes):
            field_boxes[i] = (max(0, x1 - 48), max(0, y1 - 16), min(w, x2 + 48), min(h, y2 + 16))
    return im, regions
