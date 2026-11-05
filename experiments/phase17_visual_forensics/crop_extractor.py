"""
Phase 17 High-Resolution Crop Extraction Module.
Extracts native-resolution ROI crops around key payment fields (Amount, Recipient, Transaction ID).
Preserves native aspect ratio with symmetric #121212 canvas padding.
Records precise pixel coordinates, scaling factors, and crop provenance.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger("phase17.crop_extractor")

BACKGROUND_RGB = (18, 18, 18)


@dataclass
class CropMetadata:
    sample_id: str
    pair_id: str
    source_split: str
    label: str
    edit_category: str
    layout_family: str
    field_name: str
    extraction_source: str  # "ground_truth" or "ocr"
    orig_img_width: int
    orig_img_height: int
    raw_bbox_xyxy: List[int]
    padded_bbox_xyxy: List[int]
    native_crop_width: int
    native_crop_height: int
    target_canvas_size: List[int]  # [width, height]
    padding_pixels: List[int]  # [pad_left, pad_top, pad_right, pad_bottom]
    crop_saved_path: str
    is_missing: bool = False


class NativeCropExtractor:
    """
    Extracts native-resolution field crops with context padding while preserving aspect ratio.
    """

    DEFAULT_CANVAS_SIZES = {
        "amount": (256, 128),
        "receiver": (384, 128),
        "transaction_id": (384, 128),
    }

    CONTEXT_MARGINS = {
        "amount": (0.15, 0.25),         # 15% width margin, 25% height margin
        "receiver": (0.10, 0.20),       # 10% width margin, 20% height margin
        "transaction_id": (0.10, 0.25), # 10% width margin, 25% height margin
    }

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_field_crop(
        self,
        image: Image.Image,
        field_name: str,
        bbox_xyxy: Optional[Tuple[int, int, int, int]],
        sample_id: str,
        pair_id: str,
        source_split: str,
        label: str,
        edit_category: str,
        layout_family: str,
        extraction_source: str = "ocr",
    ) -> Tuple[Image.Image, CropMetadata]:
        """
        Extracts native crop, applies symmetric dark padding to reach fixed canvas size,
        and records provenance metadata.
        """
        w_orig, h_orig = image.size
        target_w, target_h = self.DEFAULT_CANVAS_SIZES.get(field_name, (256, 128))

        if bbox_xyxy is None:
            # Handle missing crop gracefully with a blank background canvas
            blank_img = Image.new("RGB", (target_w, target_h), BACKGROUND_RGB)
            crop_path = self.output_dir / f"{sample_id}_{field_name}.png"
            blank_img.save(crop_path)
            meta = CropMetadata(
                sample_id=sample_id,
                pair_id=pair_id,
                source_split=source_split,
                label=label,
                edit_category=edit_category,
                layout_family=layout_family,
                field_name=field_name,
                extraction_source=extraction_source,
                orig_img_width=w_orig,
                orig_img_height=h_orig,
                raw_bbox_xyxy=[0, 0, 0, 0],
                padded_bbox_xyxy=[0, 0, 0, 0],
                native_crop_width=0,
                native_crop_height=0,
                target_canvas_size=[target_w, target_h],
                padding_pixels=[0, 0, 0, 0],
                crop_saved_path=str(crop_path),
                is_missing=True,
            )
            return blank_img, meta

        x0, y0, x1, y1 = bbox_xyxy
        # Ensure coordinates are within image bounds
        x0 = max(0, min(w_orig - 1, x0))
        y0 = max(0, min(h_orig - 1, y0))
        x1 = max(x0 + 1, min(w_orig, x1))
        y1 = max(y0 + 1, min(h_orig, y1))

        bw = x1 - x0
        bh = y1 - y0

        # Add context margin around target text
        wm, hm = self.CONTEXT_MARGINS.get(field_name, (0.15, 0.20))
        pad_x = int(round(bw * wm))
        pad_y = int(round(bh * hm))

        cx0 = max(0, x0 - pad_x)
        cy0 = max(0, y0 - pad_y)
        cx1 = min(w_orig, x1 + pad_x)
        cy1 = min(h_orig, y1 + pad_y)

        # Crop native-resolution patch
        native_patch = image.crop((cx0, cy0, cx1, cy1))
        pw, ph = native_patch.size

        # Fit into target canvas preserving aspect ratio
        scale = min(target_w / pw, target_h / ph)
        new_w = max(1, int(round(pw * scale)))
        new_h = max(1, int(round(ph * scale)))

        # Resample with high-quality Lanczos only when scaling
        resized_patch = native_patch.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Create target canvas and paste centered
        canvas = Image.new("RGB", (target_w, target_h), BACKGROUND_RGB)
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        canvas.paste(resized_patch, (offset_x, offset_y))

        # Save crop image
        crop_path = self.output_dir / f"{sample_id}_{field_name}.png"
        canvas.save(crop_path)

        meta = CropMetadata(
            sample_id=sample_id,
            pair_id=pair_id,
            source_split=source_split,
            label=label,
            edit_category=edit_category,
            layout_family=layout_family,
            field_name=field_name,
            extraction_source=extraction_source,
            orig_img_width=w_orig,
            orig_img_height=h_orig,
            raw_bbox_xyxy=[int(x0), int(y0), int(x1), int(y1)],
            padded_bbox_xyxy=[int(cx0), int(cy0), int(cx1), int(cy1)],
            native_crop_width=pw,
            native_crop_height=ph,
            target_canvas_size=[target_w, target_h],
            padding_pixels=[offset_x, offset_y, target_w - new_w - offset_x, target_h - new_h - offset_y],
            crop_saved_path=str(crop_path),
            is_missing=False,
        )

        return canvas, meta

