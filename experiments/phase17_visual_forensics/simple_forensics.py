"""
Phase 17 Simple Forensic Feature Extraction Module.
Computes deterministic visual forensic statistics on native text-region crops:
- High-pass Laplacian residual energy
- Sobel edge gradient statistics
- Background margin noise variance
- Error Level Analysis (ELA) simulation at JPEG 90
- Pixel intensity distributions
"""

from __future__ import annotations

import io
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageFilter


def compute_crop_forensic_features(crop_img: Image.Image) -> Dict[str, float]:
    """
    Computes visual forensic and statistical features on an extracted native crop.
    """
    w, h = crop_img.size
    if w == 0 or h == 0:
        return {
            "laplacian_energy": 0.0,
            "laplacian_std": 0.0,
            "edge_mean": 0.0,
            "edge_std": 0.0,
            "bg_noise_std": 0.0,
            "ela_mean": 0.0,
            "ela_max": 0.0,
            "mean_intensity": 0.0,
        }

    # Convert to grayscale numpy float array [0, 255]
    gray = crop_img.convert("L")
    arr = np.array(gray, dtype=np.float32)

    # 1. High-Pass Laplacian Residual
    # Kernel: [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    padded = np.pad(arr, 1, mode="edge")
    lap = (
        4.0 * arr
        - padded[:-2, 1:-1]
        - padded[2:, 1:-1]
        - padded[1:-1, :-2]
        - padded[1:-1, 2:]
    )
    lap_energy = float(np.mean(lap**2))
    lap_std = float(np.std(lap))

    # 2. Sobel Edge Gradient Magnitude
    gx = (
        padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:]
        - padded[:-2, :-2] - 2 * padded[1:-1, :-2] - padded[2:, :-2]
    ) / 8.0
    gy = (
        padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]
        - padded[:-2, :-2] - 2 * padded[:-2, 1:-1] - padded[:-2, 2:]
    ) / 8.0
    grad_mag = np.sqrt(gx**2 + gy**2)
    edge_mean = float(np.mean(grad_mag))
    edge_std = float(np.std(grad_mag))

    # 3. Background Margin Noise (pixels with intensity < 35, dark theme)
    bg_mask = arr < 35.0
    if np.sum(bg_mask) > 20:
        bg_noise_std = float(np.std(arr[bg_mask]))
    else:
        bg_noise_std = 0.0

    # 4. Error Level Analysis (ELA) at JPEG quality 90
    buf = io.BytesIO()
    crop_img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    resaved = Image.open(buf).convert("L")
    resaved_arr = np.array(resaved, dtype=np.float32)
    ela_diff = np.abs(arr - resaved_arr)
    ela_mean = float(np.mean(ela_diff))
    ela_max = float(np.max(ela_diff))

    # 5. Overall Intensity
    mean_intensity = float(np.mean(arr))

    return {
        "laplacian_energy": round(lap_energy, 4),
        "laplacian_std": round(lap_std, 4),
        "edge_mean": round(edge_mean, 4),
        "edge_std": round(edge_std, 4),
        "bg_noise_std": round(bg_noise_std, 4),
        "ela_mean": round(ela_mean, 4),
        "ela_max": round(ela_max, 4),
        "mean_intensity": round(mean_intensity, 4),
    }

