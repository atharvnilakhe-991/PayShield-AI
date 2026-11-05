"""
CBL-04: Full-Resolution Heatmap Reconstruction & Overlap Blending
Phase 6 - Member 2 (Tile Forensics CNN & Heatmap Lead)

This module reconstructs continuous full-resolution tampering probability heatmaps
from tile-level model predictions using weighted spatial window blending:
- Reconstructs exact native screenshot dimensions (H, W).
- Uses Phase 3 tile coordinates (x, y) with boundary-anchored coverage.
- Weighted overlap blending: Gaussian, Hann (raised cosine), or Uniform windows.
- Accumulates probability and weight maps to eliminate tile boundary seams.
- Normalizes by accumulated weights and strictly prevents NaN / Inf values.
- Guarantees output is float32 constrained to [0.0, 1.0].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def generate_blending_window(
    tile_size: int = 256,
    window_type: str = "gaussian",
    sigma: float = 64.0,
    min_weight: float = 0.05,
) -> np.ndarray:
    """
    Generate a 2D spatial weighting window of shape (tile_size, tile_size).

    Supported window types:
    - 'gaussian': Isotropic 2D Gaussian centered at (tile_size/2, tile_size/2).
    - 'hann': 2D Hann window (raised cosine) with a non-zero floor.
    - 'uniform': Constant 1.0 everywhere.
    """
    w_type = window_type.lower().strip()

    if w_type == "gaussian":
        center = (tile_size - 1) / 2.0
        coords = np.arange(tile_size, dtype=np.float32)
        grid_y, grid_x = np.meshgrid(coords, coords, indexing="ij")
        dist_sq = (grid_x - center) ** 2 + (grid_y - center) ** 2
        window = np.exp(-dist_sq / (2.0 * (sigma ** 2)))
        return window.astype(np.float32)

    elif w_type == "hann":
        n = tile_size
        hann_1d = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))
        window = np.outer(hann_1d, hann_1d)
        window = np.maximum(window, min_weight)
        return window.astype(np.float32)

    elif w_type in ("uniform", "flat", "box"):
        return np.ones((tile_size, tile_size), dtype=np.float32)

    else:
        raise ValueError(
            f"Unsupported window_type '{window_type}'. Choose 'gaussian', 'hann', or 'uniform'."
        )


class HeatmapReconstructor:
    """
    Reconstructs continuous full-resolution tampering probability heatmaps
    from tile predictions.
    """

    def __init__(
        self,
        tile_size: int = 256,
        window_type: str = "gaussian",
        gaussian_sigma: float = 64.0,
        hann_min_weight: float = 0.05,
        eps: float = 1e-7,
    ) -> None:
        self.tile_size = tile_size
        self.window_type = window_type
        self.gaussian_sigma = gaussian_sigma
        self.hann_min_weight = hann_min_weight
        self.eps = eps

        # Precompute canonical 2D window
        self.window = generate_blending_window(
            tile_size=tile_size,
            window_type=window_type,
            sigma=gaussian_sigma,
            min_weight=hann_min_weight,
        )

    def reconstruct(
        self,
        height: int,
        width: int,
        tile_predictions: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Blend tile predictions into a full-resolution probability heatmap of shape (height, width).

        Args:
            height: Exact target image height in pixels.
            width: Exact target image width in pixels.
            tile_predictions: List of dicts containing:
                - 'tile_x': int
                - 'tile_y': int
                - 'tampered_prob' or 'probability': float in [0, 1]
                - optional 'tile_width', 'tile_height'

        Returns:
            heatmap: (height, width) float32 array in [0.0, 1.0]
        """
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid dimensions: height={height}, width={width}")

        prob_accumulator = np.zeros((height, width), dtype=np.float32)
        weight_accumulator = np.zeros((height, width), dtype=np.float32)

        ts = self.tile_size
        window = self.window

        for pred in tile_predictions:
            tx = int(pred["tile_x"])
            ty = int(pred["tile_y"])
            prob = float(pred.get("tampered_prob", pred.get("probability", 0.0)))

            # Handle boundary anchoring or canvas edge clipping
            x_end = min(tx + ts, width)
            y_end = min(ty + ts, height)

            w_crop_w = x_end - tx
            w_crop_h = y_end - ty

            if w_crop_w <= 0 or w_crop_h <= 0:
                continue

            # Sub-window if tile exceeds canvas
            sub_window = window[:w_crop_h, :w_crop_w]

            prob_accumulator[ty:y_end, tx:x_end] += prob * sub_window
            weight_accumulator[ty:y_end, tx:x_end] += sub_window

        # Safe division to prevent division-by-zero or NaNs
        mask_valid = weight_accumulator > self.eps

        heatmap = np.zeros((height, width), dtype=np.float32)
        heatmap[mask_valid] = prob_accumulator[mask_valid] / weight_accumulator[mask_valid]

        # For any uncovered pixels (though Phase 3 edge-anchoring covers 100%), fallback to 0.0
        # Sanitize against NaN / Inf
        heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)
        heatmap = np.clip(heatmap, 0.0, 1.0).astype(np.float32)

        return heatmap

    @staticmethod
    def threshold_heatmap(
        heatmap: np.ndarray,
        threshold: float,
        binary_format: str = "0_1",
    ) -> np.ndarray:
        """
        Threshold continuous probability heatmap into binary predicted mask.

        Args:
            heatmap: (H, W) float array in [0, 1]
            threshold: Decision boundary in [0, 1]
            binary_format: '0_1' for {0, 1} uint8 or '0_255' for {0, 255} uint8

        Returns:
            binary_mask: (H, W) uint8 array
        """
        fg_val = 1 if binary_format == "0_1" else 255
        binary_mask = np.zeros(heatmap.shape, dtype=np.uint8)
        binary_mask[heatmap >= threshold] = fg_val
        return binary_mask

    def compute_coverage_map(
        self,
        height: int,
        width: int,
        tile_coordinates: List[Tuple[int, int]],
    ) -> np.ndarray:
        """
        Compute an integer spatial coverage map counting how many tiles overlap each pixel.
        """
        coverage = np.zeros((height, width), dtype=np.int32)
        ts = self.tile_size

        for tx, ty in tile_coordinates:
            x_end = min(tx + ts, width)
            y_end = min(ty + ts, height)
            coverage[ty:y_end, tx:x_end] += 1

        return coverage

