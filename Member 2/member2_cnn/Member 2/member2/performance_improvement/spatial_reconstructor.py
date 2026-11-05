"""
CBL-04: Spatial Tile Heatmap Reconstruction Engine
Phase 6 Improvement - Member 2 (Tile Forensics CNN & Heatmap Lead)

This module accumulates 2D spatial tile probability maps into a continuous
full-resolution tampering probability heatmap:
- Inputs: Batched (1, 256, 256) predicted spatial tile probabilities.
- Weighted overlap blending: 2D Gaussian window weighting.
- Accumulates pixel-by-pixel probabilities across the full screenshot canvas.
- De-biases by the accumulated spatial weight map to guarantee smooth boundaries.
- Preserves exact source image resolution (H, W).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class SpatialHeatmapReconstructor:
    """
    Reconstructs continuous full-resolution heatmaps from spatial (256, 256) tile prediction maps.
    """

    def __init__(
        self,
        tile_size: int = 256,
        window_type: str = "gaussian",
        gaussian_sigma: float = 64.0,
        eps: float = 1e-7,
    ) -> None:
        self.tile_size = tile_size
        self.eps = eps

        # 2D Gaussian window
        center = (tile_size - 1) / 2.0
        coords = np.arange(tile_size, dtype=np.float32)
        grid_y, grid_x = np.meshgrid(coords, coords, indexing="ij")
        dist_sq = (grid_x - center) ** 2 + (grid_y - center) ** 2
        self.window = np.exp(-dist_sq / (2.0 * (gaussian_sigma ** 2))).astype(np.float32)

    def reconstruct(
        self,
        height: int,
        width: int,
        tile_predictions: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Blend 2D spatial tile prediction arrays into an (height, width) continuous heatmap.

        Args:
            height: Image canvas height.
            width: Image canvas width.
            tile_predictions: List of dicts, each containing:
                - 'tile_x': int
                - 'tile_y': int
                - 'spatial_prob_map': (256, 256) float32 array in [0, 1]

        Returns:
            heatmap: (height, width) float32 array in [0.0, 1.0]
        """
        prob_accumulator = np.zeros((height, width), dtype=np.float32)
        weight_accumulator = np.zeros((height, width), dtype=np.float32)

        ts = self.tile_size
        window = self.window

        for pred in tile_predictions:
            tx = int(pred["tile_x"])
            ty = int(pred["tile_y"])
            tile_prob_map = pred["spatial_prob_map"]  # (256, 256)

            x_end = min(tx + ts, width)
            y_end = min(ty + ts, height)

            w_crop_w = x_end - tx
            w_crop_h = y_end - ty

            if w_crop_w <= 0 or w_crop_h <= 0:
                continue

            sub_map = tile_prob_map[:w_crop_h, :w_crop_w]
            sub_window = window[:w_crop_h, :w_crop_w]

            prob_accumulator[ty:y_end, tx:x_end] += sub_map * sub_window
            weight_accumulator[ty:y_end, tx:x_end] += sub_window

        mask_valid = weight_accumulator > self.eps
        heatmap = np.zeros((height, width), dtype=np.float32)
        heatmap[mask_valid] = prob_accumulator[mask_valid] / weight_accumulator[mask_valid]

        heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(heatmap, 0.0, 1.0).astype(np.float32)

