"""
CBL-04: Spatial Post-Processing & Connected-Component Filtering
Phase 6 Improvement - Member 2 (Tile Forensics CNN & Heatmap Lead)

This module implements controlled post-processing for reconstructed heatmaps:
- Decision thresholding into candidate binary masks.
- Connected-component area thresholding: eliminates isolated noise specks (< min_area pixels)
  while strictly preserving genuine character-level and digit alterations.
- Morphological cleaning: optional structuring elements for edge smoothing.
- All hyperparameters are tuned strictly on the validation set.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass
class PostProcessingConfig:
    threshold: float = 0.50
    min_component_area: int = 16  # Minimum pixel count for a connected region
    apply_opening: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HeatmapPostProcessor:
    """
    Applies spatial connected-component filtering to eliminate scattered false-positive noise.
    """

    def __init__(self, config: Optional[PostProcessingConfig] = None) -> None:
        self.config = config or PostProcessingConfig()

    def process(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Convert continuous probability heatmap into cleaned binary mask.

        Args:
            heatmap: (H, W) float32 probability array in [0, 1]

        Returns:
            cleaned_mask: (H, W) uint8 binary mask in {0, 1}
        """
        th = self.config.threshold
        min_area = self.config.min_component_area

        # 1. Decision thresholding
        binary = (heatmap >= th).astype(np.uint8)

        if min_area <= 0 or not np.any(binary):
            return binary

        # 2. Connected component labeling (8-connectivity)
        structure = ndimage.generate_binary_structure(2, 2)
        labeled, num_features = ndimage.label(binary, structure=structure)

        if num_features == 0:
            return binary

        # Count sizes of each component
        component_sizes = np.bincount(labeled.ravel())

        # Identify components meeting the area threshold (ignoring background component 0)
        valid_components = np.zeros(num_features + 1, dtype=bool)
        for comp_id in range(1, num_features + 1):
            if component_sizes[comp_id] >= min_area:
                valid_components[comp_id] = True

        cleaned_mask = valid_components[labeled].astype(np.uint8)

        # 3. Optional morphological opening
        if self.config.apply_opening:
            cleaned_mask = ndimage.binary_opening(cleaned_mask, structure=np.ones((2, 2))).astype(np.uint8)

        return cleaned_mask

