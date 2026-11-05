"""
Phase 3: Tile Extraction Pipeline (256x256 tiles, 128 stride).
Member 2 - Tile Forensics CNN & Heatmap.
"""

from .tile_extractor import (
    DEFAULT_PAD_COLOR,
    ExtractedTile,
    LazyTileDataset,
    TileConfig,
    TileExtractor,
    TileMetadata,
    compute_tile_coordinates,
    pad_image_if_needed,
    pad_mask_if_needed,
)

__all__ = [
    "TileConfig",
    "TileMetadata",
    "ExtractedTile",
    "TileExtractor",
    "LazyTileDataset",
    "DEFAULT_PAD_COLOR",
    "compute_tile_coordinates",
    "pad_image_if_needed",
    "pad_mask_if_needed",
]

