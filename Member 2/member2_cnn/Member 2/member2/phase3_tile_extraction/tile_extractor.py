"""
CBL-04: Native-Resolution Tile Extraction Engine
Phase 3 - Member 2 (Tile Forensics CNN & Heatmap Lead)

This module extracts fixed-size 256x256 tiles from native-resolution Google Pay
dark-theme screenshots with a 128-pixel stride in both horizontal and vertical directions.

Key Architecture & Capabilities:
- Tile geometry: 256x256 pixels, 128-pixel stride (50% spatial overlap).
- Edge preservation: Anchors final tiles to right and bottom canvas boundaries
  ((W - 256), (H - 256)), ensuring zero dropped pixels or cut-off transaction elements.
- RGB and resolution preservation: Never resizes, downsamples, or compresses source screenshots.
- Metadata traceability: Every tile stores complete spatial coordinates, source IDs,
  pair IDs, canvas dimensions, and leakage prevention split_group_id.
- Leakage prevention: Clean and edited screenshots from the same transaction pair share
  identical split_group_id.
- Memory & computational efficiency: Lazy on-demand extraction via generators and indexable
  LazyTileDataset. Avoids materializing tens of thousands of tiles on disk.
- Deterministic: Coordinate calculations and tile order are strictly deterministic.
- Scope boundary: Does NOT assign train/val/test splits, does NOT compute mask-based
  tile labels (Phase 4), and does NOT train CNN models (Phase 5).
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

# Setup sys.path so imports resolve both in package context and standalone script execution
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
PHASE1_DIR = HERE.parent / "phase1_dataset_loader"
PHASE2_DIR = HERE.parent / "phase2_preprocessing"

for p in [PROJECT_ROOT, HERE, PHASE1_DIR, PHASE2_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from member2.phase1_dataset_loader.dataset_loader import DatasetLoader, PairSample, SingleSample
    from member2.phase2_preprocessing.transforms import (
        DEFAULT_ALPHA_BG_COLOR,
        PreprocessConfig,
        Preprocessor,
        ProcessedPair,
        ProcessedSample,
    )
except (ImportError, ModuleNotFoundError):
    from dataset_loader import DatasetLoader, PairSample, SingleSample  # type: ignore[import-not-found]
    from transforms import (  # type: ignore[import-not-found]
        DEFAULT_ALPHA_BG_COLOR,
        PreprocessConfig,
        Preprocessor,
        ProcessedPair,
        ProcessedSample,
    )

logger = logging.getLogger(__name__)

# Standard Google Pay dark theme background (#111111)
DEFAULT_PAD_COLOR = (17, 17, 17)


@dataclass(frozen=True)
class TileConfig:
    """Configuration parameters for 256x256 tile extraction."""

    tile_size: int = 256
    stride: int = 128
    pad_color: Tuple[int, int, int] = DEFAULT_PAD_COLOR
    mask_pad_val: int = 0
    extract_mask: bool = True

    def __post_init__(self) -> None:
        if self.tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {self.tile_size}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        if self.stride > self.tile_size:
            logger.warning(
                f"Stride ({self.stride}) is greater than tile_size ({self.tile_size}), "
                "which will cause spatial gaps between adjacent tiles."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TileMetadata:
    """Structured spatial and provenance metadata for an individual tile."""

    tile_id: str
    source_id: str
    pair_id: Optional[str]
    image_path: Optional[str]
    tile_x: int
    tile_y: int
    tile_width: int
    tile_height: int
    tile_index: int
    original_width: int
    original_height: int
    split_group_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedTile:
    """Represents a fully extracted 256x256 RGB image tile with optional mask."""

    metadata: TileMetadata
    image: np.ndarray  # Shape: (256, 256, 3), dtype: uint8 (or float32)
    mask: Optional[np.ndarray] = None  # Shape: (256, 256), dtype: uint8 or None

    def __post_init__(self) -> None:
        expected_shape = (self.metadata.tile_height, self.metadata.tile_width, 3)
        if self.image.shape != expected_shape:
            raise ValueError(
                f"Tile image shape mismatch for '{self.metadata.tile_id}': "
                f"expected {expected_shape}, got {self.image.shape}"
            )
        if self.mask is not None:
            expected_mask_shape = (self.metadata.tile_height, self.metadata.tile_width)
            if self.mask.shape != expected_mask_shape:
                raise ValueError(
                    f"Tile mask shape mismatch for '{self.metadata.tile_id}': "
                    f"expected {expected_mask_shape}, got {self.mask.shape}"
                )

    @property
    def tile_id(self) -> str:
        return self.metadata.tile_id

    @property
    def source_id(self) -> str:
        return self.metadata.source_id

    @property
    def pair_id(self) -> Optional[str]:
        return self.metadata.pair_id

    @property
    def split_group_id(self) -> str:
        return self.metadata.split_group_id

    @property
    def tile_x(self) -> int:
        return self.metadata.tile_x

    @property
    def tile_y(self) -> int:
        return self.metadata.tile_y

    @property
    def tile_index(self) -> int:
        return self.metadata.tile_index

    @property
    def has_mask(self) -> bool:
        return self.mask is not None

    def to_pil_image(self) -> Image.Image:
        """Convert tile image ndarray to PIL Image in RGB mode."""
        if self.image.dtype != np.uint8:
            uint8_img = np.clip(self.image * 255.0, 0, 255).astype(np.uint8)
        else:
            uint8_img = self.image
        return Image.fromarray(uint8_img, mode="RGB")

    def to_pil_mask(self) -> Optional[Image.Image]:
        """Convert tile mask ndarray to PIL Image in L mode with {0, 255}."""
        if self.mask is None:
            return None
        display_mask = (self.mask * 255).astype(np.uint8) if np.max(self.mask) <= 1 else self.mask
        return Image.fromarray(display_mask, mode="L")

    def to_dict(self) -> Dict[str, Any]:
        d = self.metadata.to_dict()
        d["has_mask"] = self.has_mask
        d["image_shape"] = list(self.image.shape)
        d["mask_shape"] = list(self.mask.shape) if self.mask is not None else None
        return d


def compute_tile_coordinates(
    length: int,
    tile_size: int = 256,
    stride: int = 128,
) -> List[int]:
    """
    Compute 1D tile start coordinates along an axis of given length.

    Ensures complete coverage of all pixels [0, length - 1]:
    - If length <= tile_size: returns [0] (image will be padded if length < tile_size).
    - If length > tile_size: slides standard window by `stride`.
    - If the final standard window does not reach the exact boundary, anchors
      an additional tile at (length - tile_size) so right/bottom edge pixels are never lost.

    Returns:
        Sorted list of integer start coordinates.
    """
    if length <= tile_size:
        return [0]

    coords: List[int] = []
    pos = 0
    while pos + tile_size <= length:
        coords.append(pos)
        pos += stride

    # Check if the right/bottom edge was fully covered
    last_covered = coords[-1] + tile_size
    if last_covered < length:
        edge_coord = length - tile_size
        if edge_coord not in coords:
            coords.append(edge_coord)

    return sorted(coords)


def pad_image_if_needed(
    image_arr: np.ndarray,
    target_w: int,
    target_h: int,
    pad_color: Tuple[int, int, int] = DEFAULT_PAD_COLOR,
) -> np.ndarray:
    """Pad image array if dimensions are smaller than target tile size."""
    h, w, c = image_arr.shape
    pad_bottom = max(0, target_h - h)
    pad_right = max(0, target_w - w)

    if pad_bottom == 0 and pad_right == 0:
        return image_arr

    padded = np.full((h + pad_bottom, w + pad_right, c), pad_color, dtype=image_arr.dtype)
    padded[:h, :w, :] = image_arr
    return padded


def pad_mask_if_needed(
    mask_arr: Optional[np.ndarray],
    target_w: int,
    target_h: int,
    mask_pad_val: int = 0,
) -> Optional[np.ndarray]:
    """Pad mask array if dimensions are smaller than target tile size."""
    if mask_arr is None:
        return None

    h, w = mask_arr.shape
    pad_bottom = max(0, target_h - h)
    pad_right = max(0, target_w - w)

    if pad_bottom == 0 and pad_right == 0:
        return mask_arr

    padded = np.full((h + pad_bottom, w + pad_right), mask_pad_val, dtype=mask_arr.dtype)
    padded[:h, :w] = mask_arr
    return padded


class TileExtractor:
    """
    Engine for extracting 256x256 tiles with 128 stride from native-resolution screenshots.

    Usage:
        extractor = TileExtractor()
        metadata_list = extractor.get_tile_metadata_list(sample)
        tiles = extractor.extract_all_tiles_from_sample(sample)
    """

    def __init__(
        self,
        config: Optional[TileConfig] = None,
        preprocessor: Optional[Preprocessor] = None,
    ) -> None:
        self.config = config or TileConfig()
        self.preprocessor = preprocessor or Preprocessor(PreprocessConfig(preserve_native_resolution=True))

    def compute_grid_coordinates(self, width: int, height: int) -> Tuple[List[int], List[int]]:
        """Compute horizontal and vertical start coordinates covering canvas."""
        x_coords = compute_tile_coordinates(width, tile_size=self.config.tile_size, stride=self.config.stride)
        y_coords = compute_tile_coordinates(height, tile_size=self.config.tile_size, stride=self.config.stride)
        return x_coords, y_coords

    def _determine_split_group_id(self, sample: Any) -> str:
        """
        Derive strict split_group_id to prevent cross-split data leakage.

        Pair invariant: Clean and edited screenshots from the same transaction pair
        must receive the IDENTICAL split_group_id.
        """
        pair_id = getattr(sample, "pair_id", None)
        if pair_id and str(pair_id).strip() not in ("", "None", "nan"):
            return str(pair_id).strip()

        source_id = getattr(sample, "source_id", None)
        if source_id and str(source_id).strip() not in ("", "None", "nan"):
            return str(source_id).strip()

        image_id = getattr(sample, "image_id", "sample")
        return str(image_id)

    def _ensure_processed_sample(self, sample: Any) -> ProcessedSample:
        """Ensure input is a ProcessedSample from Phase 2."""
        if isinstance(sample, ProcessedSample):
            return sample
        return self.preprocessor.process_sample(sample)

    def get_tile_metadata_list(self, sample: Any) -> List[TileMetadata]:
        """
        Compute lightweight metadata for all tiles in sample without extracting pixel arrays.

        Allows fast index mapping, counts, and leakage verification without RAM overhead.
        """
        image_id = getattr(sample, "image_id", "sample")
        source_id = getattr(sample, "source_id", "source")
        pair_id = getattr(sample, "pair_id", None)
        image_path = getattr(sample, "image_path", None)
        split_group_id = self._determine_split_group_id(sample)

        # Obtain native canvas dimensions
        orig_w = getattr(sample, "width", None)
        orig_h = getattr(sample, "height", None)

        if orig_w is None or orig_h is None:
            if hasattr(sample, "processed_size"):
                orig_w, orig_h = sample.processed_size
            elif hasattr(sample, "original_size"):
                orig_w, orig_h = sample.original_size
            else:
                proc = self._ensure_processed_sample(sample)
                orig_w, orig_h = proc.processed_size

        x_coords, y_coords = self.compute_grid_coordinates(orig_w, orig_h)

        metadata_list: List[TileMetadata] = []
        tile_index = 0

        for y in y_coords:
            for x in x_coords:
                tile_id = f"{image_id}_t{tile_index:04d}_x{x}_y{y}"
                metadata_list.append(
                    TileMetadata(
                        tile_id=tile_id,
                        source_id=source_id,
                        pair_id=pair_id,
                        image_path=str(image_path) if image_path else None,
                        tile_x=x,
                        tile_y=y,
                        tile_width=self.config.tile_size,
                        tile_height=self.config.tile_size,
                        tile_index=tile_index,
                        original_width=orig_w,
                        original_height=orig_h,
                        split_group_id=split_group_id,
                    )
                )
                tile_index += 1

        return metadata_list

    def extract_tile(
        self,
        sample: Any,
        tile_x: int,
        tile_y: int,
        tile_index: int = 0,
    ) -> ExtractedTile:
        """
        Extract a single 256x256 tile at specific (x, y) coordinates.
        """
        proc = self._ensure_processed_sample(sample)
        img_arr = proc.image
        mask_arr = proc.mask if self.config.extract_mask else None

        orig_w, orig_h = proc.processed_size
        ts = self.config.tile_size

        # Apply safe padding if canvas is smaller than tile_size
        padded_img = pad_image_if_needed(img_arr, target_w=ts, target_h=ts, pad_color=self.config.pad_color)
        padded_mask = pad_mask_if_needed(mask_arr, target_w=ts, target_h=ts, mask_pad_val=self.config.mask_pad_val)

        # Coordinate bounds check
        if tile_x < 0 or tile_y < 0 or tile_x + ts > padded_img.shape[1] or tile_y + ts > padded_img.shape[0]:
            raise ValueError(
                f"Tile coordinates ({tile_x}, {tile_y}) exceed canvas boundaries {padded_img.shape[:2]} "
                f"for sample '{proc.image_id}'"
            )

        tile_img = padded_img[tile_y : tile_y + ts, tile_x : tile_x + ts, :].copy()
        tile_mask = padded_mask[tile_y : tile_y + ts, tile_x : tile_x + ts].copy() if padded_mask is not None else None

        tile_id = f"{proc.image_id}_t{tile_index:04d}_x{tile_x}_y{tile_y}"
        metadata = TileMetadata(
            tile_id=tile_id,
            source_id=proc.source_id,
            pair_id=proc.pair_id,
            image_path=str(proc.image_path) if proc.image_path else None,
            tile_x=tile_x,
            tile_y=tile_y,
            tile_width=ts,
            tile_height=ts,
            tile_index=tile_index,
            original_width=orig_w,
            original_height=orig_h,
            split_group_id=self._determine_split_group_id(proc),
        )

        return ExtractedTile(metadata=metadata, image=tile_img, mask=tile_mask)

    def iter_tiles_from_sample(self, sample: Any) -> Generator[ExtractedTile, None, None]:
        """
        Lazily yield 256x256 tiles one by one from a sample.

        Prevents accumulating large tile collections in memory.
        """
        proc = self._ensure_processed_sample(sample)
        img_arr = proc.image
        mask_arr = proc.mask if self.config.extract_mask else None

        orig_w, orig_h = proc.processed_size
        ts = self.config.tile_size

        padded_img = pad_image_if_needed(img_arr, target_w=ts, target_h=ts, pad_color=self.config.pad_color)
        padded_mask = pad_mask_if_needed(mask_arr, target_w=ts, target_h=ts, mask_pad_val=self.config.mask_pad_val)

        x_coords, y_coords = self.compute_grid_coordinates(orig_w, orig_h)
        tile_index = 0
        split_group_id = self._determine_split_group_id(proc)

        for y in y_coords:
            for x in x_coords:
                tile_img = padded_img[y : y + ts, x : x + ts, :].copy()
                tile_mask = padded_mask[y : y + ts, x : x + ts].copy() if padded_mask is not None else None

                tile_id = f"{proc.image_id}_t{tile_index:04d}_x{x}_y{y}"
                metadata = TileMetadata(
                    tile_id=tile_id,
                    source_id=proc.source_id,
                    pair_id=proc.pair_id,
                    image_path=str(proc.image_path) if proc.image_path else None,
                    tile_x=x,
                    tile_y=y,
                    tile_width=ts,
                    tile_height=ts,
                    tile_index=tile_index,
                    original_width=orig_w,
                    original_height=orig_h,
                    split_group_id=split_group_id,
                )
                yield ExtractedTile(metadata=metadata, image=tile_img, mask=tile_mask)
                tile_index += 1

    def extract_all_tiles_from_sample(self, sample: Any) -> List[ExtractedTile]:
        """Extract and return all tiles for a single sample as a list."""
        return list(self.iter_tiles_from_sample(sample))

    def extract_tiles_from_pair(self, pair: Any) -> Tuple[List[ExtractedTile], List[ExtractedTile]]:
        """
        Extract matched tiles from a clean-edited pair.

        Guarantees that clean and edited tiles have matching coordinates and shared split_group_id.
        """
        if isinstance(pair, ProcessedPair):
            proc_clean = pair.clean
            proc_edited = pair.edited
        else:
            proc_clean = self._ensure_processed_sample(pair.clean_sample)
            proc_edited = self._ensure_processed_sample(pair.edited_sample)

        clean_tiles = self.extract_all_tiles_from_sample(proc_clean)
        edited_tiles = self.extract_all_tiles_from_sample(proc_edited)

        if len(clean_tiles) != len(edited_tiles):
            raise RuntimeError(
                f"Tile count mismatch for pair '{pair.pair_id}': clean has {len(clean_tiles)} tiles, "
                f"edited has {len(edited_tiles)} tiles."
            )

        return clean_tiles, edited_tiles


class LazyTileDataset:
    """
    On-demand, indexable dataset representing all tiles across a screenshot collection.

    Memory-efficient design:
    - Precomputes an in-memory lightweight index map: (sample_idx, tile_x, tile_y, tile_index).
    - Only loads and preprocesses the specific screenshot when an item is requested via __getitem__.
    - Zero image files written to disk.
    - Zero full-dataset image arrays stored in RAM.
    """

    def __init__(
        self,
        samples: List[Any],
        extractor: Optional[TileExtractor] = None,
        preprocessor: Optional[Preprocessor] = None,
    ) -> None:
        self.samples = samples
        self.extractor = extractor or TileExtractor()
        self.preprocessor = preprocessor or self.extractor.preprocessor

        # Precompute flat tile index map
        self._tile_index_map: List[Tuple[int, int, int, int]] = []  # (sample_idx, tile_x, tile_y, tile_idx)
        self._sample_tile_counts: List[int] = []
        self._group_to_tile_indices: Dict[str, List[int]] = {}

        self._build_index()

    def _build_index(self) -> None:
        """Build flat index table mapping global tile index to sample and coordinates."""
        global_idx = 0
        for sample_idx, sample in enumerate(self.samples):
            metadata_list = self.extractor.get_tile_metadata_list(sample)
            self._sample_tile_counts.append(len(metadata_list))

            for meta in metadata_list:
                self._tile_index_map.append((sample_idx, meta.tile_x, meta.tile_y, meta.tile_index))
                group_id = meta.split_group_id
                if group_id not in self._group_to_tile_indices:
                    self._group_to_tile_indices[group_id] = []
                self._group_to_tile_indices[group_id].append(global_idx)
                global_idx += 1

    def __len__(self) -> int:
        """Total number of tiles across all samples in dataset."""
        return len(self._tile_index_map)

    def __getitem__(self, index: int) -> ExtractedTile:
        """Extract and return tile at global index on demand."""
        if index < 0 or index >= len(self._tile_index_map):
            raise IndexError(f"Tile index {index} out of range [0, {len(self._tile_index_map) - 1}]")

        sample_idx, tile_x, tile_y, tile_index = self._tile_index_map[index]
        sample = self.samples[sample_idx]
        return self.extractor.extract_tile(sample, tile_x=tile_x, tile_y=tile_y, tile_index=tile_index)

    def get_metadata(self, index: int) -> TileMetadata:
        """Retrieve tile metadata in O(1) time without loading or preprocessing image."""
        if index < 0 or index >= len(self._tile_index_map):
            raise IndexError(f"Tile index {index} out of range [0, {len(self._tile_index_map) - 1}]")

        sample_idx, tile_x, tile_y, tile_index = self._tile_index_map[index]
        sample = self.samples[sample_idx]

        orig_w = getattr(sample, "width", None)
        orig_h = getattr(sample, "height", None)
        if orig_w is None or orig_h is None:
            if hasattr(sample, "processed_size"):
                orig_w, orig_h = sample.processed_size
            else:
                proc = self.preprocessor.process_sample(sample)
                orig_w, orig_h = proc.processed_size

        image_id = getattr(sample, "image_id", "sample")
        source_id = getattr(sample, "source_id", "source")
        pair_id = getattr(sample, "pair_id", None)
        image_path = getattr(sample, "image_path", None)
        split_group_id = self.extractor._determine_split_group_id(sample)

        tile_id = f"{image_id}_t{tile_index:04d}_x{tile_x}_y{tile_y}"
        return TileMetadata(
            tile_id=tile_id,
            source_id=source_id,
            pair_id=pair_id,
            image_path=str(image_path) if image_path else None,
            tile_x=tile_x,
            tile_y=tile_y,
            tile_width=self.extractor.config.tile_size,
            tile_height=self.extractor.config.tile_size,
            tile_index=tile_index,
            original_width=orig_w,
            original_height=orig_h,
            split_group_id=split_group_id,
        )

    def get_sample_tile_counts(self) -> List[int]:
        """Return list of tile counts per source sample."""
        return list(self._sample_tile_counts)

    def get_unique_split_groups(self) -> List[str]:
        """Return sorted list of unique split_group_ids."""
        return sorted(list(self._group_to_tile_indices.keys()))

    def get_tiles_by_group(self, group_id: str) -> List[int]:
        """Return list of global tile indices belonging to given split group."""
        return self._group_to_tile_indices.get(group_id, [])

