"""
CBL-04: Mask-to-Tile Labeling Engine
Phase 4 - Member 2 (Tile Forensics CNN & Heatmap Lead)

This module assigns ground-truth forensic labels to 256x256 tiles extracted in Phase 3
using pixel-level tampering masks:
- Clean (label 0): foreground pixel count == 0.
- Tampered (label 1): foreground pixel count >= 16 pixels (configurable).
- Ambiguous (label -1): foreground pixel count between 1 and 15 pixels (configurable).
- Unlabelled (label None): unlabelled real screenshots where mask=None.

Key Features & Guarantees:
- Exact spatial alignment: masks are cropped using identical (tile_x, tile_y, tile_w, tile_h) coordinates.
- Safe binary mask normalization: handles {0, 1} and {0, 255} binary masks.
- Provenance & metadata preservation: retains all Phase 3 coordinates, source_id, pair_id,
  and split_group_id.
- Leakage prevention: clean and edited tiles from the same transaction pair share split_group_id.
- Memory efficiency: lazy on-demand tile labeling via generators and LazyLabeledTileDataset.
- Comprehensive class imbalance auditing: computes clean, tampered, ambiguous, and unlabelled counts,
  clean-to-tampered ratios, per-family statistics, and identifies edited screenshots with 0 positive tiles.
- Strict scope boundary: does NOT perform train/val/test splits, does NOT train CNN models,
  and does NOT modify original dataset files.
"""

from __future__ import annotations

import json
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
PHASE3_DIR = HERE.parent / "phase3_tile_extraction"

for p in [PROJECT_ROOT, HERE, PHASE1_DIR, PHASE2_DIR, PHASE3_DIR]:
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
    from member2.phase3_tile_extraction.tile_extractor import (
        ExtractedTile,
        LazyTileDataset,
        TileConfig,
        TileExtractor,
        TileMetadata,
        compute_tile_coordinates,
        pad_image_if_needed,
        pad_mask_if_needed,
    )
except (ImportError, ModuleNotFoundError):
    from dataset_loader import DatasetLoader, PairSample, SingleSample  # type: ignore[import-not-found]
    from tile_extractor import (  # type: ignore[import-not-found]
        ExtractedTile,
        LazyTileDataset,
        TileConfig,
        TileExtractor,
        TileMetadata,
        compute_tile_coordinates,
        pad_image_if_needed,
        pad_mask_if_needed,
    )
    from transforms import (  # type: ignore[import-not-found]
        DEFAULT_ALPHA_BG_COLOR,
        PreprocessConfig,
        Preprocessor,
        ProcessedPair,
        ProcessedSample,
    )

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = HERE / "LABELING_CONFIG.json"


@dataclass(frozen=True)
class LabelingConfig:
    """Configurable thresholds and rules for mask-to-tile labeling."""

    tampered_threshold: int = 16  # Minimum foreground pixels to classify as tampered (label 1)
    ambiguous_min: int = 1  # Minimum foreground pixels to classify as ambiguous (label -1)
    ambiguous_max: int = 15  # Maximum foreground pixels to classify as ambiguous (label -1)
    mask_foreground_threshold: int = 0  # Any mask pixel strictly greater than this is foreground
    tile_size: int = 256
    stride: int = 128

    def __post_init__(self) -> None:
        if self.tampered_threshold <= 0:
            raise ValueError(f"tampered_threshold must be positive, got {self.tampered_threshold}")
        if self.ambiguous_min < 0:
            raise ValueError(f"ambiguous_min must be non-negative, got {self.ambiguous_min}")
        if self.ambiguous_min > self.ambiguous_max:
            raise ValueError(
                f"ambiguous_min ({self.ambiguous_min}) cannot exceed ambiguous_max ({self.ambiguous_max})"
            )
        if self.ambiguous_max >= self.tampered_threshold:
            raise ValueError(
                f"ambiguous_max ({self.ambiguous_max}) must be strictly less than tampered_threshold ({self.tampered_threshold})"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LabelingConfig:
        return cls(**data)

    @classmethod
    def from_json(cls, path_or_str: Union[str, Path]) -> LabelingConfig:
        p = Path(path_or_str)
        if p.exists() and p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(str(path_or_str))
        return cls.from_dict(data)


@dataclass
class LabeledTileMetadata:
    """Complete provenance, spatial coordinates, and forensic labeling metadata for a tile."""

    tile_id: str
    source_id: str
    pair_id: Optional[str]
    image_path: Optional[str]
    mask_path: Optional[str]
    tile_x: int
    tile_y: int
    tile_width: int
    tile_height: int
    tile_index: int
    original_width: int
    original_height: int
    split_group_id: str
    tile_label: Optional[int]  # 0: clean, 1: tampered, -1: ambiguous, None: unlabelled
    label_status: str  # "clean", "tampered", "ambiguous", "unlabelled"
    tampered_pixel_count: int
    tampered_pixel_ratio: float
    label_threshold: int
    layout_family: Optional[str] = None
    sample_label: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LabeledTile:
    """Represents a fully extracted and labeled 256x256 tile."""

    metadata: LabeledTileMetadata
    image: np.ndarray  # Shape: (256, 256, 3), dtype: uint8
    mask: Optional[np.ndarray] = None  # Shape: (256, 256), dtype: uint8 or None

    def __post_init__(self) -> None:
        expected_shape = (self.metadata.tile_height, self.metadata.tile_width, 3)
        if self.image.shape != expected_shape:
            raise ValueError(
                f"LabeledTile image shape mismatch for '{self.metadata.tile_id}': "
                f"expected {expected_shape}, got {self.image.shape}"
            )
        if self.mask is not None:
            expected_mask_shape = (self.metadata.tile_height, self.metadata.tile_width)
            if self.mask.shape != expected_mask_shape:
                raise ValueError(
                    f"LabeledTile mask shape mismatch for '{self.metadata.tile_id}': "
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
    def tile_label(self) -> Optional[int]:
        return self.metadata.tile_label

    @property
    def label_status(self) -> str:
        return self.metadata.label_status

    @property
    def tampered_pixel_count(self) -> int:
        return self.metadata.tampered_pixel_count

    @property
    def tampered_pixel_ratio(self) -> float:
        return self.metadata.tampered_pixel_ratio

    @property
    def has_mask(self) -> bool:
        return self.mask is not None

    @property
    def tile_x(self) -> int:
        return self.metadata.tile_x

    @property
    def tile_y(self) -> int:
        return self.metadata.tile_y

    @property
    def tile_index(self) -> int:
        return self.metadata.tile_index

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


class TileLabeler:
    """
    Forensic tile labeling engine assigning clean, tampered, ambiguous, or unlabelled labels.

    Usage:
        config = LabelingConfig.from_json("LABELING_CONFIG.json")
        labeler = TileLabeler(config)
        labeled_tiles = labeler.label_tiles_from_sample(sample)
    """

    def __init__(
        self,
        config: Optional[LabelingConfig] = None,
        extractor: Optional[TileExtractor] = None,
        preprocessor: Optional[Preprocessor] = None,
    ) -> None:
        if config is not None:
            self.config = config
        elif DEFAULT_CONFIG_PATH.exists():
            self.config = LabelingConfig.from_json(DEFAULT_CONFIG_PATH)
        else:
            self.config = LabelingConfig()

        tile_cfg = TileConfig(tile_size=self.config.tile_size, stride=self.config.stride, extract_mask=True)
        self.extractor = extractor or TileExtractor(tile_cfg)
        self.preprocessor = preprocessor or self.extractor.preprocessor

    def compute_label_from_mask_crop(
        self,
        mask_crop: Optional[np.ndarray],
    ) -> Tuple[Optional[int], str, int, float]:
        """
        Evaluate a 2D mask crop and return (tile_label, label_status, tampered_pixel_count, tampered_pixel_ratio).

        Rules:
        - If mask_crop is None: (None, "unlabelled", 0, 0.0)
        - If count == 0: (0, "clean", 0, 0.0)
        - If count >= tampered_threshold (16): (1, "tampered", count, ratio)
        - If 1 <= count <= ambiguous_max (15): (-1, "ambiguous", count, ratio)
        """
        if mask_crop is None:
            return None, "unlabelled", 0, 0.0

        # Foreground check: strictly greater than mask_foreground_threshold (0)
        tampered_pixels = int(np.sum(mask_crop > self.config.mask_foreground_threshold))
        total_pixels = mask_crop.size
        ratio = float(tampered_pixels / total_pixels) if total_pixels > 0 else 0.0

        if tampered_pixels == 0:
            return 0, "clean", 0, 0.0
        elif tampered_pixels >= self.config.tampered_threshold:
            return 1, "tampered", tampered_pixels, ratio
        else:
            # 1 <= tampered_pixels < tampered_threshold (ambiguous)
            return -1, "ambiguous", tampered_pixels, ratio

    def label_extracted_tile(
        self,
        extracted_tile: ExtractedTile,
        mask_path: Optional[Union[str, Path]] = None,
        layout_family: Optional[str] = None,
        sample_label: Optional[int] = None,
    ) -> LabeledTile:
        """
        Assign ground truth label to a pre-extracted Phase 3 tile.
        """
        meta = extracted_tile.metadata
        mask_crop = extracted_tile.mask

        tile_label, label_status, fg_count, fg_ratio = self.compute_label_from_mask_crop(mask_crop)

        labeled_meta = LabeledTileMetadata(
            tile_id=meta.tile_id,
            source_id=meta.source_id,
            pair_id=meta.pair_id,
            image_path=meta.image_path,
            mask_path=str(mask_path) if mask_path else None,
            tile_x=meta.tile_x,
            tile_y=meta.tile_y,
            tile_width=meta.tile_width,
            tile_height=meta.tile_height,
            tile_index=meta.tile_index,
            original_width=meta.original_width,
            original_height=meta.original_height,
            split_group_id=meta.split_group_id,
            tile_label=tile_label,
            label_status=label_status,
            tampered_pixel_count=fg_count,
            tampered_pixel_ratio=fg_ratio,
            label_threshold=self.config.tampered_threshold,
            layout_family=layout_family,
            sample_label=sample_label,
        )

        return LabeledTile(metadata=labeled_meta, image=extracted_tile.image, mask=extracted_tile.mask)

    def iter_labeled_tiles_from_sample(self, sample: Any) -> Generator[LabeledTile, None, None]:
        """
        Lazily yield LabeledTile instances one by one from a sample.
        """
        mask_path = getattr(sample, "mask_path", None)
        layout_family = getattr(sample, "layout_family", None)
        sample_label = getattr(sample, "label", None)
        for tile in self.extractor.iter_tiles_from_sample(sample):
            yield self.label_extracted_tile(
                tile,
                mask_path=mask_path,
                layout_family=layout_family,
                sample_label=sample_label,
            )

    def label_tiles_from_sample(self, sample: Any) -> List[LabeledTile]:
        """Extract and label all tiles from a sample as a list."""
        return list(self.iter_labeled_tiles_from_sample(sample))

    def get_labeled_metadata_list_for_sample(self, sample: Any) -> List[LabeledTileMetadata]:
        """
        Efficiently compute tile labeling metadata without retaining raw image arrays in RAM.

        Loads mask crop directly, evaluates label, and builds LabeledTileMetadata list.
        """
        mask_path = getattr(sample, "mask_path", None)
        image_path = getattr(sample, "image_path", None)
        image_id = getattr(sample, "image_id", "sample")
        source_id = getattr(sample, "source_id", "source")
        pair_id = getattr(sample, "pair_id", None)
        layout_family = getattr(sample, "layout_family", None)
        sample_label = getattr(sample, "label", None)
        split_group_id = self.extractor._determine_split_group_id(sample)

        # Obtain canvas dimensions
        orig_w = getattr(sample, "width", None)
        orig_h = getattr(sample, "height", None)

        # Load mask array if available
        mask_arr: Optional[np.ndarray] = None
        if hasattr(sample, "load_mask") and mask_path:
            raw_mask = sample.load_mask()
            if raw_mask is not None:
                mask_arr, _ = Preprocessor.convert_mask_to_binary(raw_mask)
        elif hasattr(sample, "mask") and sample.mask is not None:
            mask_arr = sample.mask
        elif mask_path and Path(mask_path).exists():
            with Image.open(mask_path) as mim:
                mask_arr, _ = Preprocessor.convert_mask_to_binary(mim)

        if orig_w is None or orig_h is None:
            if mask_arr is not None:
                orig_h, orig_w = mask_arr.shape
            else:
                proc = self.preprocessor.process_sample(sample)
                orig_w, orig_h = proc.processed_size

        ts = self.config.tile_size
        padded_mask = pad_mask_if_needed(mask_arr, target_w=ts, target_h=ts, mask_pad_val=0) if mask_arr is not None else None

        x_coords, y_coords = self.extractor.compute_grid_coordinates(orig_w, orig_h)
        metadata_list: List[LabeledTileMetadata] = []
        tile_index = 0

        for y in y_coords:
            for x in x_coords:
                if padded_mask is not None:
                    crop = padded_mask[y : y + ts, x : x + ts]
                    label, status, count, ratio = self.compute_label_from_mask_crop(crop)
                else:
                    label, status, count, ratio = None, "unlabelled", 0, 0.0

                tile_id = f"{image_id}_t{tile_index:04d}_x{x}_y{y}"
                metadata_list.append(
                    LabeledTileMetadata(
                        tile_id=tile_id,
                        source_id=source_id,
                        pair_id=pair_id,
                        image_path=str(image_path) if image_path else None,
                        mask_path=str(mask_path) if mask_path else None,
                        tile_x=x,
                        tile_y=y,
                        tile_width=ts,
                        tile_height=ts,
                        tile_index=tile_index,
                        original_width=orig_w,
                        original_height=orig_h,
                        split_group_id=split_group_id,
                        tile_label=label,
                        label_status=status,
                        tampered_pixel_count=count,
                        tampered_pixel_ratio=ratio,
                        label_threshold=self.config.tampered_threshold,
                        layout_family=layout_family,
                        sample_label=sample_label,
                    )
                )
                tile_index += 1

        return metadata_list


class LazyLabeledTileDataset:
    """
    On-demand indexable dataset providing labeled tiles across an entire screenshot collection.

    Memory-efficient architecture:
    - Precomputes an in-memory index table of LabeledTileMetadata without storing raw images.
    - Slices raw RGB images only when __getitem__ is invoked.
    - Provides O(1) metadata access via get_metadata(index).
    - Computes complete class imbalance statistics across the dataset.
    """

    def __init__(
        self,
        samples: List[Any],
        labeler: Optional[TileLabeler] = None,
    ) -> None:
        self.samples = samples
        self.labeler = labeler or TileLabeler()
        self.extractor = self.labeler.extractor

        # Internal index structures
        self._tile_index_map: List[Tuple[int, int, int, int]] = []  # (sample_idx, tile_x, tile_y, tile_idx)
        self._tile_metadata_cache: List[LabeledTileMetadata] = []
        self._group_to_tile_indices: Dict[str, List[int]] = {}

        self._build_index()

    def _build_index(self) -> None:
        """Build flat metadata and index table across all samples."""
        global_idx = 0
        for sample_idx, sample in enumerate(self.samples):
            meta_list = self.labeler.get_labeled_metadata_list_for_sample(sample)
            for meta in meta_list:
                self._tile_metadata_cache.append(meta)
                self._tile_index_map.append((sample_idx, meta.tile_x, meta.tile_y, meta.tile_index))

                group_id = meta.split_group_id
                if group_id not in self._group_to_tile_indices:
                    self._group_to_tile_indices[group_id] = []
                self._group_to_tile_indices[group_id].append(global_idx)
                global_idx += 1

    def __len__(self) -> int:
        return len(self._tile_metadata_cache)

    def __getitem__(self, index: int) -> LabeledTile:
        """Extract and label tile on demand."""
        if index < 0 or index >= len(self._tile_index_map):
            raise IndexError(f"Index {index} out of range [0, {len(self._tile_index_map) - 1}]")

        meta = self._tile_metadata_cache[index]
        sample_idx, tile_x, tile_y, tile_idx = self._tile_index_map[index]
        sample = self.samples[sample_idx]

        # Extract Phase 3 tile on demand
        extracted_tile = self.extractor.extract_tile(sample, tile_x=tile_x, tile_y=tile_y, tile_index=tile_idx)

        # Wrap in LabeledTile using cached metadata
        return LabeledTile(metadata=meta, image=extracted_tile.image, mask=extracted_tile.mask)

    def get_metadata(self, index: int) -> LabeledTileMetadata:
        """Retrieve labeled tile metadata in O(1) time without loading image pixels."""
        if index < 0 or index >= len(self._tile_metadata_cache):
            raise IndexError(f"Index {index} out of range [0, {len(self._tile_metadata_cache) - 1}]")
        return self._tile_metadata_cache[index]

    def compute_class_imbalance_stats(self) -> Dict[str, Any]:
        """Compute comprehensive class distribution, imbalance ratio, and per-family statistics."""
        clean_count = 0
        tampered_count = 0
        ambiguous_count = 0
        unlabelled_count = 0

        # Per-image tracking
        image_tampered_counts: Dict[str, int] = {}
        image_is_edited: Dict[str, bool] = {}

        # Per-layout family tracking
        family_stats: Dict[str, Dict[str, int]] = {}

        for meta in self._tile_metadata_cache:
            status = meta.label_status
            img_id = meta.tile_id.rsplit("_t", 1)[0] if "_t" in meta.tile_id else meta.source_id

            if img_id not in image_tampered_counts:
                image_tampered_counts[img_id] = 0
                is_ed = (meta.sample_label == 1) or ("edit" in img_id.lower())
                image_is_edited[img_id] = is_ed

            fam = (meta.layout_family or "unknown").lower()
            if fam not in family_stats:
                family_stats[fam] = {"clean": 0, "tampered": 0, "ambiguous": 0, "unlabelled": 0, "total": 0}

            family_stats[fam]["total"] += 1

            if status == "clean":
                clean_count += 1
                family_stats[fam]["clean"] += 1
            elif status == "tampered":
                tampered_count += 1
                image_tampered_counts[img_id] += 1
                family_stats[fam]["tampered"] += 1
            elif status == "ambiguous":
                ambiguous_count += 1
                family_stats[fam]["ambiguous"] += 1
            elif status == "unlabelled":
                unlabelled_count += 1
                family_stats[fam]["unlabelled"] += 1

        total_labeled = clean_count + tampered_count + ambiguous_count
        tampered_pct = (tampered_count / total_labeled * 100.0) if total_labeled > 0 else 0.0
        ambiguous_pct = (ambiguous_count / total_labeled * 100.0) if total_labeled > 0 else 0.0
        clean_pct = (clean_count / total_labeled * 100.0) if total_labeled > 0 else 0.0

        imbalance_ratio = (clean_count / max(1, tampered_count)) if tampered_count > 0 else float("inf")

        # Edited screenshots containing zero positive tiles analysis
        edited_images = [img for img, is_ed in image_is_edited.items() if is_ed]
        zero_pos_edited = [img for img in edited_images if image_tampered_counts[img] == 0]
        zero_pos_pct = (len(zero_pos_edited) / len(edited_images) * 100.0) if edited_images else 0.0

        return {
            "total_tiles": len(self._tile_metadata_cache),
            "labeled_tiles_total": total_labeled,
            "clean_tiles": clean_count,
            "tampered_tiles": tampered_count,
            "ambiguous_tiles": ambiguous_count,
            "unlabelled_tiles": unlabelled_count,
            "clean_percentage": round(clean_pct, 2),
            "tampered_percentage": round(tampered_pct, 2),
            "ambiguous_percentage": round(ambiguous_pct, 2),
            "class_imbalance_ratio_clean_to_tampered": round(imbalance_ratio, 2),
            "edited_screenshots_audited": len(edited_images),
            "edited_screenshots_with_zero_positive_tiles": len(zero_pos_edited),
            "edited_screenshots_with_zero_positive_percentage": round(zero_pos_pct, 2),
            "by_layout_family": family_stats,
        }
