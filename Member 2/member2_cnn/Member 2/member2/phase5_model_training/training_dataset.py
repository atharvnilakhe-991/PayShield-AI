"""
CBL-04: PyTorch Tile Forensics Dataset & Imbalance Weighting
Phase 5 - Member 2 (Tile Forensics CNN & Heatmap Lead)

This module provides:
1. `TileForensicsDataset`: High-performance PyTorch Dataset serving 256x256 RGB tiles.
   - Strict forensic filtering: excludes ambiguous tiles (label == -1) and unlabelled real holdout images.
   - Dynamic caching: LRU image buffer avoids redundant disk I/O when extracting multiple tiles per screenshot.
   - Normalization: supports standard ImageNet statistics or unit [0, 1] scaling.
2. `compute_class_weights`: Balanced inverse-frequency class weights computed strictly on training data.
3. `create_weighted_sampler`: PyTorch WeightedRandomSampler for balanced batching.
"""

from __future__ import annotations

import collections
import logging
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
PHASE1_DIR = HERE.parent / "phase1_dataset_loader"
PHASE2_DIR = HERE.parent / "phase2_preprocessing"
PHASE3_DIR = HERE.parent / "phase3_tile_extraction"
PHASE4_DIR = HERE.parent / "phase4_tile_labeling"

for p in [PROJECT_ROOT, HERE, PHASE1_DIR, PHASE2_DIR, PHASE3_DIR, PHASE4_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from member2.phase2_preprocessing.transforms import Preprocessor
    from member2.phase3_tile_extraction.tile_extractor import TileExtractor, pad_image_if_needed
    from member2.phase4_tile_labeling.tile_labeler import LabeledTileMetadata, TileLabeler
except (ImportError, ModuleNotFoundError):
    from tile_extractor import TileExtractor, pad_image_if_needed  # type: ignore[import-not-found]
    from tile_labeler import LabeledTileMetadata, TileLabeler  # type: ignore[import-not-found]
    from transforms import Preprocessor  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

# Standard ImageNet normalization parameters
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class LRUImageCache:
    """Least-Recently-Used in-memory image cache for fast tile cropping across batches."""

    def __init__(self, capacity: int = 32) -> None:
        self.capacity = capacity
        self._cache: collections.OrderedDict[str, np.ndarray] = collections.OrderedDict()

    def get(self, key: str) -> Optional[np.ndarray]:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, value: np.ndarray) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()


class TileForensicsDataset(Dataset):
    """
    PyTorch Dataset serving 256x256 RGB screenshot tiles with ground-truth binary labels.

    Forensics Specifications:
    - Input shape: (3, 256, 256), dtype: torch.float32
    - Target: torch.long scalar (0: clean, 1: tampered)
    - Ambiguous tiles (label == -1) are strictly excluded.
    - Unlabelled real holdouts (label is None) are strictly excluded.
    """

    def __init__(
        self,
        samples: Optional[List[Any]] = None,
        tile_metadata_list: Optional[List[LabeledTileMetadata]] = None,
        labeler: Optional[TileLabeler] = None,
        normalization: str = "imagenet",
        max_samples: Optional[int] = None,
        seed: int = 42,
        cache_capacity: int = 32,
    ) -> None:
        self.normalization = normalization.lower()
        self.labeler = labeler or TileLabeler()
        self.cache = LRUImageCache(capacity=cache_capacity)

        # Build flat list of validated labeled tile metadata
        self.items: List[LabeledTileMetadata] = []

        if tile_metadata_list is not None:
            # Use precomputed metadata
            for meta in tile_metadata_list:
                if meta.tile_label in (0, 1):
                    self.items.append(meta)
        elif samples is not None:
            # Extract metadata from samples
            for s in samples:
                meta_list = self.labeler.get_labeled_metadata_list_for_sample(s)
                for meta in meta_list:
                    # Filter ambiguous (-1) and unlabelled (None)
                    if meta.tile_label in (0, 1):
                        self.items.append(meta)

        # Subsample if max_samples is specified (e.g. for fast debugging/validation)
        if max_samples is not None and 0 < max_samples < len(self.items):
            rng = random.Random(seed)
            # Stratified subsampling by label to preserve class proportions
            clean_items = [m for m in self.items if m.tile_label == 0]
            tamp_items = [m for m in self.items if m.tile_label == 1]

            n_clean = int(round(max_samples * (len(clean_items) / len(self.items))))
            n_tamp = max_samples - n_clean

            rng.shuffle(clean_items)
            rng.shuffle(tamp_items)
            selected = clean_items[:n_clean] + tamp_items[:n_tamp]
            rng.shuffle(selected)
            self.items = selected

        # Calculate class distribution summary
        self.num_clean = sum(1 for m in self.items if m.tile_label == 0)
        self.num_tampered = sum(1 for m in self.items if m.tile_label == 1)

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image array with LRU caching and alpha flattening."""
        cached = self.cache.get(image_path)
        if cached is not None:
            return cached

        with Image.open(image_path) as img:
            rgb_img = Preprocessor.convert_to_rgb(img)
            arr = np.array(rgb_img, dtype=np.uint8)

        self.cache.put(image_path, arr)
        return arr

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Fetch a 256x256 RGB tile and its label.

        Returns:
            tile_tensor: (3, 256, 256) float32 tensor
            label_tensor: scalar torch.long (0 or 1)
            info: dictionary containing tile metadata
        """
        if index < 0 or index >= len(self.items):
            raise IndexError(f"Index {index} out of bounds [0, {len(self.items) - 1}]")

        meta = self.items[index]
        image_path = meta.image_path
        if not image_path or not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        full_img = self._load_image(image_path)
        h, w = full_img.shape[:2]

        # Handle edge padding if canvas is smaller than tile_size
        ts = self.labeler.config.tile_size
        padded_img = pad_image_if_needed(full_img, target_w=ts, target_h=ts)

        # Slice 256x256 crop
        x, y = meta.tile_x, meta.tile_y
        tile_crop = padded_img[y : y + ts, x : x + ts, :]

        # Convert to float32 (3, 256, 256) in [0, 1]
        tile_chw = np.transpose(tile_crop, (2, 0, 1)).astype(np.float32) / 255.0

        # Apply normalization
        if self.normalization == "imagenet":
            tile_chw = (tile_chw - IMAGENET_MEAN) / IMAGENET_STD

        tile_tensor = torch.from_numpy(tile_chw)
        label_tensor = torch.tensor(meta.tile_label, dtype=torch.long)

        info = {
            "tile_id": meta.tile_id,
            "source_id": meta.source_id,
            "pair_id": meta.pair_id,
            "split_group_id": meta.split_group_id,
            "layout_family": meta.layout_family or "unknown",
            "tile_x": meta.tile_x,
            "tile_y": meta.tile_y,
            "tampered_pixel_count": meta.tampered_pixel_count,
        }

        return tile_tensor, label_tensor, info

    def get_labels(self) -> List[int]:
        """Return list of all integer labels in the dataset."""
        return [m.tile_label for m in self.items if m.tile_label is not None]


def compute_class_weights(dataset: TileForensicsDataset) -> torch.Tensor:
    """
    Compute balanced inverse-frequency class weights strictly on training dataset.

    Formula: w_c = N / (num_classes * N_c)
    """
    labels = dataset.get_labels()
    n_total = len(labels)
    if n_total == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)

    n_clean = sum(1 for l in labels if l == 0)
    n_tampered = sum(1 for l in labels if l == 1)

    if n_clean == 0 or n_tampered == 0:
        logger.warning(f"Extreme class imbalance: clean={n_clean}, tampered={n_tampered}")
        return torch.tensor([1.0, 1.0], dtype=torch.float32)

    w_clean = n_total / (2.0 * n_clean)
    w_tampered = n_total / (2.0 * n_tampered)

    logger.info(
        f"Computed training class weights: clean={w_clean:.4f} (count={n_clean}), "
        f"tampered={w_tampered:.4f} (count={n_tampered})"
    )
    return torch.tensor([w_clean, w_tampered], dtype=torch.float32)


def create_weighted_sampler(dataset: TileForensicsDataset) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler that samples clean and tampered tiles with equal probability.
    """
    labels = dataset.get_labels()
    n_total = len(labels)
    n_clean = sum(1 for l in labels if l == 0)
    n_tampered = sum(1 for l in labels if l == 1)

    w_clean = 1.0 / max(n_clean, 1)
    w_tampered = 1.0 / max(n_tampered, 1)

    sample_weights = [w_clean if l == 0 else w_tampered for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=n_total,
        replacement=True,
    )
