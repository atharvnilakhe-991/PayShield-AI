"""
CBL-04: Tile Spatial Segmentation Dataset
Phase 6 Improvement - Member 2 (Tile Forensics CNN & Heatmap Lead)

This module provides a PyTorch Dataset yielding paired 256x256 RGB tiles and
256x256 binary ground-truth pixel masks for supervised segmentation training:
- Input: (3, 256, 256) float32 normalized image tile.
- Target: (1, 256, 256) float32 binary mask where 1.0 = tampered foreground.
- Balanced tile sampling: balances clean tiles and tampered tiles to prevent foreground starvation.
- Conservative forensic augmentation: subtle photometric jitter (brightness, contrast, noise)
  without geometric flips that violate dark-theme screenshot typography.
"""

from __future__ import annotations

import collections
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
PHASE2_DIR = HERE.parent / "phase2_preprocessing"
PHASE3_DIR = HERE.parent / "phase3_tile_extraction"
PHASE4_DIR = HERE.parent / "phase4_tile_labeling"

for p in [PROJECT_ROOT, HERE, PHASE2_DIR, PHASE3_DIR, PHASE4_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from member2.phase2_preprocessing.transforms import Preprocessor
    from member2.phase3_tile_extraction.tile_extractor import pad_image_if_needed, pad_mask_if_needed
    from member2.phase4_tile_labeling.tile_labeler import LabeledTileMetadata, TileLabeler
except (ImportError, ModuleNotFoundError):
    from tile_extractor import pad_image_if_needed, pad_mask_if_needed  # type: ignore[import-not-found]
    from tile_labeler import LabeledTileMetadata, TileLabeler  # type: ignore[import-not-found]
    from transforms import Preprocessor  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class LRUPairedImageCache:
    """In-memory cache for RGB image and binary mask arrays."""

    def __init__(self, capacity: int = 32) -> None:
        self.capacity = capacity
        self._img_cache: collections.OrderedDict[str, np.ndarray] = collections.OrderedDict()
        self._mask_cache: collections.OrderedDict[str, np.ndarray] = collections.OrderedDict()

    def get_image(self, path: str) -> Optional[np.ndarray]:
        if path not in self._img_cache:
            return None
        self._img_cache.move_to_end(path)
        return self._img_cache[path]

    def put_image(self, path: str, arr: np.ndarray) -> None:
        if path in self._img_cache:
            self._img_cache.move_to_end(path)
        self._img_cache[path] = arr
        if len(self._img_cache) > self.capacity:
            self._img_cache.popitem(last=False)

    def get_mask(self, path: str) -> Optional[np.ndarray]:
        if path not in self._mask_cache:
            return None
        self._mask_cache.move_to_end(path)
        return self._mask_cache[path]

    def put_mask(self, path: str, arr: np.ndarray) -> None:
        if path in self._mask_cache:
            self._mask_cache.move_to_end(path)
        self._mask_cache[path] = arr
        if len(self._mask_cache) > self.capacity:
            self._mask_cache.popitem(last=False)


class TileSpatialSegmentationDataset(Dataset):
    """
    PyTorch Dataset serving paired 256x256 RGB tiles and pixel ground-truth masks.
    """

    def __init__(
        self,
        samples: Optional[List[Any]] = None,
        tile_metadata_list: Optional[List[LabeledTileMetadata]] = None,
        labeler: Optional[TileLabeler] = None,
        normalization: str = "imagenet",
        max_samples: Optional[int] = None,
        balance_ratio: float = 0.5,  # 50% tampered, 50% clean in training
        augment: bool = False,
        seed: int = 42,
        cache_capacity: int = 32,
    ) -> None:
        self.normalization = normalization.lower()
        self.labeler = labeler or TileLabeler()
        self.augment = augment
        self.rng = random.Random(seed)
        self.cache = LRUPairedImageCache(capacity=cache_capacity)
        self.ts = self.labeler.config.tile_size

        # Index metadata items
        self.items: List[LabeledTileMetadata] = []

        if tile_metadata_list is not None:
            for m in tile_metadata_list:
                if m.tile_label in (0, 1):
                    self.items.append(m)
        elif samples is not None:
            for s in samples:
                mlist = self.labeler.get_labeled_metadata_list_for_sample(s)
                for m in mlist:
                    if m.tile_label in (0, 1):
                        self.items.append(m)

        # Balanced subsampling
        clean_items = [m for m in self.items if m.tile_label == 0]
        tamp_items = [m for m in self.items if m.tile_label == 1]

        if max_samples is not None and 0 < max_samples < len(self.items):
            self.rng.shuffle(clean_items)
            self.rng.shuffle(tamp_items)

            n_tamp = min(int(max_samples * balance_ratio), len(tamp_items))
            n_clean = max_samples - n_tamp

            selected = tamp_items[:n_tamp] + clean_items[:n_clean]
            self.rng.shuffle(selected)
            self.items = selected
        elif balance_ratio > 0 and len(tamp_items) > 0 and augment:
            # For training, oversample tampered tiles if requested
            n_tamp_target = int(len(clean_items) * (balance_ratio / (1.0 - balance_ratio)))
            oversampled_tamp = (tamp_items * (n_tamp_target // len(tamp_items) + 1))[:n_tamp_target]
            selected = clean_items + oversampled_tamp
            self.rng.shuffle(selected)
            self.items = selected

        self.num_clean = sum(1 for m in self.items if m.tile_label == 0)
        self.num_tampered = sum(1 for m in self.items if m.tile_label == 1)

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, path: str) -> np.ndarray:
        cached = self.cache.get_image(path)
        if cached is not None:
            return cached
        with Image.open(path) as img:
            rgb_img = Preprocessor.convert_to_rgb(img)
            arr = np.array(rgb_img, dtype=np.uint8)
        self.cache.put_image(path, arr)
        return arr

    def _load_mask(self, path: Optional[str], target_shape: Tuple[int, int]) -> np.ndarray:
        h, w = target_shape
        if not path or not Path(path).exists():
            return np.zeros((h, w), dtype=np.uint8)

        cached = self.cache.get_mask(path)
        if cached is not None:
            return cached

        with Image.open(path) as mim:
            mask_arr, _ = Preprocessor.convert_mask_to_binary(mim, foreground_val=1)

        self.cache.put_mask(path, mask_arr)
        return mask_arr

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        meta = self.items[index]
        image_path = meta.image_path
        mask_path = meta.mask_path

        full_img = self._load_image(image_path)
        h, w = full_img.shape[:2]
        full_mask = self._load_mask(mask_path, (h, w))

        # Pad if needed
        ts = self.ts
        padded_img = pad_image_if_needed(full_img, target_w=ts, target_h=ts)
        padded_mask = pad_mask_if_needed(full_mask, target_w=ts, target_h=ts, mask_pad_val=0)

        # Slice 256x256 crops
        x, y = meta.tile_x, meta.tile_y
        img_crop = padded_img[y : y + ts, x : x + ts, :].copy()
        mask_crop = padded_mask[y : y + ts, x : x + ts].copy()

        # Conservative augmentation (photometric only)
        if self.augment and self.rng.random() > 0.5:
            # Slight brightness jitter
            factor = self.rng.uniform(0.90, 1.10)
            img_crop = np.clip(img_crop.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # Image tensor: (3, 256, 256)
        img_chw = np.transpose(img_crop, (2, 0, 1)).astype(np.float32) / 255.0
        if self.normalization == "imagenet":
            img_chw = (img_chw - IMAGENET_MEAN) / IMAGENET_STD

        # Mask tensor: (1, 256, 256) float32 in {0.0, 1.0}
        mask_chw = (mask_crop > 0).astype(np.float32)[np.newaxis, :, :]

        img_tensor = torch.from_numpy(img_chw)
        mask_tensor = torch.from_numpy(mask_chw)

        info = {
            "tile_id": meta.tile_id,
            "source_id": meta.source_id,
            "pair_id": meta.pair_id,
            "split_group_id": meta.split_group_id,
            "layout_family": meta.layout_family or "unknown",
            "tile_x": meta.tile_x,
            "tile_y": meta.tile_y,
            "tile_label": meta.tile_label,
            "tampered_pixel_count": int(np.sum(mask_crop > 0)),
        }

        return img_tensor, mask_tensor, info

