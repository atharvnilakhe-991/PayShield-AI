"""
CBL-04: Tile-Level Inference & Probability Extraction Engine
Phase 6 - Member 2 (Tile Forensics CNN & Heatmap Lead)

This module handles:
1. Loading the saved Phase 5 model checkpoint in strict evaluation mode.
2. Integrating Phase 2 preprocessing and Phase 3 tile extraction.
3. Batched GPU/CPU forward inference under torch.no_grad().
4. Extracting softmax tampering probabilities for all 256x256 tiles of a screenshot.
5. Invoking HeatmapReconstructor to assemble the full-resolution probability map.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
PHASE1_DIR = HERE.parent / "phase1_dataset_loader"
PHASE2_DIR = HERE.parent / "phase2_preprocessing"
PHASE3_DIR = HERE.parent / "phase3_tile_extraction"
PHASE5_DIR = HERE.parent / "phase5_model_training"

for p in [PROJECT_ROOT, HERE, PHASE1_DIR, PHASE2_DIR, PHASE3_DIR, PHASE5_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from member2.phase2_preprocessing.transforms import Preprocessor
    from member2.phase3_tile_extraction.tile_extractor import TileConfig, TileExtractor, pad_image_if_needed
    from member2.phase5_model_training.model import get_model
    from member2.phase6_heatmap_reconstruction.reconstructor import HeatmapReconstructor
except (ImportError, ModuleNotFoundError):
    from get_model import get_model  # type: ignore[import-not-found]
    from reconstructor import HeatmapReconstructor  # type: ignore[import-not-found]
    from tile_extractor import TileConfig, TileExtractor, pad_image_if_needed  # type: ignore[import-not-found]
    from transforms import Preprocessor  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class TileInferencer:
    """
    Inference coordinator executing tile extraction, CNN forward passes, and heatmap reconstruction.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        reconstructor: Optional[HeatmapReconstructor] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        self.device = device or torch.device("cpu")
        self.batch_size = batch_size
        self.reconstructor = reconstructor or HeatmapReconstructor()

        # Load checkpoint
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model_name = ckpt.get("model_name", "resnet18")
        self.model_config = ckpt.get("config", {})
        num_classes = self.model_config.get("num_classes", 2)

        # Initialize model architecture and load weights
        self.model = get_model(model_name=self.model_name, num_classes=num_classes)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.normalization = self.model_config.get("normalization", "imagenet").lower()
        self.tile_extractor = TileExtractor(
            TileConfig(
                tile_size=self.reconstructor.tile_size,
                stride=self.model_config.get("stride", 128),
            )
        )
        logger.info(
            f"Initialized TileInferencer with {self.model_name} from {self.checkpoint_path} "
            f"(device={self.device}, normalization={self.normalization})"
        )

    def _prepare_tile_tensor(self, tile_crops: List[np.ndarray]) -> torch.Tensor:
        """Convert a list of (256, 256, 3) uint8 crops into a normalized PyTorch tensor batch."""
        tensors = []
        for crop in tile_crops:
            # (H, W, C) -> (C, H, W) in [0, 1]
            chw = np.transpose(crop, (2, 0, 1)).astype(np.float32) / 255.0
            if self.normalization == "imagenet":
                chw = (chw - IMAGENET_MEAN) / IMAGENET_STD
            tensors.append(torch.from_numpy(chw))
        return torch.stack(tensors, dim=0)

    @torch.no_grad()
    def predict_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Tuple[int, int]]:
        """
        Run inference over an entire screenshot.

        Returns:
            heatmap: (H, W) float32 array in [0.0, 1.0]
            tile_predictions: List of dicts with coordinates and probabilities
            original_size: (height, width)
        """
        # Load and standardize image to RGB numpy array
        if isinstance(image_input, (str, Path)):
            with Image.open(image_input) as img:
                rgb_img = Preprocessor.convert_to_rgb(img)
                img_arr = np.array(rgb_img, dtype=np.uint8)
        elif isinstance(image_input, Image.Image):
            rgb_img = Preprocessor.convert_to_rgb(image_input)
            img_arr = np.array(rgb_img, dtype=np.uint8)
        elif isinstance(image_input, np.ndarray):
            img_arr = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        orig_h, orig_w = img_arr.shape[:2]

        # Pad image if smaller than tile_size
        ts = self.reconstructor.tile_size
        padded_img = pad_image_if_needed(img_arr, target_w=ts, target_h=ts)

        # Compute Phase 3 grid coordinates
        x_coords, y_coords = self.tile_extractor.compute_grid_coordinates(orig_w, orig_h)

        # Extract all crops and retain coordinates
        tile_crops: List[np.ndarray] = []
        tile_coords: List[Tuple[int, int, int]] = []  # (x, y, tile_index)

        tile_idx = 0
        for y in y_coords:
            for x in x_coords:
                crop = padded_img[y : y + ts, x : x + ts, :]
                tile_crops.append(crop)
                tile_coords.append((x, y, tile_idx))
                tile_idx += 1

        # Run batched inference
        all_probs: List[float] = []
        n_tiles = len(tile_crops)

        for i in range(0, n_tiles, self.batch_size):
            batch_crops = tile_crops[i : i + self.batch_size]
            batch_tensor = self._prepare_tile_tensor(batch_crops).to(self.device)

            logits = self.model(batch_tensor)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
            all_probs.extend(probs)

        # Assemble tile prediction records
        tile_predictions: List[Dict[str, Any]] = []
        for (tx, ty, tidx), p in zip(tile_coords, all_probs):
            tile_predictions.append({
                "tile_index": tidx,
                "tile_x": tx,
                "tile_y": ty,
                "tile_width": ts,
                "tile_height": ts,
                "tampered_prob": float(p),
            })

        # Reconstruct full-resolution continuous heatmap
        heatmap = self.reconstructor.reconstruct(
            height=orig_h,
            width=orig_w,
            tile_predictions=tile_predictions,
        )

        return heatmap, tile_predictions, (orig_h, orig_w)

    def compute_image_level_scores(
        self,
        tile_predictions: List[Dict[str, Any]],
        heatmap: np.ndarray,
        threshold: float = 0.5,
        top_k: int = 3,
    ) -> Dict[str, float]:
        """
        Compute aggregate image-level tampering scores.
        """
        probs = [p["tampered_prob"] for p in tile_predictions]
        if not probs:
            return {"max_tile_prob": 0.0, "top_k_mean": 0.0, "tampered_pixel_ratio": 0.0}

        sorted_probs = sorted(probs, reverse=True)
        max_prob = sorted_probs[0]
        k = min(top_k, len(sorted_probs))
        top_k_mean = float(np.mean(sorted_probs[:k]))
        tampered_pixel_ratio = float(np.mean(heatmap >= threshold))

        return {
            "max_tile_prob": float(max_prob),
            "top_k_mean": float(top_k_mean),
            "tampered_pixel_ratio": float(tampered_pixel_ratio),
        }

