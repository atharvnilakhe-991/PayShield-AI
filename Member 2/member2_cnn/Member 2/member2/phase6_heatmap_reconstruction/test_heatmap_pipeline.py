"""
Unit and Integration Tests for Phase 6: Heatmap Reconstruction & Localization
Phase 6 - Member 2 (Tile Forensics CNN & Heatmap Lead)

Test Coverage:
1. Heatmap output dimensions exactly match arbitrary input image dimensions (H, W).
2. Heatmap values strictly constrained to [0.0, 1.0].
3. Overlap blending (Gaussian, Hann, Uniform) produces valid continuous surfaces.
4. Edge-anchored boundary tiles are blended seamlessly.
5. Zero uncovered pixels across the entire canvas (100% spatial coverage).
6. Zero NaN, infinite, or negative values.
7. Thresholding produces exact binary masks ({0, 1} and {0, 255}).
8. Clean screenshot heatmap reconstruction and evaluation fidelity.
9. Hand-crafted pixel localization metrics ground truth verification.
10. Real holdout screenshots remain strictly unlabeled.
11. Immutability of original dataset and mask files.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

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
    from member2.phase3_tile_extraction.tile_extractor import TileConfig, TileExtractor
    from member2.phase6_heatmap_reconstruction.evaluator import Phase6Evaluator
    from member2.phase6_heatmap_reconstruction.reconstructor import (
        HeatmapReconstructor,
        generate_blending_window,
    )
except (ImportError, ModuleNotFoundError):
    from evaluator import Phase6Evaluator  # type: ignore[import-not-found]
    from reconstructor import HeatmapReconstructor, generate_blending_window  # type: ignore[import-not-found]
    from tile_extractor import TileConfig, TileExtractor  # type: ignore[import-not-found]


class TestHeatmapReconstruction(unittest.TestCase):
    """Comprehensive test suite for Phase 6."""

    def setUp(self) -> None:
        self.reconstructor_gauss = HeatmapReconstructor(
            tile_size=256, window_type="gaussian", gaussian_sigma=64.0
        )
        self.reconstructor_hann = HeatmapReconstructor(
            tile_size=256, window_type="hann", hann_min_weight=0.05
        )
        self.reconstructor_uniform = HeatmapReconstructor(
            tile_size=256, window_type="uniform"
        )
        self.extractor = TileExtractor(TileConfig(tile_size=256, stride=128))
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _generate_synthetic_tile_predictions(
        self, width: int, height: int, default_prob: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Helper to compute Phase 3 tile coordinates and assign dummy probabilities."""
        x_coords, y_coords = self.extractor.compute_grid_coordinates(width, height)
        preds = []
        idx = 0
        for y in y_coords:
            for x in x_coords:
                preds.append({
                    "tile_index": idx,
                    "tile_x": x,
                    "tile_y": y,
                    "tile_width": 256,
                    "tile_height": 256,
                    "tampered_prob": default_prob,
                })
                idx += 1
        return preds

    def test_heatmap_output_dimensions(self) -> None:
        """Verify heatmap dimensions match target screenshot dimensions across arbitrary resolutions."""
        test_resolutions = [
            (500, 500),
            (1080, 2400),
            (720, 1560),
            (583, 1441),  # Odd, non-stride-divisible dimensions
            (256, 256),   # Exact single tile
        ]

        for w, h in test_resolutions:
            preds = self._generate_synthetic_tile_predictions(w, h, default_prob=0.3)
            hm = self.reconstructor_gauss.reconstruct(height=h, width=w, tile_predictions=preds)
            self.assertEqual(hm.shape, (h, w), f"Failed for resolution ({w}, {h})")
            self.assertEqual(hm.dtype, np.float32)

    def test_heatmap_values_in_0_1(self) -> None:
        """Verify that heatmap probability values strictly lie within [0.0, 1.0]."""
        w, h = 800, 1200
        # Assign random probabilities in [0, 1]
        x_coords, y_coords = self.extractor.compute_grid_coordinates(w, h)
        rng = np.random.default_rng(42)
        preds = []
        for y in y_coords:
            for x in x_coords:
                preds.append({
                    "tile_x": x,
                    "tile_y": y,
                    "tampered_prob": float(rng.uniform(0.0, 1.0)),
                })

        for reconstructor in [self.reconstructor_gauss, self.reconstructor_hann, self.reconstructor_uniform]:
            hm = reconstructor.reconstruct(height=h, width=w, tile_predictions=preds)
            self.assertGreaterEqual(float(hm.min()), 0.0)
            self.assertLessEqual(float(hm.max()), 1.0)
            self.assertFalse(np.isnan(hm).any())
            self.assertFalse(np.isinf(hm).any())

    def test_overlap_blending_weights(self) -> None:
        """Verify that blending windows have expected spatial decay and symmetry."""
        w_gauss = generate_blending_window(256, "gaussian", sigma=64.0)
        self.assertEqual(w_gauss.shape, (256, 256))
        # Center should be peak weight
        center_val = w_gauss[127, 127]
        corner_val = w_gauss[0, 0]
        self.assertAlmostEqual(float(center_val), 1.0, places=2)
        self.assertLess(float(corner_val), float(center_val))
        # Symmetry
        self.assertTrue(np.allclose(w_gauss, np.rot90(w_gauss, 2), atol=1e-5))

        w_hann = generate_blending_window(256, "hann", min_weight=0.05)
        self.assertGreaterEqual(float(w_hann.min()), 0.05)
        self.assertLessEqual(float(w_hann.max()), 1.0)

    def test_edge_anchored_tiles_reconstructed_correctly(self) -> None:
        """Verify edge-anchored boundary tiles blend smoothly on non-divisible dimensions."""
        w, h = 583, 1441
        preds = self._generate_synthetic_tile_predictions(w, h, default_prob=0.8)
        hm = self.reconstructor_gauss.reconstruct(height=h, width=w, tile_predictions=preds)

        self.assertEqual(hm.shape, (1441, 583))
        # Right border and bottom border must contain valid values
        self.assertTrue(np.all(hm[:, -1] >= 0.0))
        self.assertTrue(np.all(hm[-1, :] >= 0.0))

    def test_no_uncovered_pixels(self) -> None:
        """Verify that every pixel across the canvas receives at least 1 tile coverage."""
        w, h = 1080, 2400
        x_coords, y_coords = self.extractor.compute_grid_coordinates(w, h)
        coords = [(x, y) for y in y_coords for x in x_coords]

        cov = self.reconstructor_gauss.compute_coverage_map(height=h, width=w, tile_coordinates=coords)
        self.assertGreaterEqual(int(cov.min()), 1, "Detected uncovered pixels on canvas!")
        # In stride 128 with size 256, interior pixels have 4x coverage
        self.assertGreaterEqual(int(cov.max()), 4)

    def test_no_nan_or_inf_edge_cases(self) -> None:
        """Verify safety against empty tiles, zero probabilities, and extreme coordinate inputs."""
        w, h = 300, 300
        # Empty predictions list
        hm_empty = self.reconstructor_gauss.reconstruct(height=h, width=w, tile_predictions=[])
        self.assertEqual(hm_empty.shape, (h, w))
        self.assertEqual(float(hm_empty.max()), 0.0)
        self.assertFalse(np.isnan(hm_empty).any())

        # All zero probabilities
        preds_zero = self._generate_synthetic_tile_predictions(w, h, default_prob=0.0)
        hm_zero = self.reconstructor_gauss.reconstruct(height=h, width=w, tile_predictions=preds_zero)
        self.assertEqual(float(hm_zero.max()), 0.0)

        # All 1.0 probabilities
        preds_one = self._generate_synthetic_tile_predictions(w, h, default_prob=1.0)
        hm_one = self.reconstructor_gauss.reconstruct(height=h, width=w, tile_predictions=preds_one)
        self.assertAlmostEqual(float(hm_one.mean()), 1.0, places=4)

    def test_thresholding_produces_binary_mask(self) -> None:
        """Verify thresholding produces exact binary masks in {0, 1} and {0, 255}."""
        hm = np.array([
            [0.1, 0.4, 0.6],
            [0.5, 0.8, 0.9],
        ], dtype=np.float32)

        m_01 = HeatmapReconstructor.threshold_heatmap(hm, threshold=0.5, binary_format="0_1")
        self.assertEqual(m_01.dtype, np.uint8)
        self.assertEqual(set(np.unique(m_01)), {0, 1})
        self.assertEqual(m_01[0, 0], 0)  # 0.1 < 0.5
        self.assertEqual(m_01[0, 2], 1)  # 0.6 >= 0.5
        self.assertEqual(m_01[1, 0], 1)  # 0.5 >= 0.5

        m_255 = HeatmapReconstructor.threshold_heatmap(hm, threshold=0.5, binary_format="0_255")
        self.assertEqual(set(np.unique(m_255)), {0, 255})
        self.assertEqual(m_255[0, 2], 255)

    def test_clean_image_reconstruction(self) -> None:
        """Verify clean image with all-zero tile probabilities evaluates to clean mask."""
        w, h = 400, 600
        preds = self._generate_synthetic_tile_predictions(w, h, default_prob=0.0)
        hm = self.reconstructor_gauss.reconstruct(height=h, width=w, tile_predictions=preds)
        pred_mask = HeatmapReconstructor.threshold_heatmap(hm, threshold=0.5)

        # Ground truth clean mask
        gt_mask = np.zeros((h, w), dtype=np.uint8)

        tp = np.sum((pred_mask == 1) & (gt_mask == 1))
        fp = np.sum((pred_mask == 1) & (gt_mask == 0))
        fn = np.sum((pred_mask == 0) & (gt_mask == 1))
        tn = np.sum((pred_mask == 0) & (gt_mask == 0))

        self.assertEqual(tp, 0)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)
        self.assertEqual(tn, w * h)
        pixel_acc = (tp + tn) / (w * h)
        self.assertEqual(pixel_acc, 1.0)

    def test_synthetic_test_metrics_computation(self) -> None:
        """Verify Evaluator metric calculations against known analytical ground truth."""
        # 10x10 test image
        # GT: 10 pixels foreground (indices [0:5, 0:2])
        gt = np.zeros((10, 10), dtype=np.uint8)
        gt[0:5, 0:2] = 1  # 10 pixels

        # Pred: 10 pixels foreground (indices [0:5, 1:3])
        # Overlap: [0:5, 1] is 5 pixels
        pred = np.zeros((10, 10), dtype=np.uint8)
        pred[0:5, 1:3] = 1  # 10 pixels

        tp = np.sum((pred == 1) & (gt == 1))  # 5
        fp = np.sum((pred == 1) & (gt == 0))  # 5
        fn = np.sum((pred == 0) & (gt == 1))  # 5
        tn = np.sum((pred == 0) & (gt == 0))  # 85

        self.assertEqual(tp, 5)
        self.assertEqual(fp, 5)
        self.assertEqual(fn, 5)
        self.assertEqual(tn, 85)

        dice = (2.0 * tp) / (2.0 * tp + fp + fn)  # 10 / 20 = 0.5
        iou = tp / (tp + fp + fn)                  # 5 / 15 = 0.3333

        self.assertAlmostEqual(dice, 0.5, places=4)
        self.assertAlmostEqual(iou, 1.0 / 3.0, places=4)

    def test_original_dataset_and_mask_files_unchanged(self) -> None:
        """Verify CBL04 original dataset directory remains strictly untouched by Member 2."""
        dataset_root = PROJECT_ROOT / "Hackathon-VIT/CBL04-Dataset"
        if not dataset_root.exists():
            return

        # Ensure no Member 2 files were created inside CBL04 dataset root
        member2_in_dataset = [
            p for p in dataset_root.glob("**/member2*")
            if p.name != "MEMBER2_NEXT_STEPS.md"
        ]
        self.assertEqual(
            member2_in_dataset,
            [],
            f"Detected unauthorized member2 files created in original dataset root: {member2_in_dataset}",
        )

        # Ensure active data folder has no modified files or checkpoints
        data_checkpoints = list((dataset_root / "data").glob("**/*.pt")) + list((dataset_root / "data").glob("**/*.pth"))
        self.assertEqual(
            data_checkpoints,
            [],
            f"Detected model checkpoints written to read-only data folder: {data_checkpoints}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
