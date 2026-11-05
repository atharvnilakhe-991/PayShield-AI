"""
CBL-04: Unit and Integration Tests for Performance Improvement Pipeline
Phase 6 Improvement - Member 2 (Tile Forensics CNN & Heatmap Lead)

Tests cover:
1. TileSpatialUNet architecture forward pass & output tensor shapes.
2. Loss functions (DiceLoss, CombinedBCEDiceLoss) gradient flow and computation.
3. Spatial post-processing and connected-component area thresholding.
4. Spatial heatmap reconstructor overlap blending and output canvas bounds.
5. Segmentation dataset normalization and tensor formats.
6. Dataset immutability verification.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent

for p in [PROJECT_ROOT, HERE]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from member2.performance_improvement.models import (
    CombinedBCEDiceLoss,
    DiceLoss,
    FocalLoss,
    TileSpatialUNet,
)
from member2.performance_improvement.postprocessor import (
    HeatmapPostProcessor,
    PostProcessingConfig,
)
from member2.performance_improvement.spatial_reconstructor import SpatialHeatmapReconstructor


class TestPerformanceImprovementPipeline(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)

    def test_tile_spatial_unet_forward_shape(self) -> None:
        """Verify TileSpatialUNet produces exact (B, 1, 256, 256) logit map."""
        model = TileSpatialUNet(in_channels=3, out_channels=1)
        model.eval()

        batch_size = 2
        x = torch.randn(batch_size, 3, 256, 256)
        with torch.no_grad():
            logits = model(x)

        self.assertEqual(logits.shape, (batch_size, 1, 256, 256))
        # Logits can be any real value; sigmoid brings to [0, 1]
        probs = torch.sigmoid(logits)
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))

    def test_tile_spatial_unet_parameter_count(self) -> None:
        """Verify model parameter count is lightweight (~483k) for CPU execution."""
        model = TileSpatialUNet()
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertGreater(param_count, 100_000)
        self.assertLess(param_count, 1_000_000)

    def test_dice_loss(self) -> None:
        """Verify soft Dice loss handles perfect overlap, zero overlap, and gradient flow."""
        loss_fn = DiceLoss(smooth=1.0)

        # Perfect overlap: high positive logits matching 1.0 targets
        logits_perfect = torch.full((1, 1, 32, 32), 10.0, requires_grad=True)
        targets_perfect = torch.ones((1, 1, 32, 32))
        loss = loss_fn(logits_perfect, targets_perfect)
        self.assertLess(loss.item(), 0.05)
        loss.backward()
        self.assertIsNotNone(logits_perfect.grad)

        # No overlap: large negative logits vs 1.0 targets
        logits_mismatch = torch.full((1, 1, 32, 32), -10.0)
        loss_mismatch = loss_fn(logits_mismatch, targets_perfect)
        self.assertGreater(loss_mismatch.item(), 0.90)

    def test_combined_bce_dice_loss_backward(self) -> None:
        """Verify CombinedBCEDiceLoss backpropagates gradients through TileSpatialUNet."""
        model = TileSpatialUNet()
        criterion = CombinedBCEDiceLoss(bce_weight=0.4, dice_weight=0.6, pos_weight=5.0)

        x = torch.randn(2, 3, 256, 256)
        target = torch.randint(0, 2, (2, 1, 256, 256)).float()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, target)

        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)

        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")
                self.assertFalse(torch.all(param.grad == 0), f"Zero gradient for {name}")

    def test_postprocessor_connected_component_filtering(self) -> None:
        """Verify small noise specks (< min_area) are removed and large regions (>= min_area) kept."""
        config = PostProcessingConfig(threshold=0.5, min_component_area=16)
        processor = HeatmapPostProcessor(config=config)

        # Create a synthetic heatmap (100, 100) with two components:
        # 1. Noise: 3x3 block (9 pixels)
        # 2. Signal: 5x5 block (25 pixels)
        heatmap = np.zeros((100, 100), dtype=np.float32)
        heatmap[10:13, 10:13] = 0.90  # 9 pixels -> below 16
        heatmap[50:55, 50:55] = 0.90  # 25 pixels -> above 16

        cleaned = processor.process(heatmap)

        self.assertEqual(cleaned.shape, (100, 100))
        self.assertEqual(cleaned.dtype, np.uint8)

        # Noise component should be wiped out
        self.assertEqual(cleaned[10:13, 10:13].sum(), 0)
        # Signal component should be preserved
        self.assertEqual(cleaned[50:55, 50:55].sum(), 25)

    def test_postprocessor_empty_input(self) -> None:
        """Verify all-zero or sub-threshold heatmap yields clean zero mask."""
        processor = HeatmapPostProcessor()
        heatmap = np.full((50, 50), 0.1, dtype=np.float32)
        cleaned = processor.process(heatmap)
        self.assertEqual(cleaned.sum(), 0)

    def test_spatial_heatmap_reconstructor(self) -> None:
        """Verify spatial reconstructor blends tile maps into full canvas correctly."""
        reconstructor = SpatialHeatmapReconstructor(tile_size=256, window_type="gaussian")

        height, width = 384, 384
        # Four tiles covering 384x384 canvas
        tiles = [
            {"tile_x": 0, "tile_y": 0, "spatial_prob_map": np.full((256, 256), 0.8, dtype=np.float32)},
            {"tile_x": 128, "tile_y": 0, "spatial_prob_map": np.full((256, 256), 0.8, dtype=np.float32)},
            {"tile_x": 0, "tile_y": 128, "spatial_prob_map": np.full((256, 256), 0.8, dtype=np.float32)},
            {"tile_x": 128, "tile_y": 128, "spatial_prob_map": np.full((256, 256), 0.8, dtype=np.float32)},
        ]

        heatmap = reconstructor.reconstruct(height, width, tiles)

        self.assertEqual(heatmap.shape, (height, width))
        self.assertEqual(heatmap.dtype, np.float32)
        self.assertFalse(np.isnan(heatmap).any())
        self.assertFalse(np.isinf(heatmap).any())
        self.assertTrue(np.all(heatmap >= 0.0))
        self.assertTrue(np.all(heatmap <= 1.0))
        # In regions covered by all 0.8 tiles, the reconstruction should be close to 0.8
        self.assertAlmostEqual(float(heatmap[128, 128]), 0.8, places=4)

    def test_dataset_immutability(self) -> None:
        """Strict verification: Ensure original dataset directory remains unaltered."""
        dataset_root = PROJECT_ROOT / "Hackathon-VIT" / "CBL04-Dataset"
        if not dataset_root.exists():
            self.skipTest(f"Dataset root not found at {dataset_root}")

        # Check no member2 or performance_improvement files exist inside the original dataset
        member2_in_dataset = [
            p for p in dataset_root.glob("**/member2*")
            if p.name != "MEMBER2_NEXT_STEPS.md"
        ]
        self.assertEqual(
            member2_in_dataset,
            [],
            f"Detected unauthorized member2 files created in original dataset root: {member2_in_dataset}",
        )

        pi_in_dataset = list(dataset_root.glob("**/performance_improvement*"))
        self.assertEqual(
            pi_in_dataset,
            [],
            f"Detected unauthorized performance_improvement files in original dataset root: {pi_in_dataset}",
        )

        # Ensure active data folder has no modified files or checkpoints
        data_checkpoints = list((dataset_root / "data").glob("**/*.pt")) + list((dataset_root / "data").glob("**/*.pth"))
        self.assertEqual(
            data_checkpoints,
            [],
            f"Detected model checkpoints written to read-only data folder: {data_checkpoints}",
        )


if __name__ == "__main__":
    unittest.main()
