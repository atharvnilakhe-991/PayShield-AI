"""
Unit and Integration Tests for CBL-04 Mask-to-Tile Labeling Engine
Phase 4 - Member 2 (Tile Forensics CNN & Heatmap Lead)

Comprehensive 14-point test suite:
1. Completely clean mask produces strictly label 0 for all tiles.
2. Tile with >= 16 foreground pixels is labeled 1 (tampered).
3. Tile with 1 to 15 foreground pixels is labeled -1 (ambiguous).
4. Correct mask-to-tile coordinate alignment (tampered region mapped to correct spatial tile).
5. Binary mask normalization handles both values 1 and 255 equivalently.
6. Missing mask file paths are not silently replaced with zeros.
7. Unlabelled real screenshots remain unlabelled (tile_label=None, status="unlabelled").
8. Clean and edited tiles from the same transaction pair share split_group_id.
9. Deterministic labeling yields identical results across repeated runs.
10. Boundary anchored tiles are correctly labeled without coordinate clipping.
11. Threshold configuration via LabelingConfig alters labeling as configured.
12. LazyLabeledTileDataset provides on-demand labeled tile access.
13. Class imbalance statistics correctly identify zero-positive edited screenshots.
14. Original dataset files remain completely untouched and immutable (MD5 verified).
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

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
    from member2.phase2_preprocessing.transforms import PreprocessConfig, Preprocessor, ProcessedSample
    from member2.phase3_tile_extraction.tile_extractor import (
        ExtractedTile,
        LazyTileDataset,
        TileConfig,
        TileExtractor,
        TileMetadata,
    )
    from member2.phase4_tile_labeling.tile_labeler import (
        LabeledTile,
        LabeledTileMetadata,
        LabelingConfig,
        LazyLabeledTileDataset,
        TileLabeler,
    )
except (ImportError, ModuleNotFoundError):
    from dataset_loader import DatasetLoader, PairSample, SingleSample  # type: ignore[import-not-found]
    from tile_extractor import (  # type: ignore[import-not-found]
        ExtractedTile,
        LazyTileDataset,
        TileConfig,
        TileExtractor,
        TileMetadata,
    )
    from tile_labeler import (  # type: ignore[import-not-found]
        LabeledTile,
        LabeledTileMetadata,
        LabelingConfig,
        LazyLabeledTileDataset,
        TileLabeler,
    )
    from transforms import PreprocessConfig, Preprocessor, ProcessedSample  # type: ignore[import-not-found]


class TestTileLabeler(unittest.TestCase):
    """Test suite executing isolated unit tests and dataset integration tests."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_root = Path(self.temp_dir.name)
        (self.test_root / "images").mkdir(parents=True, exist_ok=True)
        (self.test_root / "masks").mkdir(parents=True, exist_ok=True)

        self.config = LabelingConfig(tampered_threshold=16, ambiguous_min=1, ambiguous_max=15)
        self.labeler = TileLabeler(self.config)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_synthetic_sample(
        self,
        image_id: str = "synth_label_001",
        size: tuple[int, int] = (583, 1441),
        foreground_regions: list[tuple[int, int, int, int]] | None = None,  # [(x1, y1, x2, y2)]
        foreground_value: int = 255,
        pair_id: str = "pair_100",
        source_id: str = "compact_100",
    ) -> SingleSample:
        """Create test image and mask on disk."""
        img_path = self.test_root / "images" / f"{image_id}.png"
        mask_path = self.test_root / "masks" / f"{image_id}_mask.png"

        w, h = size
        img = Image.new("RGB", (w, h), (18, 18, 18))
        img.save(img_path)

        mask_arr = np.zeros((h, w), dtype=np.uint8)
        if foreground_regions:
            for x1, y1, x2, y2 in foreground_regions:
                mask_arr[y1:y2, x1:x2] = foreground_value
        Image.fromarray(mask_arr, mode="L").save(mask_path)

        is_clean = not bool(foreground_regions)
        return SingleSample(
            image_id=image_id,
            source_id=source_id,
            pair_id=pair_id,
            image_path=img_path,
            mask_path=mask_path,
            label=0 if is_clean else 1,
            label_name="clean" if is_clean else "edited",
            is_synthetic=True,
            width=w,
            height=h,
        )

    def test_1_completely_clean_mask_all_label_zero(self) -> None:
        """1. A completely clean mask produces strictly label 0 for all tiles."""
        clean_sample = self._create_synthetic_sample(image_id="clean_all", foreground_regions=None)
        labeled_tiles = self.labeler.label_tiles_from_sample(clean_sample)

        self.assertGreater(len(labeled_tiles), 0)
        for tile in labeled_tiles:
            self.assertEqual(tile.tile_label, 0)
            self.assertEqual(tile.label_status, "clean")
            self.assertEqual(tile.tampered_pixel_count, 0)
            self.assertEqual(tile.tampered_pixel_ratio, 0.0)

    def test_2_tile_with_ge_16_foreground_pixels_is_tampered(self) -> None:
        """2. Tile with >= 16 foreground pixels is labeled 1 (tampered)."""
        # Create a 4x4 box = 16 pixels inside tile [0, 256] x [0, 256]
        sample = self._create_synthetic_sample(
            image_id="tampered_16",
            size=(512, 512),
            foreground_regions=[(50, 50, 54, 54)],  # 4x4 = 16 pixels
        )
        tiles = self.labeler.label_tiles_from_sample(sample)

        # Tile (0, 0) should be tampered
        tile_0 = next(t for t in tiles if t.tile_x == 0 and t.tile_y == 0)
        self.assertEqual(tile_0.tile_label, 1)
        self.assertEqual(tile_0.label_status, "tampered")
        self.assertEqual(tile_0.tampered_pixel_count, 16)
        self.assertAlmostEqual(tile_0.tampered_pixel_ratio, 16 / (256 * 256))

    def test_3_tile_with_1_to_15_foreground_pixels_is_ambiguous(self) -> None:
        """3. Tile with 1 to 15 foreground pixels is labeled -1 (ambiguous)."""
        # Create a 3x3 box = 9 pixels inside tile [0, 256] x [0, 256]
        sample = self._create_synthetic_sample(
            image_id="ambiguous_9",
            size=(512, 512),
            foreground_regions=[(50, 50, 53, 53)],  # 3x3 = 9 pixels
        )
        tiles = self.labeler.label_tiles_from_sample(sample)

        tile_0 = next(t for t in tiles if t.tile_x == 0 and t.tile_y == 0)
        self.assertEqual(tile_0.tile_label, -1)
        self.assertEqual(tile_0.label_status, "ambiguous")
        self.assertEqual(tile_0.tampered_pixel_count, 9)
        self.assertAlmostEqual(tile_0.tampered_pixel_ratio, 9 / (256 * 256))

    def test_4_mask_to_tile_coordinate_alignment(self) -> None:
        """4. Tampering at specific coordinates correctly labels only overlapping tiles."""
        # Place tampering in region [300, 320] x [400, 420] (20x20 = 400 pixels)
        sample = self._create_synthetic_sample(
            image_id="coord_align",
            size=(600, 800),
            foreground_regions=[(300, 400, 320, 420)],
        )
        tiles = self.labeler.label_tiles_from_sample(sample)

        for tile in tiles:
            x, y = tile.tile_x, tile.tile_y
            # Tile covers [x, x+256] x [y, y+256]
            x_overlap = max(0, min(x + 256, 320) - max(x, 300))
            y_overlap = max(0, min(y + 256, 420) - max(y, 400))
            expected_overlap_pixels = x_overlap * y_overlap

            self.assertEqual(
                tile.tampered_pixel_count,
                expected_overlap_pixels,
                f"Mismatch at tile ({x}, {y}): expected {expected_overlap_pixels}, got {tile.tampered_pixel_count}",
            )

    def test_5_binary_mask_normalization_values_1_and_255(self) -> None:
        """5. Foreground masks with value 1 and value 255 produce identical counts."""
        sample_255 = self._create_synthetic_sample(
            image_id="mask_255",
            size=(512, 512),
            foreground_regions=[(50, 50, 60, 60)],  # 10x10 = 100 pixels
            foreground_value=255,
        )
        sample_1 = self._create_synthetic_sample(
            image_id="mask_1",
            size=(512, 512),
            foreground_regions=[(50, 50, 60, 60)],  # 10x10 = 100 pixels
            foreground_value=1,
        )

        tiles_255 = self.labeler.label_tiles_from_sample(sample_255)
        tiles_1 = self.labeler.label_tiles_from_sample(sample_1)

        self.assertEqual(len(tiles_255), len(tiles_1))
        for t255, t1 in zip(tiles_255, tiles_1):
            self.assertEqual(t255.tile_label, t1.tile_label)
            self.assertEqual(t255.tampered_pixel_count, t1.tampered_pixel_count)

    def test_6_missing_mask_handled_safely(self) -> None:
        """6. A sample with invalid/missing mask path raises appropriate error or is rejected."""
        fake_sample = SingleSample(
            image_id="missing_mask",
            source_id="src_missing",
            image_path=self.test_root / "images" / "missing.png",
            mask_path=self.test_root / "masks" / "non_existent_mask.png",
            label=1,
            label_name="edited",
        )
        # Attempting to load missing image should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            self.labeler.label_tiles_from_sample(fake_sample)

    def test_7_real_unlabelled_screenshots_handled(self) -> None:
        """7. Real screenshots with mask=None receive label=None and status='unlabelled'."""
        img_path = self.test_root / "images" / "real_sample.png"
        Image.new("RGB", (400, 800), (20, 20, 20)).save(img_path)

        real_sample = SingleSample(
            image_id="real_gpay_099",
            source_id="real_gpay_099",
            image_path=img_path,
            mask_path=None,
            label=None,
            label_name="unlabeled",
            is_synthetic=False,
            width=400,
            height=800,
        )

        labeled_tiles = self.labeler.label_tiles_from_sample(real_sample)
        self.assertGreater(len(labeled_tiles), 0)
        for tile in labeled_tiles:
            self.assertIsNone(tile.tile_label)
            self.assertEqual(tile.label_status, "unlabelled")
            self.assertIsNone(tile.mask)

    def test_8_pair_and_source_group_id_preserved(self) -> None:
        """8. Clean and edited tiles from same pair retain identical split_group_id."""
        clean = self._create_synthetic_sample(image_id="p_clean", pair_id="pair_tx_777")
        edited = self._create_synthetic_sample(
            image_id="p_edited",
            pair_id="pair_tx_777",
            foreground_regions=[(50, 50, 100, 100)],
        )

        clean_tiles = self.labeler.label_tiles_from_sample(clean)
        edited_tiles = self.labeler.label_tiles_from_sample(edited)

        self.assertEqual(len(clean_tiles), len(edited_tiles))
        for ct, et in zip(clean_tiles, edited_tiles):
            self.assertEqual(ct.split_group_id, "pair_tx_777")
            self.assertEqual(et.split_group_id, "pair_tx_777")
            self.assertEqual(ct.split_group_id, et.split_group_id)

    def test_9_deterministic_labeling(self) -> None:
        """9. Repeated labeling on the same sample yields bitwise identical results."""
        sample = self._create_synthetic_sample(
            image_id="repeat_test",
            size=(400, 600),
            foreground_regions=[(80, 80, 120, 120)],
        )
        run1 = self.labeler.label_tiles_from_sample(sample)
        run2 = self.labeler.label_tiles_from_sample(sample)

        self.assertEqual(len(run1), len(run2))
        for t1, t2 in zip(run1, run2):
            self.assertEqual(t1.tile_id, t2.tile_id)
            self.assertEqual(t1.tile_label, t2.tile_label)
            self.assertEqual(t1.tampered_pixel_count, t2.tampered_pixel_count)

    def test_10_boundary_anchored_tiles_labeled_correctly(self) -> None:
        """10. Boundary anchored tiles at canvas borders accurately detect edge tampering."""
        w, h = 583, 1441
        # Place tampering in the bottom-right corner [570, 580] x [1430, 1440] = 10x10 = 100 px
        sample = self._create_synthetic_sample(
            image_id="edge_tamper",
            size=(w, h),
            foreground_regions=[(570, 1430, 580, 1440)],
        )
        tiles = self.labeler.label_tiles_from_sample(sample)

        # The bottom-right anchored tile is at x = (583 - 256) = 327, y = (1441 - 256) = 1185
        br_tile = next(t for t in tiles if t.tile_x == w - 256 and t.tile_y == h - 256)
        self.assertEqual(br_tile.tile_label, 1)
        self.assertEqual(br_tile.label_status, "tampered")
        self.assertEqual(br_tile.tampered_pixel_count, 100)

    def test_11_threshold_configuration(self) -> None:
        """11. Custom threshold alters classification of identical pixel counts."""
        crop = np.zeros((256, 256), dtype=np.uint8)
        crop[0:5, 0:4] = 1  # 20 foreground pixels

        # Default config: threshold 16 -> 20 pixels is tampered (1)
        default_label, default_status, _, _ = self.labeler.compute_label_from_mask_crop(crop)
        self.assertEqual(default_label, 1)
        self.assertEqual(default_status, "tampered")

        # Custom config: threshold 25 -> 20 pixels is ambiguous (-1)
        custom_cfg = LabelingConfig(tampered_threshold=25, ambiguous_min=1, ambiguous_max=24)
        custom_labeler = TileLabeler(custom_cfg)
        custom_label, custom_status, _, _ = custom_labeler.compute_label_from_mask_crop(crop)
        self.assertEqual(custom_label, -1)
        self.assertEqual(custom_status, "ambiguous")

    def test_12_lazy_labeled_dataset(self) -> None:
        """12. LazyLabeledTileDataset indexes tiles and computes class statistics."""
        s1 = self._create_synthetic_sample(image_id="lazy_clean", foreground_regions=None)
        s2 = self._create_synthetic_sample(
            image_id="lazy_tampered",
            foreground_regions=[(50, 50, 100, 100)],
        )
        dataset = LazyLabeledTileDataset([s1, s2], labeler=self.labeler)

        self.assertGreater(len(dataset), 0)
        tile_0 = dataset[0]
        self.assertIsInstance(tile_0, LabeledTile)
        self.assertIn(tile_0.label_status, ["clean", "tampered", "ambiguous"])

        stats = dataset.compute_class_imbalance_stats()
        self.assertGreater(stats["clean_tiles"], 0)
        self.assertGreater(stats["tampered_tiles"], 0)
        self.assertGreater(stats["class_imbalance_ratio_clean_to_tampered"], 1.0)

    def test_13_class_imbalance_identifies_zero_positive_edited(self) -> None:
        """13. Statistics accurately identify edited screenshots containing zero positive tiles."""
        # An edited screenshot where tampering was only 5 pixels (all ambiguous, 0 positive)
        s_ambiguous_only = self._create_synthetic_sample(
            image_id="edited_ambiguous_only",
            foreground_regions=[(50, 50, 55, 51)],  # 5x1 = 5 pixels (< 16)
        )
        dataset = LazyLabeledTileDataset([s_ambiguous_only], labeler=self.labeler)
        stats = dataset.compute_class_imbalance_stats()

        self.assertEqual(stats["tampered_tiles"], 0)
        self.assertGreater(stats["ambiguous_tiles"], 0)
        self.assertEqual(stats["edited_screenshots_with_zero_positive_tiles"], 1)

    def test_14_dataset_immutability(self) -> None:
        """14. Original dataset files remain strictly unchanged during labeling."""
        dataset_root = PROJECT_ROOT / "Hackathon-VIT" / "CBL04-Dataset"
        manifest_file = dataset_root / "metadata" / "phase4_manifest.csv"
        if not manifest_file.exists():
            self.skipTest("Manifest file not found for immutability check")

        hash_before = hashlib.md5(manifest_file.read_bytes()).hexdigest()

        loader = DatasetLoader(manifest_path=manifest_file)
        sample = loader.load(mode="SingleMode")[0]
        _ = self.labeler.label_tiles_from_sample(sample)

        hash_after = hashlib.md5(manifest_file.read_bytes()).hexdigest()
        self.assertEqual(hash_before, hash_after, "Dataset was altered during tile labeling!")


if __name__ == "__main__":
    unittest.main()

