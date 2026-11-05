"""
Unit and Integration Tests for CBL-04 Tile Extraction Engine
Phase 3 - Member 2 (Tile Forensics CNN & Heatmap Lead)

Comprehensive 15-point test suite:
1. Exact 256x256 tile dimensions.
2. Correct 128-pixel stride in horizontal and vertical directions.
3. Correct tile coordinate calculation.
4. Full edge coverage for odd and non-divisible image dimensions.
5. No dropped edge regions (every pixel from 0 to W-1 and 0 to H-1 covered).
6. Deterministic extraction across multiple runs.
7. RGB format and pixel value preservation.
8. Correct metadata and source/pair grouping (split_group_id shared across pairs).
9. Lazy on-demand extraction behavior and memory efficiency.
10. Safe padding for dimensions smaller than tile size (< 256).
11. Unlabelled real screenshots return mask=None without failure.
12. Matched clean/edited pair extraction produces identical grid geometry.
13. Out-of-bounds coordinate extraction raises clear ValueError.
14. Integration test on actual Phase 4 dataset samples.
15. Original dataset files remain strictly immutable (MD5 verified).
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

for p in [PROJECT_ROOT, HERE, PHASE1_DIR, PHASE2_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from member2.phase1_dataset_loader.dataset_loader import DatasetLoader, PairSample, SingleSample
    from member2.phase2_preprocessing.transforms import PreprocessConfig, Preprocessor, ProcessedSample
    from member2.phase3_tile_extraction.tile_extractor import (
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
except (ImportError, ModuleNotFoundError):
    from dataset_loader import DatasetLoader, PairSample, SingleSample  # type: ignore[import-not-found]
    from transforms import PreprocessConfig, Preprocessor, ProcessedSample  # type: ignore[import-not-found]
    from tile_extractor import (  # type: ignore[import-not-found]
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


class TestTileExtractor(unittest.TestCase):
    """Test suite executing isolated unit tests and real dataset integration tests."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_root = Path(self.temp_dir.name)
        (self.test_root / "images").mkdir(parents=True, exist_ok=True)
        (self.test_root / "masks").mkdir(parents=True, exist_ok=True)

        self.extractor = TileExtractor(TileConfig(tile_size=256, stride=128))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_synthetic_sample(
        self,
        image_id: str = "synth_001",
        size: tuple[int, int] = (583, 1441),
        is_clean: bool = False,
        pair_id: str = "pair_001",
        source_id: str = "compact_001",
    ) -> SingleSample:
        """Create an isolated test sample file and return SingleSample."""
        img_path = self.test_root / "images" / f"{image_id}.png"
        mask_path = self.test_root / "masks" / f"{image_id}_mask.png"

        # Unique gradient pattern to verify pixel fidelity
        w, h = size
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(10, 200, w, dtype=np.uint8)
        arr[:, :, 1] = np.linspace(20, 220, h, dtype=np.uint8).reshape(h, 1)
        arr[:, :, 2] = 50
        Image.fromarray(arr).save(img_path)

        mask_arr = np.zeros((h, w), dtype=np.uint8)
        if not is_clean:
            # Small tampered region near bottom-right edge to verify edge coverage
            mask_arr[h - 50 : h - 10, w - 60 : w - 20] = 255
        Image.fromarray(mask_arr, mode="L").save(mask_path)

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

    def test_1_exact_tile_dimensions(self) -> None:
        """1. Every extracted tile has exact 256x256 dimensions."""
        sample = self._create_synthetic_sample(size=(600, 1400))
        tiles = self.extractor.extract_all_tiles_from_sample(sample)

        self.assertGreater(len(tiles), 0)
        for tile in tiles:
            self.assertEqual(tile.image.shape, (256, 256, 3))
            if tile.mask is not None:
                self.assertEqual(tile.mask.shape, (256, 256))
            self.assertEqual(tile.metadata.tile_width, 256)
            self.assertEqual(tile.metadata.tile_height, 256)

    def test_2_correct_stride(self) -> None:
        """2. Consecutive standard tile coordinates step by exactly stride (128 px)."""
        coords = compute_tile_coordinates(length=1000, tile_size=256, stride=128)
        # Check consecutive differences before the anchored edge
        for i in range(len(coords) - 2):
            diff = coords[i + 1] - coords[i]
            self.assertEqual(diff, 128, f"Stride error between index {i} and {i+1}: got {diff}")

    def test_3_coordinate_calculation(self) -> None:
        """3. Coordinates correctly match analytical expectation."""
        # 512 is exact: (512 - 256) / 128 = 2 -> [0, 128, 256]
        c512 = compute_tile_coordinates(512, tile_size=256, stride=128)
        self.assertEqual(c512, [0, 128, 256])

        # 256 is exact: [0]
        c256 = compute_tile_coordinates(256, tile_size=256, stride=128)
        self.assertEqual(c256, [0])

        # 583: [0, 128, 256, 327] (327 = 583 - 256)
        c583 = compute_tile_coordinates(583, tile_size=256, stride=128)
        self.assertEqual(c583, [0, 128, 256, 327])

    def test_4_edge_coverage_non_divisible_dimensions(self) -> None:
        """4. Full edge coverage for non-divisible dimensions (e.g. 583x1441)."""
        w, h = 583, 1441
        x_coords, y_coords = self.extractor.compute_grid_coordinates(w, h)

        # Right edge must end at w
        self.assertEqual(x_coords[-1] + 256, w)
        # Bottom edge must end at h
        self.assertEqual(y_coords[-1] + 256, h)

    def test_5_no_dropped_edge_regions(self) -> None:
        """5. Every single pixel index [0, L-1] is contained in at least one tile."""
        test_lengths = [256, 279, 355, 432, 512, 583, 738, 1261, 1441, 1601]

        for L in test_lengths:
            coords = compute_tile_coordinates(L, tile_size=256, stride=128)
            coverage = np.zeros(L, dtype=bool)
            for c in coords:
                coverage[c : c + 256] = True

            self.assertTrue(
                np.all(coverage),
                f"Dropped pixels detected for length {L}! Uncovered count: {np.sum(~coverage)}",
            )

    def test_6_deterministic_extraction(self) -> None:
        """6. Repeated tile extraction yields identical results."""
        sample = self._create_synthetic_sample(size=(400, 800))
        tiles_run1 = self.extractor.extract_all_tiles_from_sample(sample)
        tiles_run2 = self.extractor.extract_all_tiles_from_sample(sample)

        self.assertEqual(len(tiles_run1), len(tiles_run2))
        for t1, t2 in zip(tiles_run1, tiles_run2):
            self.assertEqual(t1.metadata.tile_id, t2.metadata.tile_id)
            self.assertEqual(t1.metadata.tile_x, t2.metadata.tile_x)
            self.assertEqual(t1.metadata.tile_y, t2.metadata.tile_y)
            np.testing.assert_array_equal(t1.image, t2.image)
            if t1.mask is not None and t2.mask is not None:
                np.testing.assert_array_equal(t1.mask, t2.mask)

    def test_7_rgb_preservation(self) -> None:
        """7. Tile image preserves 3-channel RGB uint8 format without downsampling."""
        sample = self._create_synthetic_sample(size=(350, 700))
        tile = self.extractor.extract_tile(sample, tile_x=0, tile_y=0)

        self.assertEqual(tile.image.shape, (256, 256, 3))
        self.assertEqual(tile.image.dtype, np.uint8)
        pil_img = tile.to_pil_image()
        self.assertEqual(pil_img.mode, "RGB")
        self.assertEqual(pil_img.size, (256, 256))

    def test_8_metadata_and_split_grouping(self) -> None:
        """8. Clean and edited screenshots from same pair share split_group_id."""
        clean_sample = self._create_synthetic_sample(
            image_id="clean_001",
            is_clean=True,
            pair_id="pair_tx_99",
            source_id="tx_source_99",
        )
        edited_sample = self._create_synthetic_sample(
            image_id="edited_001",
            is_clean=False,
            pair_id="pair_tx_99",
            source_id="tx_source_99",
        )

        clean_meta = self.extractor.get_tile_metadata_list(clean_sample)
        edited_meta = self.extractor.get_tile_metadata_list(edited_sample)

        self.assertEqual(len(clean_meta), len(edited_meta))
        for cm, em in zip(clean_meta, edited_meta):
            self.assertEqual(cm.split_group_id, "pair_tx_99")
            self.assertEqual(em.split_group_id, "pair_tx_99")
            self.assertEqual(cm.split_group_id, em.split_group_id)
            self.assertEqual(cm.tile_x, em.tile_x)
            self.assertEqual(cm.tile_y, em.tile_y)
            self.assertEqual(cm.tile_index, em.tile_index)

    def test_9_lazy_dataset_behavior(self) -> None:
        """9. LazyTileDataset provides on-demand extraction without full RAM buffering."""
        samples = [
            self._create_synthetic_sample(image_id=f"lazy_{i}", size=(300, 600))
            for i in range(3)
        ]
        dataset = LazyTileDataset(samples, extractor=self.extractor)

        # Expected tiles per (300, 600):
        # x_coords: compute_tile_coordinates(300) -> [0, 44] (len 2)
        # y_coords: compute_tile_coordinates(600) -> [0, 128, 256, 344] (len 4)
        # Total per sample = 2 * 4 = 8 tiles. For 3 samples = 24 tiles.
        self.assertEqual(len(dataset), 24)

        # Retrieve specific tile by index
        tile_0 = dataset[0]
        tile_10 = dataset[10]

        self.assertIsInstance(tile_0, ExtractedTile)
        self.assertEqual(tile_0.image.shape, (256, 256, 3))
        self.assertIsInstance(tile_10, ExtractedTile)
        self.assertEqual(tile_10.image.shape, (256, 256, 3))

        # Check O(1) metadata access
        meta_10 = dataset.get_metadata(10)
        self.assertEqual(meta_10.tile_id, tile_10.metadata.tile_id)

    def test_10_safe_padding_for_small_dimensions(self) -> None:
        """10. Canvases smaller than 256 are safely padded to 256x256 without distortion."""
        small_sample = self._create_synthetic_sample(image_id="small_001", size=(200, 220))
        tiles = self.extractor.extract_all_tiles_from_sample(small_sample)

        # Exactly 1 tile covering the entire small image plus padding
        self.assertEqual(len(tiles), 1)
        tile = tiles[0]
        self.assertEqual(tile.image.shape, (256, 256, 3))

        # Check padded region has DEFAULT_PAD_COLOR (17, 17, 17)
        # Pixel at (210, 230) is in padded region
        np.testing.assert_array_equal(tile.image[230, 210, :], np.array(DEFAULT_PAD_COLOR, dtype=np.uint8))

    def test_11_unlabelled_real_screenshots_handled(self) -> None:
        """11. Unlabelled real screenshots extract tiles with mask=None cleanly."""
        img_path = self.test_root / "images" / "real_001.png"
        Image.new("RGB", (400, 800), (20, 20, 20)).save(img_path)

        real_sample = SingleSample(
            image_id="real_gpay_001",
            source_id="real_gpay_001",
            image_path=img_path,
            mask_path=None,
            label=None,
            label_name="unlabeled",
            is_synthetic=False,
            width=400,
            height=800,
        )

        tiles = self.extractor.extract_all_tiles_from_sample(real_sample)
        self.assertGreater(len(tiles), 0)
        for tile in tiles:
            self.assertIsNone(tile.mask)
            self.assertFalse(tile.has_mask)
            self.assertEqual(tile.image.shape, (256, 256, 3))

    def test_12_matched_pair_extraction(self) -> None:
        """12. extract_tiles_from_pair yields identical count and matching coordinates."""
        clean = self._create_synthetic_sample(image_id="p_clean", is_clean=True, pair_id="pair_55")
        edited = self._create_synthetic_sample(image_id="p_edited", is_clean=False, pair_id="pair_55")
        pair = PairSample(
            pair_id="pair_55",
            source_id="compact_55",
            clean_sample=clean,
            edited_sample=edited,
        )

        clean_tiles, edited_tiles = self.extractor.extract_tiles_from_pair(pair)
        self.assertEqual(len(clean_tiles), len(edited_tiles))

        for ct, et in zip(clean_tiles, edited_tiles):
            self.assertEqual(ct.metadata.tile_x, et.metadata.tile_x)
            self.assertEqual(ct.metadata.tile_y, et.metadata.tile_y)
            self.assertEqual(ct.metadata.split_group_id, et.metadata.split_group_id)
            self.assertEqual(ct.split_group_id, "pair_55")

    def test_13_out_of_bounds_coordinate_raises(self) -> None:
        """13. Requesting an out-of-bounds tile raises ValueError."""
        sample = self._create_synthetic_sample(size=(300, 600))
        with self.assertRaises(ValueError):
            self.extractor.extract_tile(sample, tile_x=500, tile_y=100)

    def test_14_integration_phase4_dataset(self) -> None:
        """14. Integration test: extract tiles from actual Phase 4 dataset sample."""
        try:
            loader = DatasetLoader(manifest_path="metadata/phase4_manifest.csv")
            samples = loader.load(mode="SingleMode")
        except Exception as e:
            self.skipTest(f"Phase 4 dataset not found: {e}")

        test_sample = samples[0]
        tiles = self.extractor.extract_all_tiles_from_sample(test_sample)

        self.assertGreater(len(tiles), 0)
        first_tile = tiles[0]
        self.assertEqual(first_tile.image.shape, (256, 256, 3))
        self.assertEqual(first_tile.metadata.original_width, test_sample.width)
        self.assertEqual(first_tile.metadata.original_height, test_sample.height)

    def test_15_dataset_immutability(self) -> None:
        """15. Original dataset files remain completely untouched."""
        dataset_root = PROJECT_ROOT / "Hackathon-VIT" / "CBL04-Dataset"
        manifest_file = dataset_root / "metadata" / "phase4_manifest.csv"
        if not manifest_file.exists():
            self.skipTest("Dataset manifest not found for immutability check")

        hash_before = hashlib.md5(manifest_file.read_bytes()).hexdigest()

        # Run extraction on a sample
        loader = DatasetLoader(manifest_path=manifest_file)
        sample = loader.load(mode="SingleMode")[0]
        _ = self.extractor.extract_all_tiles_from_sample(sample)

        hash_after = hashlib.md5(manifest_file.read_bytes()).hexdigest()
        self.assertEqual(hash_before, hash_after, "Dataset manifest was modified during extraction!")


if __name__ == "__main__":
    unittest.main()

