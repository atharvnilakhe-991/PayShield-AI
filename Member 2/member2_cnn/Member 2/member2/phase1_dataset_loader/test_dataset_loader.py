"""
Unit and Integration Tests for CBL-04 DatasetLoader & Validation Engine
Phase 1 - Member 2 (Tile Forensics CNN & Heatmap Lead)

Tests all required edge cases using isolated temporary fixtures outside the original dataset:
1. Valid clean/edited pair
2. Missing image file
3. Missing required mask file
4. Missing pair ID
5. Duplicate manifest row
6. Unlabelled sample (real screenshot simulation)
7. Corrupt image file
8. Image/mask dimension mismatch
9. Integration test on actual Phase 4 dataset
10. Integration test on actual Phase 6 dataset
"""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

# Ensure current directory is on sys.path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dataset_loader import DatasetLoader, PairSample, SingleSample


class TestDatasetLoaderFixtures(unittest.TestCase):
    """Isolated test fixtures executed in an independent temporary directory."""

    def setUp(self) -> None:
        # Create an isolated temporary directory outside the dataset
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        (self.root / "masks").mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_image(self, path: Path, size=(100, 200), color=(17, 17, 17)) -> None:
        img = Image.new("RGB", size, color)
        img.save(path, format="PNG")

    def _create_mask(self, path: Path, size=(100, 200), is_clean: bool = True) -> None:
        mask = Image.new("L", size, 0)
        if not is_clean:
            # Add a white altered region in the center
            for x in range(40, 60):
                for y in range(80, 120):
                    mask.putpixel((x, y), 255)
        mask.save(path, format="PNG")

    def _write_manifest(self, rows: list[dict[str, str]], filename: str = "manifest.csv") -> Path:
        p = self.root / filename
        fieldnames = list(rows[0].keys())
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        return p

    def test_1_valid_clean_edited_pair(self) -> None:
        """Test loading and validating a complete, correct clean/edited pair."""
        c_img = self.root / "images/P1_clean.png"
        e_img = self.root / "images/P1_edited.png"
        c_msk = self.root / "masks/P1_clean_mask.png"
        e_msk = self.root / "masks/P1_edited_mask.png"

        self._create_image(c_img, (100, 200))
        self._create_image(e_img, (100, 200))
        self._create_mask(c_msk, (100, 200), is_clean=True)
        self._create_mask(e_msk, (100, 200), is_clean=False)

        manifest = self._write_manifest([
            {
                "image_id": "P1_clean", "source_id": "P1", "pair_id": "P1",
                "image_path": "images/P1_clean.png", "mask_path": "masks/P1_clean_mask.png",
                "label": "0", "app_name": "Google Pay", "theme": "dark", "is_synthetic": "True",
            },
            {
                "image_id": "P1_edited", "source_id": "P1", "pair_id": "P1",
                "image_path": "images/P1_edited.png", "mask_path": "masks/P1_edited_mask.png",
                "label": "1", "app_name": "Google Pay", "theme": "dark", "is_synthetic": "True",
            },
        ])

        loader = DatasetLoader(dataset_root=self.root, manifest_path=manifest)
        res = loader.validate()
        self.assertTrue(res.is_valid, f"Validation failed: {res.errors}")
        self.assertEqual(res.pair_count, 1)
        self.assertEqual(res.broken_pair_count, 0)
        self.assertEqual(res.missing_image_count, 0)
        self.assertEqual(res.missing_mask_count, 0)

        pairs = loader.load(mode="PairMode")
        self.assertEqual(len(pairs), 1)
        self.assertIsInstance(pairs[0], PairSample)
        self.assertEqual(pairs[0].pair_id, "P1")

        # Verify lazy load
        c_loaded, e_loaded, cm_loaded, em_loaded = pairs[0].load_all()
        self.assertEqual(c_loaded.size, (100, 200))
        self.assertEqual(cm_loaded.getextrema(), (0, 0))
        self.assertEqual(em_loaded.getextrema(), (0, 255))

    def test_2_missing_image(self) -> None:
        """Test reporting of missing image file."""
        manifest = self._write_manifest([
            {
                "image_id": "P2_clean", "source_id": "P2", "pair_id": "P2",
                "image_path": "images/non_existent.png", "mask_path": "",
                "label": "0", "app_name": "Google Pay", "theme": "dark",
            }
        ])
        loader = DatasetLoader(dataset_root=self.root, manifest_path=manifest)
        res = loader.validate()
        self.assertFalse(res.is_valid)
        self.assertEqual(res.missing_image_count, 1)
        self.assertTrue(any("Missing image file" in err for err in res.errors))

    def test_3_missing_required_mask(self) -> None:
        """Test reporting of missing required mask for an edited image."""
        e_img = self.root / "images/P3_edited.png"
        self._create_image(e_img, (100, 200))

        # mask_path points to non-existent file
        manifest = self._write_manifest([
            {
                "image_id": "P3_edited", "source_id": "P3", "pair_id": "P3",
                "image_path": "images/P3_edited.png", "mask_path": "masks/missing_mask.png",
                "label": "1", "app_name": "Google Pay", "theme": "dark",
            }
        ])
        loader = DatasetLoader(dataset_root=self.root, manifest_path=manifest)
        res = loader.validate()
        self.assertFalse(res.is_valid)
        self.assertEqual(res.missing_mask_count, 1)
        self.assertTrue(any("Missing mask file" in err for err in res.errors))

    def test_4_missing_pair_id(self) -> None:
        """Test handling and warnings when pair ID is missing for clean/edited samples."""
        c_img = self.root / "images/P4_clean.png"
        self._create_image(c_img, (100, 200))

        manifest = self._write_manifest([
            {
                "image_id": "P4_clean", "source_id": "", "pair_id": "",
                "image_path": "images/P4_clean.png", "mask_path": "",
                "label": "clean", "app_name": "Google Pay", "theme": "dark",
            }
        ])
        loader = DatasetLoader(dataset_root=self.root, manifest_path=manifest)
        res = loader.validate()
        self.assertTrue(any("has label 'clean' but no pair_id" in w for w in res.warnings))
        pairs = loader.load(mode="PairMode")
        self.assertEqual(len(pairs), 0)

    def test_5_duplicate_manifest_row(self) -> None:
        """Test detection and warning on duplicate manifest entries."""
        c_img = self.root / "images/P5_clean.png"
        self._create_image(c_img, (100, 200))

        manifest = self._write_manifest([
            {
                "image_id": "P5_clean", "source_id": "P5", "pair_id": "P5",
                "image_path": "images/P5_clean.png", "mask_path": "",
                "label": "0", "app_name": "Google Pay", "theme": "dark",
            },
            {
                "image_id": "P5_clean", "source_id": "P5", "pair_id": "P5",
                "image_path": "images/P5_clean.png", "mask_path": "",
                "label": "0", "app_name": "Google Pay", "theme": "dark",
            },
        ])
        loader = DatasetLoader(dataset_root=self.root, manifest_path=manifest)
        res = loader.validate()
        self.assertEqual(res.duplicate_entry_count, 1)
        self.assertEqual(res.duplicate_path_count, 1)
        self.assertTrue(any("Duplicate image_id 'P5_clean'" in w for w in res.warnings))

    def test_6_unlabelled_sample(self) -> None:
        """Test explicit handling of unlabelled samples without artificial label inference."""
        r_img = self.root / "images/REAL_001.jpg"
        self._create_image(r_img, (100, 200))

        manifest = self._write_manifest([
            {
                "image_id": "REAL_001", "source_id": "REAL_001", "pair_id": "",
                "image_path": "images/REAL_001.jpg", "mask_path": "",
                "label": "unlabeled", "app_name": "Google Pay", "theme": "dark", "is_synthetic": "False",
            }
        ])
        loader = DatasetLoader(dataset_root=self.root, manifest_path=manifest)
        res = loader.validate()
        self.assertTrue(res.is_valid)
        self.assertEqual(res.unlabeled_sample_count, 1)

        singles = loader.load(mode="SingleMode")
        self.assertEqual(len(singles), 1)
        self.assertIsNone(singles[0].label)
        self.assertEqual(singles[0].label_name, "unlabeled")
        self.assertFalse(singles[0].is_synthetic)

        # In PairMode, unlabelled single sample is not grouped into a pair
        pairs = loader.load(mode="PairMode")
        self.assertEqual(len(pairs), 0)

    def test_7_corrupt_image(self) -> None:
        """Test reporting of corrupt / unreadable image file."""
        corrupt_img = self.root / "images/corrupt.png"
        corrupt_img.write_bytes(b"INVALID_HEADER_GARBAGE_BYTES_12345")

        manifest = self._write_manifest([
            {
                "image_id": "CORRUPT_01", "source_id": "C1", "pair_id": "C1",
                "image_path": "images/corrupt.png", "mask_path": "",
                "label": "0", "app_name": "Google Pay", "theme": "dark",
            }
        ])
        loader = DatasetLoader(dataset_root=self.root, manifest_path=manifest)
        res = loader.validate()
        self.assertFalse(res.is_valid)
        self.assertEqual(res.corrupt_file_count, 1)
        self.assertTrue(any("Corrupt image file" in err for err in res.errors))

    def test_8_dimension_mismatch(self) -> None:
        """Test detection of dimension mismatch between screenshot and mask."""
        c_img = self.root / "images/mismatch_clean.png"
        c_msk = self.root / "masks/mismatch_mask.png"
        self._create_image(c_img, size=(100, 200))
        self._create_mask(c_msk, size=(80, 150))  # Different dimension

        manifest = self._write_manifest([
            {
                "image_id": "MISMATCH_01", "source_id": "M1", "pair_id": "M1",
                "image_path": "images/mismatch_clean.png", "mask_path": "masks/mismatch_mask.png",
                "label": "clean", "app_name": "Google Pay", "theme": "dark",
            }
        ])
        loader = DatasetLoader(dataset_root=self.root, manifest_path=manifest)
        res = loader.validate()
        self.assertFalse(res.is_valid)
        self.assertEqual(res.dimension_mismatch_count, 1)
        self.assertTrue(any("Dimension mismatch" in err for err in res.errors))


class TestActualDatasetIntegration(unittest.TestCase):
    """Integration test against the actual CBL-04 dataset repository."""

    def setUp(self) -> None:
        self.dataset_root = Path(r"c:\Users\athar\OneDrive\Desktop\Member 2\Hackathon-VIT\CBL04-Dataset")
        if not self.dataset_root.exists():
            self.skipTest(f"Actual dataset directory not found at {self.dataset_root}")

    def test_phase4_bulk_manifest(self) -> None:
        """Test full validation and loading on actual Phase 4 bulk manifest."""
        loader = DatasetLoader(
            dataset_root=self.dataset_root,
            manifest_path="metadata/phase4_manifest.csv",
        )
        res = loader.validate()
        self.assertTrue(res.is_valid, f"Phase 4 validation failed: {res.errors}")
        self.assertEqual(res.manifest_row_count, 1000)
        self.assertEqual(res.unique_image_count, 1000)
        self.assertEqual(res.clean_sample_count, 500)
        self.assertEqual(res.edited_sample_count, 500)
        self.assertEqual(res.pair_count, 500)
        self.assertEqual(res.broken_pair_count, 0)
        self.assertEqual(res.missing_image_count, 0)
        self.assertEqual(res.missing_mask_count, 0)
        self.assertEqual(res.corrupt_file_count, 0)
        self.assertEqual(res.dimension_mismatch_count, 0)

        pairs = loader.load(mode="PairMode")
        self.assertEqual(len(pairs), 500)

        singles = loader.load(mode="SingleMode")
        self.assertEqual(len(singles), 1000)

        # Check group mapping for split safety
        groups = loader.get_groups()
        self.assertEqual(len(groups), 500)
        for grp_id, members in groups.items():
            self.assertEqual(len(members), 2, f"Group {grp_id} does not have exactly 2 members")

    def test_phase6_combined_manifest(self) -> None:
        """Test validation and loading on Phase 6 manifest (including real holdouts)."""
        loader = DatasetLoader(
            dataset_root=self.dataset_root,
            manifest_path="metadata/phase6_manifest.csv",
        )
        res = loader.validate()
        self.assertTrue(res.is_valid, f"Phase 6 validation failed: {res.errors}")
        self.assertEqual(res.manifest_row_count, 1387)
        self.assertEqual(res.clean_sample_count, 500)
        self.assertEqual(res.edited_sample_count, 500)
        self.assertEqual(res.unlabeled_sample_count, 387)
        self.assertEqual(res.pair_count, 500)
        self.assertEqual(res.broken_pair_count, 0)

        singles = loader.load(mode="SingleMode")
        self.assertEqual(len(singles), 1387)
        unlabeled = [s for s in singles if s.label_name == "unlabeled"]
        self.assertEqual(len(unlabeled), 387)
        for u in unlabeled:
            self.assertIsNone(u.label)
            self.assertFalse(u.is_synthetic)


if __name__ == "__main__":
    unittest.main(verbosity=2)

