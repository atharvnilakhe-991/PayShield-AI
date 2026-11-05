"""
Unit and Integration Tests for CBL-04 Native-Resolution Preprocessing
Phase 2 - Member 2 (Tile Forensics CNN & Heatmap Lead)

Comprehensive 16-point test suite:
1. Native-resolution dimensions are preserved.
2. RGB conversion works correctly.
3. Grayscale conversion works correctly.
4. RGBA conversion follows documented dark-theme alpha compositing policy.
5. Image-mask dimensions remain aligned.
6. Binary masks retain only background (0) and foreground (1) values.
7. Clean masks remain all-black / all-background.
8. Edited masks retain non-zero foreground pixels.
9. Missing masks are not silently replaced.
10. Unlabelled samples return mask=None.
11. JPEG compression does not modify original files on disk.
12. Gaussian blur does not modify masks.
13. Random transformations are reproducible with a fixed seed.
14. Pair processing preserves geometric alignment and identical parameters.
15. Invalid image/mask dimensions produce a clear error.
16. Original dataset files remain completely untouched and immutable.
"""

from __future__ import annotations

import csv
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

for p in [PROJECT_ROOT, HERE, PHASE1_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from member2.phase1_dataset_loader.dataset_loader import DatasetLoader, PairSample, SingleSample
    from member2.phase2_preprocessing.transforms import PreprocessConfig, Preprocessor, ProcessedPair, ProcessedSample
except (ImportError, ModuleNotFoundError):
    from dataset_loader import DatasetLoader, PairSample, SingleSample  # type: ignore[import-not-found]
    from transforms import PreprocessConfig, Preprocessor, ProcessedPair, ProcessedSample  # type: ignore[import-not-found]


class TestTransforms(unittest.TestCase):
    """Test suite executing isolated tests and actual dataset validation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_root = Path(self.temp_dir.name)
        (self.test_root / "images").mkdir(parents=True, exist_ok=True)
        (self.test_root / "masks").mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_image(self, path: Path, size=(300, 600), mode="RGB", color=(20, 20, 20)) -> None:
        img = Image.new(mode, size, color)
        img.save(path, format="PNG")

    def _create_mask(self, path: Path, size=(300, 600), is_clean: bool = True) -> None:
        mask = Image.new("L", size, 0)
        if not is_clean:
            for x in range(50, 100):
                for y in range(100, 150):
                    mask.putpixel((x, y), 255)
        mask.save(path, format="PNG")

    def test_1_native_resolution_dimensions_preserved(self) -> None:
        """1. Native-resolution dimensions are preserved."""
        test_sizes = [(540, 1600), (738, 1601), (353, 1280), (270, 1248)]
        preprocessor = Preprocessor(PreprocessConfig(preserve_native_resolution=True))

        for w, h in test_sizes:
            img = Image.new("RGB", (w, h), (18, 18, 18))
            processed = preprocessor.process_sample(
                sample=SingleSample(
                    image_id=f"test_{w}_{h}",
                    source_id="test",
                    image_path=self.test_root / f"test_{w}_{h}.png",
                ),
                image_override=img,
            )
            self.assertEqual(processed.processed_size, (w, h))
            self.assertEqual(processed.original_size, (w, h))
            self.assertEqual(processed.image.shape, (h, w, 3))

    def test_2_rgb_conversion(self) -> None:
        """2. RGB conversion works correctly."""
        img = Image.new("RGB", (200, 400), (34, 157, 190))
        preprocessor = Preprocessor()
        processed = preprocessor.process_sample(
            sample=SingleSample(image_id="rgb_test", source_id="s1", image_path=self.test_root / "rgb.png"),
            image_override=img,
        )
        self.assertEqual(processed.image_pil.mode, "RGB")
        self.assertEqual(processed.image.shape, (400, 200, 3))
        self.assertEqual(processed.image.dtype, np.uint8)

    def test_3_grayscale_conversion(self) -> None:
        """3. Grayscale conversion works correctly (replicates channel across R, G, B)."""
        gray_img = Image.new("L", (150, 300), 128)
        preprocessor = Preprocessor()
        processed = preprocessor.process_sample(
            sample=SingleSample(image_id="gray_test", source_id="s2", image_path=self.test_root / "gray.png"),
            image_override=gray_img,
        )
        self.assertEqual(processed.image_pil.mode, "RGB")
        arr = processed.image
        self.assertEqual(arr.shape, (300, 150, 3))
        # Ensure R == G == B
        np.testing.assert_array_equal(arr[:, :, 0], arr[:, :, 1])
        np.testing.assert_array_equal(arr[:, :, 1], arr[:, :, 2])
        self.assertEqual(arr[0, 0, 0], 128)

    def test_4_rgba_conversion_follows_alpha_policy(self) -> None:
        """4. RGBA conversion follows the documented dark-theme alpha compositing policy."""
        # Create an RGBA image with fully transparent pixels
        rgba = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        # Add a red square with 50% opacity in the center
        for x in range(25, 75):
            for y in range(25, 75):
                rgba.putpixel((x, y), (255, 0, 0, 128))

        preprocessor = Preprocessor(PreprocessConfig(alpha_fill_color=(17, 17, 17)))
        processed = preprocessor.process_sample(
            sample=SingleSample(image_id="rgba_test", source_id="s3", image_path=self.test_root / "rgba.png"),
            image_override=rgba,
        )
        self.assertEqual(processed.image_pil.mode, "RGB")
        arr = processed.image
        # Background pixel (transparent) should be filled with alpha_fill_color (17, 17, 17)
        self.assertEqual(tuple(arr[0, 0]), (17, 17, 17))
        # Partially transparent pixel should be blended, not pure red
        blended = arr[50, 50]
        self.assertGreater(blended[0], 17)
        self.assertLess(blended[0], 255)

    def test_5_image_mask_dimensions_remain_aligned(self) -> None:
        """5. Image-mask dimensions remain aligned."""
        img = Image.new("RGB", (256, 512), (18, 18, 18))
        mask = Image.new("L", (256, 512), 0)
        preprocessor = Preprocessor()
        processed = preprocessor.process_sample(
            sample=SingleSample(image_id="align_test", source_id="s4", image_path=self.test_root / "a.png"),
            image_override=img,
            mask_override=mask,
        )
        self.assertEqual(processed.image.shape[:2], (512, 256))
        self.assertEqual(processed.mask.shape, (512, 256))
        self.assertEqual(processed.processed_size, (256, 512))

    def test_6_binary_masks_retain_only_background_and_foreground(self) -> None:
        """6. Binary masks retain only background (0) and foreground (1) values."""
        mask = Image.new("L", (100, 100), 0)
        mask.putpixel((10, 10), 127)  # Intermediate anti-aliasing value
        mask.putpixel((20, 20), 255)  # Full foreground

        preprocessor = Preprocessor(PreprocessConfig(mask_foreground_value=1))
        processed = preprocessor.process_sample(
            sample=SingleSample(image_id="bin_test", source_id="s5", image_path=self.test_root / "b.png"),
            image_override=Image.new("RGB", (100, 100)),
            mask_override=mask,
        )
        unique_vals = set(np.unique(processed.mask))
        self.assertTrue(unique_vals.issubset({0, 1}))
        self.assertEqual(processed.mask[10, 10], 1)
        self.assertEqual(processed.mask[20, 20], 1)
        self.assertEqual(processed.mask[0, 0], 0)

    def test_7_clean_masks_remain_all_black(self) -> None:
        """7. Clean masks remain all-black / all-background."""
        clean_mask = Image.new("L", (200, 300), 0)
        preprocessor = Preprocessor()
        processed = preprocessor.process_sample(
            sample=SingleSample(image_id="clean_test", source_id="s6", label=0, label_name="clean", image_path=self.test_root / "c.png"),
            image_override=Image.new("RGB", (200, 300)),
            mask_override=clean_mask,
        )
        self.assertEqual(np.sum(processed.mask), 0)
        self.assertEqual(processed.tampered_pixel_count, 0)
        self.assertEqual(processed.tampered_pixel_ratio, 0.0)

    def test_8_edited_masks_retain_non_zero_foreground_pixels(self) -> None:
        """8. Edited masks retain non-zero foreground pixels."""
        edited_mask = Image.new("L", (200, 300), 0)
        # Add 50 tampered pixels
        for i in range(50):
            edited_mask.putpixel((i, 10), 255)

        preprocessor = Preprocessor()
        processed = preprocessor.process_sample(
            sample=SingleSample(image_id="edited_test", source_id="s7", label=1, label_name="edited", image_path=self.test_root / "e.png"),
            image_override=Image.new("RGB", (200, 300)),
            mask_override=edited_mask,
        )
        self.assertEqual(processed.tampered_pixel_count, 50)
        self.assertAlmostEqual(processed.tampered_pixel_ratio, 50 / (200 * 300))

    def test_9_missing_masks_are_not_silently_replaced(self) -> None:
        """9. Missing masks are not silently replaced with fake zeros."""
        sample = SingleSample(
            image_id="missing_mask_test",
            source_id="s8",
            label=1,
            label_name="edited",
            image_path=self.test_root / "img.png",
            mask_path=self.test_root / "non_existent_mask.png",
        )
        self._create_image(sample.image_path)
        preprocessor = Preprocessor()
        # Loading sample with missing mask file should raise FileNotFoundError, not fabricate zeros
        with self.assertRaises(FileNotFoundError):
            preprocessor.process_sample(sample)

    def test_10_unlabelled_samples_return_mask_none(self) -> None:
        """10. Unlabelled samples return mask=None."""
        sample = SingleSample(
            image_id="real_test",
            source_id="real_s",
            label=None,
            label_name="unlabeled",
            is_synthetic=False,
            image_path=self.test_root / "real.jpg",
            mask_path=None,
        )
        self._create_image(sample.image_path)
        preprocessor = Preprocessor()
        processed = preprocessor.process_sample(sample)
        self.assertIsNone(processed.mask)
        self.assertIsNone(processed.mask_pil)
        self.assertFalse(processed.has_mask)
        self.assertEqual(processed.sample_type, "real_sensitive")

    def test_11_jpeg_compression_does_not_modify_original_file(self) -> None:
        """11. JPEG compression does not modify the original file on disk."""
        img_p = self.test_root / "images/jpeg_test.png"
        self._create_image(img_p, (200, 200))
        orig_bytes = img_p.read_bytes()
        orig_sha = hashlib.sha256(orig_bytes).hexdigest()

        preprocessor = Preprocessor(PreprocessConfig(jpeg_quality=80))
        sample = SingleSample(image_id="j_test", source_id="s9", image_path=img_p)
        processed = preprocessor.process_sample(sample)

        # Processed image exists and is modified in memory
        self.assertEqual(processed.processed_size, (200, 200))
        # Original file on disk must be byte-for-byte identical
        new_bytes = img_p.read_bytes()
        self.assertEqual(hashlib.sha256(new_bytes).hexdigest(), orig_sha)

    def test_12_gaussian_blur_does_not_modify_masks(self) -> None:
        """12. Gaussian blur applies to image data only and never modifies masks."""
        mask = Image.new("L", (100, 100), 0)
        mask.putpixel((50, 50), 255)
        raw_mask_arr = np.array(mask)

        preprocessor = Preprocessor(PreprocessConfig(gaussian_sigma=0.5))
        processed = preprocessor.process_sample(
            sample=SingleSample(image_id="blur_test", source_id="s10", image_path=self.test_root / "b.png"),
            image_override=Image.new("RGB", (100, 100), (50, 50, 50)),
            mask_override=mask,
        )
        # Mask must retain sharp binary values without blur/smoothing
        self.assertEqual(processed.mask[50, 50], 1)
        self.assertEqual(processed.mask[50, 51], 0)
        self.assertEqual(processed.mask[49, 50], 0)
        self.assertEqual(processed.tampered_pixel_count, 1)

    def test_13_random_transformations_reproducible_with_seed(self) -> None:
        """13. Random transformations are reproducible with a fixed seed."""
        cfg1 = PreprocessConfig(
            jpeg_quality=(75, 95),
            gaussian_sigma=(0.2, 0.6),
            brightness_factor=(0.95, 1.05),
            contrast_factor=(0.95, 1.05),
            seed=20261103,
        )
        cfg2 = PreprocessConfig(
            jpeg_quality=(75, 95),
            gaussian_sigma=(0.2, 0.6),
            brightness_factor=(0.95, 1.05),
            contrast_factor=(0.95, 1.05),
            seed=20261103,
        )
        cfg_different = PreprocessConfig(
            jpeg_quality=(75, 95),
            gaussian_sigma=(0.2, 0.6),
            brightness_factor=(0.95, 1.05),
            contrast_factor=(0.95, 1.05),
            seed=99999999,
        )

        img = Image.new("RGB", (150, 150), (120, 80, 40))
        p1 = Preprocessor(cfg1).process_sample(SingleSample("s1", "s", image_path=None), image_override=img)
        p2 = Preprocessor(cfg2).process_sample(SingleSample("s2", "s", image_path=None), image_override=img)
        p_diff = Preprocessor(cfg_different).process_sample(SingleSample("s3", "s", image_path=None), image_override=img)

        # Same seed must produce identical arrays
        np.testing.assert_array_equal(p1.image, p2.image)
        # Different seed must produce different arrays
        self.assertFalse(np.array_equal(p1.image, p_diff.image))

    def test_14_pair_processing_preserves_geometric_alignment(self) -> None:
        """14. Pair processing preserves geometric alignment and identical parameters."""
        c_path = self.test_root / "images/pair_c.png"
        e_path = self.test_root / "images/pair_e.png"
        cm_path = self.test_root / "masks/pair_cm.png"
        em_path = self.test_root / "masks/pair_em.png"

        self._create_image(c_path, (200, 400), color=(18, 18, 18))
        self._create_image(e_path, (200, 400), color=(18, 18, 18))
        self._create_mask(cm_path, (200, 400), is_clean=True)
        self._create_mask(em_path, (200, 400), is_clean=False)

        clean_sample = SingleSample("c_id", "src1", image_path=c_path, mask_path=cm_path, label=0, label_name="clean")
        edited_sample = SingleSample("e_id", "src1", image_path=e_path, mask_path=em_path, label=1, label_name="edited")
        pair = PairSample(pair_id="src1", source_id="src1", clean_sample=clean_sample, edited_sample=edited_sample)

        # Configure randomized robustness transformations
        cfg = PreprocessConfig(jpeg_quality=(75, 95), gaussian_sigma=(0.2, 0.6), seed=42)
        preprocessor = Preprocessor(cfg)

        processed_pair = preprocessor.process_pair(pair)
        # Check alignment
        self.assertTrue(processed_pair.verify_alignment())
        self.assertEqual(processed_pair.clean.processed_size, (200, 400))
        self.assertEqual(processed_pair.edited.processed_size, (200, 400))
        # Check that both clean and edited received identical transformation parameters
        self.assertEqual(
            processed_pair.clean.metadata["transform_params"],
            processed_pair.edited.metadata["transform_params"],
        )

    def test_15_invalid_dimensions_produce_error(self) -> None:
        """15. Invalid image/mask dimensions produce a clear error."""
        img = Image.new("RGB", (200, 400))
        mismatch_mask = Image.new("L", (180, 350))  # Different dimension

        preprocessor = Preprocessor()
        sample = SingleSample("mismatch", "s", image_path=None)
        with self.assertRaises(ValueError) as ctx:
            preprocessor.process_sample(sample, image_override=img, mask_override=mismatch_mask)
        self.assertIn("Dimension mismatch", str(ctx.exception))

    def test_16_original_dataset_immutability(self) -> None:
        """16. Original dataset files remain completely unchanged when processed."""
        dataset_root = Path(r"c:\Users\athar\OneDrive\Desktop\Member 2\Hackathon-VIT\CBL04-Dataset")
        if not dataset_root.exists():
            self.skipTest("Actual dataset root not found.")

        # Load samples from Phase 1 loader
        loader = DatasetLoader(dataset_root=dataset_root)
        pairs = loader.load(mode="PairMode")
        self.assertGreater(len(pairs), 0)

        # Inspect the first pair's files on disk
        first_pair = pairs[0]
        c_path = first_pair.clean_image_path
        c_mtime_before = c_path.stat().st_mtime
        c_hash_before = hashlib.sha256(c_path.read_bytes()).hexdigest()

        # Run preprocessing with JPEG and blur transforms
        cfg = PreprocessConfig(jpeg_quality=85, gaussian_sigma=0.3)
        preprocessor = Preprocessor(cfg)
        proc_pair = preprocessor.process_pair(first_pair)

        # Verify output
        self.assertTrue(proc_pair.verify_alignment())
        self.assertEqual(proc_pair.clean.processed_size, first_pair.clean_sample.load_image().size)

        # Verify disk files remain untouched
        c_mtime_after = c_path.stat().st_mtime
        c_hash_after = hashlib.sha256(c_path.read_bytes()).hexdigest()
        self.assertEqual(c_mtime_before, c_mtime_after)
        self.assertEqual(c_hash_before, c_hash_after)


if __name__ == "__main__":
    unittest.main(verbosity=2)
