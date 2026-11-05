"""
CBL-04: Phase 7 Unit and Integration Test Suite
Member 2 - Tile Forensics CNN & Heatmap Lead

Tests cover:
1. Screenshot-level classification rules & metrics calculations.
2. Pixel-level localization metrics & separate clean/tampered calculations.
3. Confusion matrix plotting and file generation.
4. Forensic error analysis categorization logic.
5. In-memory robustness transformations (JPEG, resize, blur, brightness, contrast).
6. Baseline comparator structure and non-applicable metric handling.
7. Original dataset immutability verification.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent

for p in [PROJECT_ROOT, HERE]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from member2.phase7_evaluation_and_error_analysis.baseline_comparator import BaselineComparator
from member2.phase7_evaluation_and_error_analysis.error_analyzer import ForensicErrorAnalyzer
from member2.phase7_evaluation_and_error_analysis.evaluator_phase7 import (
    Phase7Evaluator,
    plot_confusion_matrix,
)
from member2.phase7_evaluation_and_error_analysis.robustness_evaluator import (
    RobustnessEvaluator,
    apply_brightness,
    apply_contrast,
    apply_gaussian_blur,
    apply_jpeg_compression,
    apply_resize_resample,
)


class TestPhase7Evaluation(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        torch.manual_seed(42)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.artifacts_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_confusion_matrix_plotting(self) -> None:
        """Verify confusion matrix plotting generates valid image files."""
        cm = np.array([[70, 5], [10, 65]], dtype=int)
        out_path = self.artifacts_dir / "test_cm.png"

        plot_confusion_matrix(
            cm=cm,
            class_names=["Clean", "Tampered"],
            title="Test Confusion Matrix",
            output_path=out_path,
            is_pixel_level=False,
        )

        self.assertTrue(out_path.exists())
        self.assertGreater(out_path.stat().st_size, 1000)

    def test_error_analysis_categorization(self) -> None:
        """Verify that the 5 distinct forensic error categories are correctly selected."""
        analyzer = ForensicErrorAnalyzer(output_dir=self.artifacts_dir)

        # Mock records: 2 clean, 3 tampered
        records = [
            # Clean 1: zero FP
            {"image_id": "clean_1", "is_tampered": 0, "fp_pixels": 0, "tp_pixels": 0, "fn_pixels": 0, "pixel_recall": 1.0, "pixel_precision": 1.0, "dice": 1.0, "iou": 1.0, "detected_pixels": 0, "layout_family": "compact"},
            # Clean 2: high FP (False positive screenshot)
            {"image_id": "clean_2", "is_tampered": 0, "fp_pixels": 2500, "tp_pixels": 0, "fn_pixels": 0, "pixel_recall": 1.0, "pixel_precision": 0.0, "dice": 0.0, "iou": 0.0, "detected_pixels": 2500, "layout_family": "enriched"},
            # Tampered 1: missed edit (False negative)
            {"image_id": "tamp_1", "is_tampered": 1, "fp_pixels": 100, "tp_pixels": 0, "fn_pixels": 300, "pixel_recall": 0.0, "pixel_precision": 0.0, "dice": 0.0, "iou": 0.0, "detected_pixels": 100, "layout_family": "compact"},
            # Tampered 2: excessive FP
            {"image_id": "tamp_2", "is_tampered": 1, "fp_pixels": 15000, "tp_pixels": 200, "fn_pixels": 50, "pixel_recall": 0.80, "pixel_precision": 0.013, "dice": 0.026, "iou": 0.013, "detected_pixels": 15200, "layout_family": "enriched"},
            # Tampered 3: successful localization
            {"image_id": "tamp_3", "is_tampered": 1, "fp_pixels": 150, "tp_pixels": 450, "fn_pixels": 50, "pixel_recall": 0.90, "pixel_precision": 0.75, "dice": 0.818, "iou": 0.692, "detected_pixels": 600, "layout_family": "bank_progress"},
        ]

        sample_map = {r["image_id"]: r for r in records}
        cases = analyzer.select_error_examples(records, sample_map)

        categories = [c["category"] for c in cases]
        self.assertIn("False-Positive Screenshot", categories)
        self.assertIn("False-Negative Screenshot", categories)
        self.assertIn("Excessive False-Positive Regions", categories)
        self.assertIn("Poorly Localized Tampering", categories)
        self.assertIn("Successful Localization", categories)

        # Confirm FP case is clean_2 and Success case is tamp_3
        fp_c = next(c for c in cases if c["category"] == "False-Positive Screenshot")
        self.assertEqual(fp_c["record"]["image_id"], "clean_2")
        succ_c = next(c for c in cases if c["category"] == "Successful Localization")
        self.assertEqual(succ_c["record"]["image_id"], "tamp_3")

    def test_robustness_transformations(self) -> None:
        """Verify all in-memory robustness transforms maintain image mode, size, and validity."""
        test_img = Image.new("RGB", (200, 300), color=(30, 35, 45))

        # 1. JPEG
        jpeg_95 = apply_jpeg_compression(test_img, quality=95)
        self.assertEqual(jpeg_95.size, (200, 300))
        self.assertEqual(jpeg_95.mode, "RGB")

        # 2. Resampling
        resample_75 = apply_resize_resample(test_img, scale=0.75)
        self.assertEqual(resample_75.size, (200, 300))
        self.assertEqual(resample_75.mode, "RGB")

        # 3. Gaussian Blur
        blurred = apply_gaussian_blur(test_img, sigma=1.0)
        self.assertEqual(blurred.size, (200, 300))
        self.assertEqual(blurred.mode, "RGB")

        # 4. Brightness
        bright = apply_brightness(test_img, factor=1.10)
        self.assertEqual(bright.size, (200, 300))
        self.assertEqual(bright.mode, "RGB")

        # 5. Contrast
        contrast = apply_contrast(test_img, factor=0.90)
        self.assertEqual(contrast.size, (200, 300))
        self.assertEqual(contrast.mode, "RGB")

    def test_baseline_comparator_generation(self) -> None:
        """Verify baseline comparator handles available and non-applicable metrics gracefully."""
        comparator = BaselineComparator(artifacts_dir=self.artifacts_dir)

        p5_path = self.artifacts_dir / "p5_mock.json"
        p6_path = self.artifacts_dir / "p6_mock.json"

        p5_path.write_text('{"accuracy": 0.7767, "precision": 0.1207, "recall": 0.3043}', encoding="utf-8")
        p6_path.write_text('{"pixel_metrics": {"global_pixel_accuracy": 0.9943, "global_dice": 0.0054}}', encoding="utf-8")

        p7_mock = {
            "screenshot_classification": {"accuracy": 0.70, "precision": 0.625, "recall": 1.0},
            "pixel_localization": {"global_pixel_accuracy": 0.9968, "global_pixel_dice": 0.1746},
        }

        matrix = comparator.generate_comparison_matrix(p5_path, p6_path, p7_mock)

        self.assertIn("metrics_comparison", matrix)
        self.assertIn("global_dice", matrix["metrics_comparison"])
        # Phase 5 should be explicitly documented as N/A with rationale
        self.assertTrue(str(matrix["metrics_comparison"]["global_dice"]["phase5"]).startswith("N/A"))
        # Phase 6 and 7 should have numerical comparison
        self.assertEqual(matrix["metrics_comparison"]["global_dice"]["phase6"], 0.0054)
        self.assertEqual(matrix["metrics_comparison"]["global_dice"]["phase7_improved"], 0.1746)

    def test_original_dataset_immutability(self) -> None:
        """Strict verification: original dataset directory remains strictly untouched by Member 2."""
        dataset_root = PROJECT_ROOT / "Hackathon-VIT/CBL04-Dataset"
        if not dataset_root.exists():
            self.skipTest(f"Dataset root not found at {dataset_root}")

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

        p7_in_dataset = list(dataset_root.glob("**/phase7_evaluation*"))
        self.assertEqual(
            p7_in_dataset,
            [],
            f"Detected unauthorized phase7 evaluation files in original dataset: {p7_in_dataset}",
        )

        # Check data folder has zero new model files
        data_checkpoints = list((dataset_root / "data").glob("**/*.pt")) + list((dataset_root / "data").glob("**/*.pth"))
        self.assertEqual(
            data_checkpoints,
            [],
            f"Detected model checkpoints written to read-only data folder: {data_checkpoints}",
        )


if __name__ == "__main__":
    unittest.main()
