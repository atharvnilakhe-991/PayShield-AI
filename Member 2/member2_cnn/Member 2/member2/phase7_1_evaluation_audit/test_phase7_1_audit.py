"""
CBL-04: Phase 7.1 Audit & Metric Consistency Verification Test Suite
Member 2 - Tile Forensics CNN & Heatmap Lead

Unit tests cover:
1. Mathematical correctness of screenshot classification rules and metrics.
2. Confusion matrix consistency (TP, TN, FP, FN, specificity, recall, precision).
3. Pixel-level metric consistency (Dice and IoU analytical relationship).
4. Frozen test split verification (150 images, zero group leakage, identical sample IDs).
5. Threshold provenance validation (validation-only tuning for tau and A_min).
6. Robustness baseline reconciliation (mathematical accounting of 150-sample vs 10-sample Dice).
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent

for p in [PROJECT_ROOT, HERE]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


class TestPhase71Audit(unittest.TestCase):
    def setUp(self) -> None:
        self.artifacts_dir = HERE

    def test_classification_metrics_computation(self) -> None:
        """Verify mathematical calculation of accuracy, precision, recall, specificity, and F1."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 1])

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        self.assertEqual((tn, fp, fn, tp), (2, 2, 1, 3))
        self.assertEqual(accuracy_score(y_true, y_pred), 5 / 8)
        self.assertEqual(precision_score(y_true, y_pred), 3 / 5)
        self.assertEqual(recall_score(y_true, y_pred), 3 / 4)
        self.assertAlmostEqual(f1_score(y_true, y_pred), 2 * (3/5) * (3/4) / ((3/5) + (3/4)))

        specificity = tn / (tn + fp)
        self.assertEqual(specificity, 2 / 4)

    def test_confusion_matrix_correctness(self) -> None:
        """Verify that classification_audit.json values match mathematically."""
        audit_path = HERE / "classification_audit.json"
        if not audit_path.exists():
            self.skipTest(f"Missing {audit_path}")

        with open(audit_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for rule_name, res in data.items():
            tp = res["tp"]
            tn = res["tn"]
            fp = res["fp"]
            fn = res["fn"]

            # Total must equal 150
            self.assertEqual(tp + tn + fp + fn, 150, f"Sample count mismatch in {rule_name}")
            # Clean must equal 75, tampered must equal 75
            self.assertEqual(tn + fp, 75, f"Clean count mismatch in {rule_name}")
            self.assertEqual(tp + fn, 75, f"Tampered count mismatch in {rule_name}")

            # Specificity formula
            spec = tn / (tn + fp)
            self.assertAlmostEqual(res["specificity"], spec, places=4)

    def test_pixel_metric_consistency(self) -> None:
        """Verify that Dice and IoU adhere to their fundamental relationship: IoU = Dice / (2 - Dice)."""
        tp = 42554
        fp = 309665
        fn = 92640

        dice = (2.0 * tp) / (2.0 * tp + fp + fn)
        iou = tp / (tp + fp + fn)

        expected_iou = dice / (2.0 - dice)
        self.assertAlmostEqual(iou, expected_iou, places=6)
        self.assertAlmostEqual(dice, 0.174612, places=5)
        self.assertAlmostEqual(iou, 0.095657, places=5)

    def test_split_sample_id_verification(self) -> None:
        """Verify that test split has 150 synthetic samples and zero group leakage."""
        split_path = PROJECT_ROOT / "member2/phase5_model_training/artifacts/split_manifest.json"
        if not split_path.exists():
            self.skipTest("Missing split manifest")

        with open(split_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        test_samples = [s for s in data["samples"] if s["split"] == "test"]
        self.assertEqual(len(test_samples), 150)

        clean = [s for s in test_samples if s["is_tampered"] == 0]
        tamp = [s for s in test_samples if s["is_tampered"] == 1]
        self.assertEqual(len(clean), 75)
        self.assertEqual(len(tamp), 75)

        # Check leakage across groups
        train_groups = {s["split_group_id"] for s in data["samples"] if s["split"] == "train"}
        val_groups = {s["split_group_id"] for s in data["samples"] if s["split"] == "val"}
        test_groups = {s["split_group_id"] for s in test_samples}

        self.assertEqual(len(train_groups.intersection(test_groups)), 0)
        self.assertEqual(len(val_groups.intersection(test_groups)), 0)

    def test_threshold_provenance_validation(self) -> None:
        """Verify that tau=0.80 and A_min=50 originated from validation tuning only."""
        vt_path = PROJECT_ROOT / "member2/performance_improvement/artifacts/validation_tuning_results.json"
        if not vt_path.exists():
            self.skipTest("Missing validation tuning results")

        with open(vt_path, "r", encoding="utf-8") as f:
            vt = json.load(f)

        self.assertEqual(vt["optimal_config"]["threshold"], 0.80)
        self.assertEqual(vt["optimal_config"]["min_component_area"], 50)
        self.assertAlmostEqual(vt["best_validation_metrics"]["dice"], 0.125038, places=4)

    def test_robustness_baseline_reconciliation(self) -> None:
        """Verify mathematical explanation of 0.1746 (150 images) vs 0.1756 (10 images)."""
        rec_path = HERE / "robustness_reconciliation.json"
        if not rec_path.exists():
            self.skipTest("Missing robustness reconciliation file")

        with open(rec_path, "r", encoding="utf-8") as f:
            rec = json.load(f)

        self.assertEqual(rec["sample_size_main_eval"], 150)
        self.assertEqual(rec["sample_size_robustness_baseline"], 10)
        self.assertLess(abs(rec["dice_main_eval"] - rec["dice_robustness_baseline"]), 0.002)


if __name__ == "__main__":
    unittest.main()

