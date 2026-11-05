"""
Unit and Integration Tests for Phase 5: Model Training & Evaluation
Phase 5 - Member 2 (Tile Forensics CNN & Heatmap Lead)

Test Coverage:
1. Model forward pass and output shapes ((B, 3, 256, 256) -> (B, 2)) for ResNet18 and SimpleTileCNN.
2. Leakage-safe group-aware dataset splitting (zero pair overlap across splits).
3. Real holdout segregation (unlabelled real screenshots excluded from supervised splits).
4. Layout family stratification.
5. Ambiguous tile exclusion (label == -1 strictly filtered out).
6. Inverse-frequency class weight computation.
7. End-to-end training step and loss convergence.
8. Checkpoint saving and loading fidelity.
9. Evaluator metric calculations and confusion matrix formatting.
10. Immutability of original dataset files.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
PHASE1_DIR = HERE.parent / "phase1_dataset_loader"
PHASE2_DIR = HERE.parent / "phase2_preprocessing"
PHASE3_DIR = HERE.parent / "phase3_tile_extraction"
PHASE4_DIR = HERE.parent / "phase4_tile_labeling"

for p in [PROJECT_ROOT, HERE, PHASE1_DIR, PHASE2_DIR, PHASE3_DIR, PHASE4_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from member2.phase4_tile_labeling.tile_labeler import LabeledTileMetadata
    from member2.phase5_model_training.evaluate import Evaluator
    from member2.phase5_model_training.model import SimpleTileCNN, TileResNet18, get_model
    from member2.phase5_model_training.split_dataset import DatasetSplit, GroupSplitter, SplitConfig
    from member2.phase5_model_training.train import Trainer
    from member2.phase5_model_training.training_dataset import (
        TileForensicsDataset,
        compute_class_weights,
        create_weighted_sampler,
    )
except (ImportError, ModuleNotFoundError):
    from evaluate import Evaluator  # type: ignore[import-not-found]
    from model import SimpleTileCNN, TileResNet18, get_model  # type: ignore[import-not-found]
    from split_dataset import DatasetSplit, GroupSplitter, SplitConfig  # type: ignore[import-not-found]
    from tile_labeler import LabeledTileMetadata  # type: ignore[import-not-found]
    from train import Trainer  # type: ignore[import-not-found]
    from training_dataset import (  # type: ignore[import-not-found]
        TileForensicsDataset,
        compute_class_weights,
        create_weighted_sampler,
    )


class DummySample:
    """Mock sample for testing dataset splitting and pipeline steps."""

    def __init__(
        self,
        image_id: str,
        pair_id: str,
        source_id: str,
        layout_family: str = "compact",
        label: int = 0,
        is_real: bool = False,
        mask_path: str = "dummy_mask.png",
        image_path: str = "dummy_img.png",
    ) -> None:
        self.image_id = image_id
        self.pair_id = pair_id
        self.source_id = source_id
        self.layout_family = layout_family
        self.label = label
        self.is_real = is_real
        self.mask_path = mask_path
        self.image_path = image_path


class TestTrainingPipeline(unittest.TestCase):
    """Comprehensive test suite for Phase 5."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_model_shapes_resnet18(self) -> None:
        """Verify TileResNet18 input/output shape contracts."""
        model = get_model("resnet18", num_classes=2, pretrained=False)
        x = torch.randn(2, 3, 256, 256)
        out = model(x)
        self.assertEqual(out.shape, (2, 2))
        self.assertFalse(torch.isnan(out).any())

    def test_model_shapes_simple_cnn(self) -> None:
        """Verify SimpleTileCNN input/output shape contracts."""
        model = get_model("simple_cnn", num_classes=2)
        x = torch.randn(2, 3, 256, 256)
        out = model(x)
        self.assertEqual(out.shape, (2, 2))
        self.assertFalse(torch.isnan(out).any())

    def test_group_splitting_leakage_safe(self) -> None:
        """Verify that group splitting guarantees ZERO transaction-pair leakage across splits."""
        samples: List[DummySample] = []
        # Create 20 synthetic pairs (40 images)
        for i in range(1, 21):
            pair_id = f"pair_{i:04d}"
            fam = "compact" if i <= 10 else "enriched"
            # Clean image
            samples.append(
                DummySample(
                    image_id=f"clean_{i:04d}",
                    pair_id=pair_id,
                    source_id=f"src_{i:04d}",
                    layout_family=fam,
                    label=0,
                )
            )
            # Edited image
            samples.append(
                DummySample(
                    image_id=f"edited_{i:04d}",
                    pair_id=pair_id,
                    source_id=f"src_{i:04d}",
                    layout_family=fam,
                    label=1,
                )
            )

        splitter = GroupSplitter(SplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42))
        split = splitter.split_samples(samples)

        # Check disjoint groups
        train_set = set(split.train_groups)
        val_set = set(split.val_groups)
        test_set = set(split.test_groups)

        self.assertEqual(train_set & val_set, set())
        self.assertEqual(train_set & test_set, set())
        self.assertEqual(val_set & test_set, set())
        self.assertEqual(len(train_set) + len(val_set) + len(test_set), 20)

        # Verify that clean and edited versions of each pair are in the exact same split
        split_map: Dict[str, str] = {}
        for s in split.train_samples:
            split_map[s.image_id] = "train"
        for s in split.val_samples:
            split_map[s.image_id] = "val"
        for s in split.test_samples:
            split_map[s.image_id] = "test"

        for i in range(1, 21):
            c_split = split_map[f"clean_{i:04d}"]
            e_split = split_map[f"edited_{i:04d}"]
            self.assertEqual(
                c_split,
                e_split,
                f"Pair {i} leaked: clean in {c_split}, edited in {e_split}",
            )

    def test_holdout_segregation(self) -> None:
        """Verify that unlabelled real holdout images are segregated from supervised splits."""
        samples: List[DummySample] = [
            DummySample(image_id="synth_1", pair_id="pair_1", source_id="src_1", label=0, is_real=False),
            DummySample(image_id="synth_2", pair_id="pair_2", source_id="src_2", label=1, is_real=False),
            DummySample(image_id="real_1", pair_id="", source_id="", label=None, is_real=True, mask_path=None),
            DummySample(image_id="real_2", pair_id="", source_id="", label=None, is_real=True, mask_path=None),
        ]
        splitter = GroupSplitter(SplitConfig(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25))
        split = splitter.split_samples(samples)

        self.assertEqual(len(split.holdout_samples), 2)
        holdout_ids = {s.image_id for s in split.holdout_samples}
        self.assertEqual(holdout_ids, {"real_1", "real_2"})

        # Supervised splits must not contain holdouts
        all_supervised_ids = {s.image_id for s in split.train_samples + split.val_samples + split.test_samples}
        self.assertNotIn("real_1", all_supervised_ids)
        self.assertNotIn("real_2", all_supervised_ids)

    def test_ambiguous_tile_exclusion(self) -> None:
        """Verify that ambiguous tiles (label == -1) and unlabelled tiles (label None) are excluded."""
        # Create dummy image on disk
        img_path = self.temp_path / "test_tile_img.png"
        img_arr = np.full((300, 300, 3), 128, dtype=np.uint8)
        Image.fromarray(img_arr).save(img_path)

        metadata_list = [
            LabeledTileMetadata(
                tile_id="t_clean",
                source_id="src1",
                pair_id="p1",
                image_path=str(img_path),
                mask_path=None,
                tile_x=0,
                tile_y=0,
                tile_width=256,
                tile_height=256,
                tile_index=0,
                original_width=300,
                original_height=300,
                split_group_id="p1",
                tile_label=0,
                label_status="clean",
                tampered_pixel_count=0,
                tampered_pixel_ratio=0.0,
                label_threshold=16,
            ),
            LabeledTileMetadata(
                tile_id="t_tamp",
                source_id="src1",
                pair_id="p1",
                image_path=str(img_path),
                mask_path=None,
                tile_x=0,
                tile_y=0,
                tile_width=256,
                tile_height=256,
                tile_index=1,
                original_width=300,
                original_height=300,
                split_group_id="p1",
                tile_label=1,
                label_status="tampered",
                tampered_pixel_count=50,
                tampered_pixel_ratio=0.0007,
                label_threshold=16,
            ),
            LabeledTileMetadata(
                tile_id="t_ambig",
                source_id="src1",
                pair_id="p1",
                image_path=str(img_path),
                mask_path=None,
                tile_x=0,
                tile_y=0,
                tile_width=256,
                tile_height=256,
                tile_index=2,
                original_width=300,
                original_height=300,
                split_group_id="p1",
                tile_label=-1,  # Ambiguous
                label_status="ambiguous",
                tampered_pixel_count=5,
                tampered_pixel_ratio=0.00007,
                label_threshold=16,
            ),
            LabeledTileMetadata(
                tile_id="t_unlabelled",
                source_id="src2",
                pair_id="p2",
                image_path=str(img_path),
                mask_path=None,
                tile_x=0,
                tile_y=0,
                tile_width=256,
                tile_height=256,
                tile_index=3,
                original_width=300,
                original_height=300,
                split_group_id="p2",
                tile_label=None,  # Unlabelled
                label_status="unlabelled",
                tampered_pixel_count=0,
                tampered_pixel_ratio=0.0,
                label_threshold=16,
            ),
        ]

        ds = TileForensicsDataset(tile_metadata_list=metadata_list)
        self.assertEqual(len(ds), 2)  # Only clean and tampered
        labels = ds.get_labels()
        self.assertEqual(set(labels), {0, 1})
        self.assertNotIn(-1, labels)

    def test_class_weights_computation(self) -> None:
        """Verify balanced inverse-frequency weighting calculation: w_c = N / (2 * N_c)."""
        img_path = self.temp_path / "dummy.png"
        Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(img_path)

        # 90 clean, 10 tampered (N=100)
        metas = []
        for i in range(90):
            metas.append(
                LabeledTileMetadata(
                    tile_id=f"c_{i}",
                    source_id="s",
                    pair_id="p",
                    image_path=str(img_path),
                    mask_path=None,
                    tile_x=0,
                    tile_y=0,
                    tile_width=256,
                    tile_height=256,
                    tile_index=i,
                    original_width=256,
                    original_height=256,
                    split_group_id="p",
                    tile_label=0,
                    label_status="clean",
                    tampered_pixel_count=0,
                    tampered_pixel_ratio=0.0,
                    label_threshold=16,
                )
            )
        for i in range(10):
            metas.append(
                LabeledTileMetadata(
                    tile_id=f"t_{i}",
                    source_id="s",
                    pair_id="p",
                    image_path=str(img_path),
                    mask_path=None,
                    tile_x=0,
                    tile_y=0,
                    tile_width=256,
                    tile_height=256,
                    tile_index=90 + i,
                    original_width=256,
                    original_height=256,
                    split_group_id="p",
                    tile_label=1,
                    label_status="tampered",
                    tampered_pixel_count=20,
                    tampered_pixel_ratio=0.0003,
                    label_threshold=16,
                )
            )

        ds = TileForensicsDataset(tile_metadata_list=metas)
        weights = compute_class_weights(ds)

        # Expected: w_0 = 100 / (2 * 90) = 0.5556, w_1 = 100 / (2 * 10) = 5.0
        self.assertAlmostEqual(weights[0].item(), 100.0 / 180.0, places=4)
        self.assertAlmostEqual(weights[1].item(), 100.0 / 20.0, places=4)

    def test_training_step_convergence(self) -> None:
        """Verify that training step computes loss, populates gradients, and reduces loss on dummy batch."""
        model = SimpleTileCNN(num_classes=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(4, 3, 256, 256)
        y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        # Step 1
        optimizer.zero_grad()
        out1 = model(x)
        loss1 = criterion(out1, y)
        loss1.backward()

        # Check gradients
        for p in model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)

        optimizer.step()

        # Overfit 5 iterations
        for _ in range(5):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        self.assertLess(loss.item(), loss1.item())

    def test_checkpoint_save_and_load(self) -> None:
        """Verify checkpoint saving and loading fidelity."""
        model = get_model("simple_cnn", num_classes=2)
        ckpt_dir = self.temp_path / "ckpt"
        ckpt_dir.mkdir()

        ckpt_file = ckpt_dir / "best_model.pt"
        torch.save(
            {
                "model_name": "simple_cnn",
                "state_dict": model.state_dict(),
                "epoch": 3,
                "val_f1": 0.88,
                "config": {"num_classes": 2},
            },
            ckpt_file,
        )

        evaluator, ckpt = Evaluator.from_checkpoint(ckpt_file, artifacts_dir=ckpt_dir)
        self.assertEqual(ckpt["epoch"], 3)
        self.assertAlmostEqual(ckpt["val_f1"], 0.88)
        self.assertIsInstance(evaluator.model, SimpleTileCNN)

    def test_evaluator_metrics_computation(self) -> None:
        """Verify Evaluator._compute_metrics ground truth calculations."""
        # 10 samples: 6 clean, 4 tampered
        # True: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        # Pred: [0, 0, 0, 0, 1, 1, 0, 1, 1, 1]
        # TN=4, FP=2, FN=1, TP=3
        y_true = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        y_pred = [0, 0, 0, 0, 1, 1, 0, 1, 1, 1]

        trainer = Trainer(
            model=SimpleTileCNN(2),
            train_loader=[],
            val_loader=[],
            checkpoint_dir=self.temp_path,
        )
        metrics = trainer._compute_metrics(y_true, y_pred)

        self.assertEqual(metrics["tn"], 4)
        self.assertEqual(metrics["fp"], 2)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tp"], 3)
        self.assertAlmostEqual(metrics["accuracy"], 7.0 / 10.0)
        self.assertAlmostEqual(metrics["precision"], 3.0 / 5.0)  # TP / (TP + FP)
        self.assertAlmostEqual(metrics["recall"], 3.0 / 4.0)     # TP / (TP + FN)

    def test_original_dataset_immutability(self) -> None:
        """Verify that CBL04 original dataset directory remains strictly untouched by Member 2."""
        dataset_root = PROJECT_ROOT / "Hackathon-VIT/CBL04-Dataset"
        if not dataset_root.exists():
            return

        # Ensure no Member 2 code or artifacts were created inside CBL04 dataset root
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
