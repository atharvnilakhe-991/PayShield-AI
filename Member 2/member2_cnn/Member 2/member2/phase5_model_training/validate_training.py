"""
CBL-04: Phase 5 Training & Validation Orchestration Engine
Phase 5 - Member 2 (Tile Forensics CNN & Heatmap Lead)

This script runs the complete Phase 5 pipeline end-to-end:
1. Loads Phase 1 dataset manifest (synthetic pairs + holdout).
2. Executes leakage-safe group splitting (70% train, 15% val, 15% test).
3. Verifies zero pair leakage and saves split manifests.
4. Initializes TileForensicsDataset for train, val, and test splits (excluding ambiguous tiles).
5. Computes training-only balanced class weights.
6. Trains the tile forensics CNN with AdamW and early stopping.
7. Evaluates the trained model on unseen test split.
8. Computes accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, and per-family metrics.
9. Compares against the ~50.5% global downsampled baseline.
10. Saves all artifacts to member2/phase5_model_training/artifacts/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

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
    from member2.phase1_dataset_loader.dataset_loader import DatasetLoader
    from member2.phase5_model_training.evaluate import Evaluator
    from member2.phase5_model_training.model import get_model
    from member2.phase5_model_training.split_dataset import GroupSplitter, SplitConfig
    from member2.phase5_model_training.train import Trainer
    from member2.phase5_model_training.training_dataset import (
        TileForensicsDataset,
        compute_class_weights,
        create_weighted_sampler,
    )
except (ImportError, ModuleNotFoundError):
    from dataset_loader import DatasetLoader  # type: ignore[import-not-found]
    from evaluate import Evaluator  # type: ignore[import-not-found]
    from model import get_model  # type: ignore[import-not-found]
    from split_dataset import GroupSplitter, SplitConfig  # type: ignore[import-not-found]
    from train import Trainer  # type: ignore[import-not-found]
    from training_dataset import (  # type: ignore[import-not-found]
        TileForensicsDataset,
        compute_class_weights,
        create_weighted_sampler,
    )

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Phase5Validator")


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(
    config_path: Path,
    artifacts_dir: Path,
    fast_mode: bool = False,
) -> Dict[str, Any]:
    start_time = time.time()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path)

    logger.info("=" * 70)
    logger.info("PHASE 5: TILE FORENSICS CNN TRAINING & BASELINE REPRODUCTION")
    logger.info(f"Model Architecture: {cfg.get('model_architecture', 'resnet18')}")
    logger.info(f"Artifacts Directory: {artifacts_dir}")
    logger.info("=" * 70)

    # 1. Load dataset via Phase 1 loader
    logger.info("[Step 1/6] Loading Phase 1 dataset...")
    p6_manifest = PROJECT_ROOT / "Hackathon-VIT/CBL04-Dataset/metadata/phase6_manifest.csv"
    p4_manifest = PROJECT_ROOT / "Hackathon-VIT/CBL04-Dataset/metadata/phase4_manifest.csv"
    manifest_to_use = p6_manifest if p6_manifest.exists() else p4_manifest
    logger.info(f"Using manifest: {manifest_to_use}")

    loader = DatasetLoader(manifest_path=manifest_to_use)
    samples = loader.load(mode="SingleMode")
    logger.info(f"Loaded {len(samples)} total samples from manifest.")

    # 2. Group-Aware Splitting
    logger.info("[Step 2/6] Performing group-aware dataset splitting...")
    split_cfg = SplitConfig(
        train_ratio=cfg["split_proportions"]["train"],
        val_ratio=cfg["split_proportions"]["val"],
        test_ratio=cfg["split_proportions"]["test"],
        random_seed=cfg.get("random_seed", 42),
        stratify_by_family=True,
    )
    splitter = GroupSplitter(split_cfg)
    split = splitter.split_samples(samples)

    assert split.verify_no_leakage(), "CRITICAL: Split leakage verification failed!"
    logger.info(
        f"Splits verified (zero leakage): Train={len(split.train_groups)} groups ({len(split.train_samples)} samples), "
        f"Val={len(split.val_groups)} groups ({len(split.val_samples)} samples), "
        f"Test={len(split.test_groups)} groups ({len(split.test_samples)} samples), "
        f"Holdout={len(split.holdout_samples)} samples."
    )

    # Save split manifests
    splitter.save_split_manifests(split, artifacts_dir)

    # 3. Create Datasets
    logger.info("[Step 3/6] Indexing tiles and constructing PyTorch Datasets...")
    max_train = 1000 if fast_mode else cfg.get("max_train_samples_per_epoch", 2000)
    max_val = 300 if fast_mode else cfg.get("max_val_samples", 600)
    max_test = 300 if fast_mode else cfg.get("max_test_samples", 600)

    train_ds = TileForensicsDataset(
        samples=split.train_samples,
        normalization=cfg.get("normalization", "imagenet"),
        max_samples=max_train,
        seed=cfg.get("random_seed", 42),
    )
    val_ds = TileForensicsDataset(
        samples=split.val_samples,
        normalization=cfg.get("normalization", "imagenet"),
        max_samples=max_val,
        seed=cfg.get("random_seed", 42),
    )
    test_ds = TileForensicsDataset(
        samples=split.test_samples,
        normalization=cfg.get("normalization", "imagenet"),
        max_samples=max_test,
        seed=cfg.get("random_seed", 42),
    )

    logger.info(
        f"Dataset items: Train={len(train_ds)} (clean={train_ds.num_clean}, tampered={train_ds.num_tampered}), "
        f"Val={len(val_ds)} (clean={val_ds.num_clean}, tampered={val_ds.num_tampered}), "
        f"Test={len(test_ds)} (clean={test_ds.num_clean}, tampered={test_ds.num_tampered})"
    )

    # 4. Class Imbalance Handling
    logger.info("[Step 4/6] Handling class imbalance...")
    class_weights = compute_class_weights(train_ds)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    batch_size = cfg.get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 5. Model Initialization & Training
    logger.info("[Step 5/6] Initializing model and starting training...")
    arch = cfg.get("model_architecture", "resnet18")
    model = get_model(
        model_name=arch,
        num_classes=cfg.get("num_classes", 2),
        pretrained=cfg.get("pretrained", False),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("learning_rate", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    num_epochs = 3 if fast_mode else cfg.get("num_epochs", 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        early_stopping_patience=cfg.get("early_stopping_patience", 3),
        early_stopping_min_delta=cfg.get("early_stopping_min_delta", 0.001),
        early_stopping_metric=cfg.get("early_stopping_metric", "val_f1"),
        checkpoint_dir=artifacts_dir,
        model_name=arch,
        config_dict=cfg,
    )
    training_res = trainer.fit()
    logger.info(
        f"Training complete. Best epoch: {training_res.best_epoch}, "
        f"Best Val F1: {training_res.best_val_f1:.4f}, Best Val Acc: {training_res.best_val_accuracy:.4f}"
    )

    # 6. Evaluation on Test Split
    logger.info("[Step 6/6] Evaluating best model on held-out test split...")
    best_ckpt = artifacts_dir / "best_model.pt"
    evaluator, _ = Evaluator.from_checkpoint(best_ckpt, artifacts_dir=artifacts_dir)
    eval_res = evaluator.evaluate(test_loader, save_artifacts=True)

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("EVALUATION RESULTS (HELD-OUT TEST SET):")
    logger.info(f"  Test Accuracy:     {eval_res.accuracy * 100:.2f}%  (Baseline: 50.46%)")
    logger.info(f"  Tampered Precision:{eval_res.precision:.4f}  (Baseline: 0.5043)")
    logger.info(f"  Tampered Recall:   {eval_res.recall:.4f}  (Baseline: 0.5461)")
    logger.info(f"  Tampered F1:       {eval_res.f1:.4f}  (Baseline: 0.5243)")
    logger.info(f"  ROC-AUC:           {eval_res.roc_auc:.4f}")
    logger.info(f"  PR-AUC:            {eval_res.pr_auc:.4f}")
    logger.info(f"  Confusion Matrix:  TN={eval_res.confusion_matrix['tn']}, FP={eval_res.confusion_matrix['fp']}, FN={eval_res.confusion_matrix['fn']}, TP={eval_res.confusion_matrix['tp']}")
    logger.info(f"  False Positive Rt: {eval_res.false_positive_rate * 100:.2f}%")
    logger.info(f"  False Negative Rt: {eval_res.false_negative_rate * 100:.2f}%")
    logger.info(f"  Elapsed Time:      {elapsed:.1f}s")
    logger.info("=" * 70)

    # Save summary report payload
    summary_path = artifacts_dir / "training_summary.json"
    summary_data = {
        "status": "SUCCESS",
        "elapsed_seconds": elapsed,
        "config": cfg,
        "training": training_res.to_dict(),
        "evaluation": eval_res.to_dict(),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 5 Model Training & Evaluation")
    parser.add_argument(
        "--config",
        type=Path,
        default=HERE / "training_config.json",
        help="Path to training config JSON",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=HERE / "artifacts",
        help="Path to artifacts output directory",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run in fast mode with reduced epochs and subset sampling for quick testing",
    )
    args = parser.parse_args()

    run_pipeline(args.config, args.artifacts_dir, fast_mode=args.fast)
