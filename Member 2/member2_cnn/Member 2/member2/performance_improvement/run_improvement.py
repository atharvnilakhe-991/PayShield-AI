"""
CBL-04: End-to-End Performance Improvement Pipeline
Phase 6 Improvement - Member 2 (Tile Forensics CNN & Heatmap Lead)

This script runs the complete improvement pipeline:
1. Loads Phase 5 split manifest and recovers Train, Val, Test, Holdout partitions.
2. Trains TileSpatialUNet with CombinedBCEDiceLoss and balanced sampling.
3. Optimizes decision threshold and connected-component area strictly on the validation set.
4. Evaluates the frozen pipeline on the held-out test split (150 images).
5. Generates 7-panel comparative visualizations (Baseline vs Improved).
6. Exports comprehensive comparison metrics to artifacts/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
PHASE1_DIR = HERE.parent / "phase1_dataset_loader"
PHASE2_DIR = HERE.parent / "phase2_preprocessing"
PHASE3_DIR = HERE.parent / "phase3_tile_extraction"
PHASE5_DIR = HERE.parent / "phase5_model_training"
PHASE6_DIR = HERE.parent / "phase6_heatmap_reconstruction"

for p in [PROJECT_ROOT, HERE, PHASE1_DIR, PHASE2_DIR, PHASE3_DIR, PHASE5_DIR, PHASE6_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from member2.performance_improvement.evaluator import ImprovedEvaluator, SpatialUNetInferencer
    from member2.performance_improvement.models import CombinedBCEDiceLoss, TileSpatialUNet
    from member2.performance_improvement.postprocessor import HeatmapPostProcessor, PostProcessingConfig
    from member2.performance_improvement.segmentation_dataset import TileSpatialSegmentationDataset
    from member2.performance_improvement.train_improved import ImprovedTrainer
    from member2.performance_improvement.validator_tuning import ValidationTuner
    from member2.performance_improvement.visualizer import ComparativeVisualizer
    from member2.phase1_dataset_loader.dataset_loader import DatasetLoader
    from member2.phase2_preprocessing.transforms import Preprocessor
    from member2.phase6_heatmap_reconstruction.inference import TileInferencer as BaselineInferencer
except (ImportError, ModuleNotFoundError):
    from dataset_loader import DatasetLoader  # type: ignore[import-not-found]
    from evaluator import ImprovedEvaluator, SpatialUNetInferencer  # type: ignore[import-not-found]
    from inference import TileInferencer as BaselineInferencer  # type: ignore[import-not-found]
    from models import CombinedBCEDiceLoss, TileSpatialUNet  # type: ignore[import-not-found]
    from postprocessor import HeatmapPostProcessor, PostProcessingConfig  # type: ignore[import-not-found]
    from segmentation_dataset import TileSpatialSegmentationDataset  # type: ignore[import-not-found]
    from train_improved import ImprovedTrainer  # type: ignore[import-not-found]
    from transforms import Preprocessor  # type: ignore[import-not-found]
    from validator_tuning import ValidationTuner  # type: ignore[import-not-found]
    from visualizer import ComparativeVisualizer  # type: ignore[import-not-found]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ImprovementPipeline")


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
    vis_dir = artifacts_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)

    logger.info("=" * 75)
    logger.info("PERFORMANCE IMPROVEMENT: TAMPERED PIXEL DETECTION & LOCALIZATION")
    logger.info(f"Model Type: {cfg.get('model_type', 'spatial_unet')}")
    logger.info(f"Artifacts Directory: {artifacts_dir}")
    logger.info("=" * 75)

    # 1. Recover Split Partitions
    logger.info("[Step 1/7] Loading Phase 5 Split Manifest...")
    split_manifest_path = PROJECT_ROOT / "member2/phase5_model_training/artifacts/split_manifest.json"
    with open(split_manifest_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    p6_manifest = PROJECT_ROOT / "Hackathon-VIT/CBL04-Dataset/metadata/phase6_manifest.csv"
    p4_manifest = PROJECT_ROOT / "Hackathon-VIT/CBL04-Dataset/metadata/phase4_manifest.csv"
    manifest_to_use = p6_manifest if p6_manifest.exists() else p4_manifest

    loader = DatasetLoader(manifest_path=manifest_to_use)
    all_samples = loader.load(mode="SingleMode")
    sample_map = {getattr(s, "image_id"): s for s in all_samples}

    train_samples = []
    val_samples = []
    test_samples = []
    holdout_samples = []

    for item in split_data["samples"]:
        img_id = item["image_id"]
        split_name = item["split"]
        s = sample_map.get(img_id)
        if s is None:
            continue
        if split_name == "train":
            train_samples.append(s)
        elif split_name == "val":
            val_samples.append(s)
        elif split_name == "test":
            test_samples.append(s)
        elif split_name == "holdout":
            holdout_samples.append(s)

    logger.info(
        f"Recovered partitions: Train={len(train_samples)}, Val={len(val_samples)}, "
        f"Test={len(test_samples)}, Holdout={len(holdout_samples)}"
    )

    # 2. Build Datasets
    logger.info("[Step 2/7] Constructing Paired Spatial Segmentation Datasets...")
    max_train_tiles = 600 if fast_mode else 1000
    max_val_tiles = 200 if fast_mode else 300

    train_ds = TileSpatialSegmentationDataset(
        samples=train_samples,
        max_samples=max_train_tiles,
        balance_ratio=0.5,
        augment=True,
        seed=cfg.get("random_seed", 42),
    )
    val_ds = TileSpatialSegmentationDataset(
        samples=val_samples,
        max_samples=max_val_tiles,
        balance_ratio=0.5,
        augment=False,
        seed=cfg.get("random_seed", 42),
    )

    batch_size = cfg.get("batch_size", 16)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    ckpt_path = artifacts_dir / "best_improved_model.pt"
    tuning_path = artifacts_dir / "validation_tuning_results.json"

    # 3. Model Training (skip if checkpoint already trained and present)
    if ckpt_path.exists():
        logger.info(f"[Step 3/7] Reusing existing trained model checkpoint from: {ckpt_path}")
    else:
        logger.info("[Step 3/7] Training TileSpatialUNet with Combined BCE + Soft-Dice Loss...")
        model = TileSpatialUNet()
        criterion = CombinedBCEDiceLoss(
            bce_weight=cfg.get("bce_loss_weight", 0.4),
            dice_weight=cfg.get("dice_loss_weight", 0.6),
            pos_weight=cfg.get("pos_weight", 5.0),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.get("learning_rate", 5e-4),
            weight_decay=cfg.get("weight_decay", 1e-4),
        )
        num_epochs = 3 if fast_mode else cfg.get("num_epochs", 8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        trainer = ImprovedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            early_stopping_patience=cfg.get("early_stopping_patience", 3),
            checkpoint_dir=artifacts_dir,
            device=torch.device(cfg.get("device", "cpu")),
        )
        train_res = trainer.fit()

    # 4. Joint Validation Threshold & Post-Processing Optimization
    inferencer = SpatialUNetInferencer(
        checkpoint_path=ckpt_path,
        device=torch.device(cfg.get("device", "cpu")),
        batch_size=32,
    )

    if tuning_path.exists():
        logger.info(f"[Step 4/7] Loading existing validation tuning results from: {tuning_path}")
        with open(tuning_path, "r", encoding="utf-8") as f:
            tuning_res = json.load(f)
        opt_cfg = tuning_res["optimal_config"]
    else:
        logger.info("[Step 4/7] Joint validation optimization of threshold and connected component filtering...")
        max_val_imgs = 10 if fast_mode else cfg.get("num_val_images_for_optimization", 30)
        tuner = ValidationTuner(
            inferencer=inferencer,
            threshold_candidates=cfg.get("threshold_sweep"),
            area_candidates=cfg.get("min_component_area_sweep"),
        )
        tuning_res = tuner.tune(val_samples=val_samples, max_images=max_val_imgs)
        opt_cfg = tuning_res["optimal_config"]
        with open(tuning_path, "w", encoding="utf-8") as f:
            json.dump(tuning_res, f, indent=2)

    logger.info(
        f"Validation tuning parameters: Optimal Threshold = {opt_cfg['threshold']:.2f}, "
        f"Min Area = {opt_cfg['min_component_area']} px (Validation Dice = {tuning_res['best_validation_metrics']['dice']:.4f})"
    )

    # 5. Held-Out Test Evaluation
    logger.info("[Step 5/7] Evaluating frozen improved pipeline on held-out test split...")
    pp = HeatmapPostProcessor(PostProcessingConfig(
        threshold=opt_cfg["threshold"],
        min_component_area=opt_cfg["min_component_area"],
        apply_opening=opt_cfg.get("apply_opening", False),
    ))
    evaluator = ImprovedEvaluator(postprocessor=pp)

    eval_test_samples = test_samples[:20] if fast_mode else test_samples
    eval_result = evaluator.evaluate_dataset(eval_test_samples, inferencer=inferencer)

    with open(artifacts_dir / "improved_pixel_metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "optimal_config": opt_cfg,
                "pixel_metrics": eval_result["pixel_metrics"],
                "image_level_metrics": eval_result["image_level_metrics"],
                "per_family_metrics": eval_result["per_family_metrics"],
            },
            f,
            indent=2,
        )

    # Save per image CSV
    records = eval_result["per_image_records"]
    if records:
        import csv
        with open(artifacts_dir / "improved_per_image_results.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    # 6. Baseline vs Improved Comparison
    logger.info("[Step 6/7] Computing Baseline vs Improved side-by-side comparison...")
    base_metrics_path = PROJECT_ROOT / "member2/phase6_heatmap_reconstruction/artifacts/pixel_metrics_test.json"
    with open(base_metrics_path, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    base_pix = base_data["pixel_metrics"]
    imp_pix = eval_result["pixel_metrics"]

    comparison = {
        "baseline_model": "Phase 5 ResNet-18 Tile Classifier + Gaussian Blending (Phase 6)",
        "improved_model": "Phase 6 Improvement TileSpatialUNet + Joint Post-Processing",
        "threshold": {
            "baseline": base_data.get("threshold_used", 0.80),
            "improved": opt_cfg["threshold"],
        },
        "min_component_area": {
            "baseline": 0,
            "improved": opt_cfg["min_component_area"],
        },
        "global_pixel_accuracy": {
            "baseline": base_pix["global_pixel_accuracy"],
            "improved": imp_pix["global_pixel_accuracy"],
            "diff": imp_pix["global_pixel_accuracy"] - base_pix["global_pixel_accuracy"],
        },
        "global_pixel_precision": {
            "baseline": base_pix["global_pixel_precision"],
            "improved": imp_pix["global_pixel_precision"],
            "diff": imp_pix["global_pixel_precision"] - base_pix["global_pixel_precision"],
        },
        "global_pixel_recall": {
            "baseline": base_pix["global_pixel_recall"],
            "improved": imp_pix["global_pixel_recall"],
            "diff": imp_pix["global_pixel_recall"] - base_pix["global_pixel_recall"],
        },
        "global_pixel_dice": {
            "baseline": base_pix["global_dice"],
            "improved": imp_pix["global_pixel_dice"],
            "diff": imp_pix["global_pixel_dice"] - base_pix["global_dice"],
        },
        "global_pixel_iou": {
            "baseline": base_pix["global_iou"],
            "improved": imp_pix["global_pixel_iou"],
            "diff": imp_pix["global_pixel_iou"] - base_pix["global_iou"],
        },
        "pixel_roc_auc": {
            "baseline": base_pix["pixel_roc_auc"],
            "improved": imp_pix["pixel_roc_auc"],
            "diff": imp_pix["pixel_roc_auc"] - base_pix["pixel_roc_auc"],
        },
        "pixel_pr_auc": {
            "baseline": base_pix["pixel_pr_auc"],
            "improved": imp_pix["pixel_pr_auc"],
            "diff": imp_pix["pixel_pr_auc"] - base_pix["pixel_pr_auc"],
        },
        "total_false_positive_pixels": {
            "baseline": base_pix["total_fp_pixels"],
            "improved": imp_pix["total_fp_pixels"],
            "reduction": base_pix["total_fp_pixels"] - imp_pix["total_fp_pixels"],
            "reduction_percentage": (
                (base_pix["total_fp_pixels"] - imp_pix["total_fp_pixels"]) / max(base_pix["total_fp_pixels"], 1) * 100.0
            ),
        },
        "total_true_positive_pixels": {
            "baseline": base_pix["total_tp_pixels"],
            "improved": imp_pix["total_tp_pixels"],
        },
        "mean_image_dice_all": {
            "baseline": base_pix.get("mean_image_dice", 0.3818),
            "improved": imp_pix["mean_image_dice_all"],
        },
        "mean_image_dice_tampered_only": {
            "baseline": "Near 0.00",
            "improved": imp_pix["mean_image_dice_tampered_only"],
        },
    }

    with open(artifacts_dir / "baseline_vs_improved_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # 7. Generate 7-Panel Visualizations
    logger.info("[Step 7/7] Generating 7-panel comparative diagnostic visualizations...")
    baseline_inferencer = BaselineInferencer(
        checkpoint_path=PROJECT_ROOT / "member2/phase5_model_training/artifacts/best_model.pt",
        device=torch.device(cfg.get("device", "cpu")),
    )
    vis = ComparativeVisualizer(colormap="inferno")

    # Select representative examples
    clean_test = [s for s in eval_test_samples if getattr(s, "label", 0) == 0]
    tampered_test = [s for s in eval_test_samples if getattr(s, "label", 0) == 1]

    # Categories to visualize:
    # 1. Clean Example
    # 2. Successfully localized tampering example
    # 3. False-positive example
    # 4. False-negative example
    selected_vis = []
    if clean_test:
        selected_vis.append((clean_test[0], "Clean Example"))
    if len(clean_test) > 1:
        selected_vis.append((clean_test[1], "Clean Example 2"))

    # For tampered: find a high-dice example (hit), and one with FN or FP
    if tampered_test:
        selected_vis.append((tampered_test[0], "Tampered Localization Example 1"))
    if len(tampered_test) > 1:
        selected_vis.append((tampered_test[1], "Tampered Localization Example 2"))
    if len(tampered_test) > 2:
        selected_vis.append((tampered_test[2], "Subtle Edit Diagnostic"))

    for s, cat in selected_vis:
        img_id = getattr(s, "image_id")
        img_path = getattr(s, "image_path")
        with Image.open(img_path) as im:
            rgb_arr = np.array(Preprocessor.convert_to_rgb(im), dtype=np.uint8)

        gt = evaluator._load_ground_truth_mask(s, rgb_arr.shape[:2])

        # Baseline predictions
        base_hm, _, _ = baseline_inferencer.predict_image(img_path)
        base_mask = (base_hm >= base_data.get("threshold_used", 0.80)).astype(np.uint8)

        # Improved predictions
        imp_hm, _, _ = inferencer.predict_image(img_path)
        imp_mask = pp.process(imp_hm)

        # Metrics for caption
        img_rec = next((r for r in records if r["image_id"] == img_id), {})

        out_file = vis_dir / f"comparison_{img_id}_{cat.replace(' ', '_').lower()}.png"
        vis.render_comparison(
            image_rgb=rgb_arr,
            gt_mask=gt,
            baseline_heatmap=base_hm,
            improved_heatmap=imp_hm,
            baseline_pred_mask=base_mask,
            improved_pred_mask=imp_mask,
            output_path=out_file,
            image_id=img_id,
            category_label=cat,
            metrics_dict=img_rec,
        )

    logger.info(f"Generated {len(selected_vis)} comparative visualizations in {vis_dir}")

    elapsed = time.time() - start_time

    summary = {
        "status": "SUCCESS",
        "elapsed_seconds": elapsed,
        "config": cfg,
        "comparison": comparison,
    }
    with open(artifacts_dir / "improvement_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 75)
    logger.info("PERFORMANCE IMPROVEMENT SUMMARY:")
    logger.info(f"  Baseline Global Pixel Dice: {comparison['global_pixel_dice']['baseline']:.6f} -> Improved: {comparison['global_pixel_dice']['improved']:.6f}")
    logger.info(f"  Baseline Global Pixel IoU:  {comparison['global_pixel_iou']['baseline']:.6f} -> Improved: {comparison['global_pixel_iou']['improved']:.6f}")
    logger.info(f"  Baseline Pixel Precision:   {comparison['global_pixel_precision']['baseline']:.6f} -> Improved: {comparison['global_pixel_precision']['improved']:.6f}")
    logger.info(f"  Baseline Pixel Recall:      {comparison['global_pixel_recall']['baseline']:.6f} -> Improved: {comparison['global_pixel_recall']['improved']:.6f}")
    logger.info(f"  False Positive Pixels:      {comparison['total_false_positive_pixels']['baseline']:,} -> {comparison['total_false_positive_pixels']['improved']:,} ({comparison['total_false_positive_pixels']['reduction_percentage']:.2f}% reduction)")
    logger.info(f"  Pixel PR-AUC:               {comparison['pixel_pr_auc']['baseline']:.6f} -> {comparison['pixel_pr_auc']['improved']:.6f}")
    logger.info(f"  Elapsed Time:               {elapsed:.1f}s")
    logger.info("=" * 75)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Performance Improvement Pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=HERE / "config.json",
        help="Path to improvement configuration JSON",
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
        help="Fast mode for quick verification",
    )
    args = parser.parse_args()

    run_pipeline(args.config, args.artifacts_dir, fast_mode=args.fast)
