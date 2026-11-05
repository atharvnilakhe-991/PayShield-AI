"""
CBL-04: Phase 6 Heatmap Reconstruction & Evaluation Orchestrator
Phase 6 - Member 2 (Tile Forensics CNN & Heatmap Lead)

This script executes the complete Phase 6 pipeline:
1. Loads Phase 5 model checkpoint and split manifest.
2. Selects optimal decision threshold strictly on the validation set.
3. Evaluates full-resolution reconstructed heatmaps on the held-out test split.
4. Generates multi-panel visualizations for synthetic test and real holdout screenshots.
5. Saves all forensic metrics and artifacts to member2/phase6_heatmap_reconstruction/artifacts/.
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

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
PHASE1_DIR = HERE.parent / "phase1_dataset_loader"
PHASE2_DIR = HERE.parent / "phase2_preprocessing"
PHASE3_DIR = HERE.parent / "phase3_tile_extraction"
PHASE5_DIR = HERE.parent / "phase5_model_training"

for p in [PROJECT_ROOT, HERE, PHASE1_DIR, PHASE2_DIR, PHASE3_DIR, PHASE5_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from member2.phase1_dataset_loader.dataset_loader import DatasetLoader
    from member2.phase2_preprocessing.transforms import Preprocessor
    from member2.phase6_heatmap_reconstruction.evaluator import Phase6Evaluator
    from member2.phase6_heatmap_reconstruction.inference import TileInferencer
    from member2.phase6_heatmap_reconstruction.reconstructor import HeatmapReconstructor
    from member2.phase6_heatmap_reconstruction.thresholding import ThresholdOptimizer
    from member2.phase6_heatmap_reconstruction.visualizer import Phase6Visualizer
except (ImportError, ModuleNotFoundError):
    from dataset_loader import DatasetLoader  # type: ignore[import-not-found]
    from evaluator import Phase6Evaluator  # type: ignore[import-not-found]
    from inference import TileInferencer  # type: ignore[import-not-found]
    from reconstructor import HeatmapReconstructor  # type: ignore[import-not-found]
    from thresholding import ThresholdOptimizer  # type: ignore[import-not-found]
    from transforms import Preprocessor  # type: ignore[import-not-found]
    from visualizer import Phase6Visualizer  # type: ignore[import-not-found]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Phase6Orchestrator")


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_phase6_pipeline(
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
    logger.info("PHASE 6: FULL-RESOLUTION HEATMAP RECONSTRUCTION & LOCALIZATION")
    logger.info(f"Model Checkpoint: {cfg['checkpoint_path']}")
    logger.info(f"Window Type: {cfg.get('window_type', 'gaussian')}")
    logger.info(f"Artifacts Directory: {artifacts_dir}")
    logger.info("=" * 75)

    # 1. Load Dataset & Phase 5 Splits
    logger.info("[Step 1/5] Loading Dataset and Phase 5 Split Manifest...")
    split_manifest_path = PROJECT_ROOT / "member2/phase5_model_training/artifacts/split_manifest.json"
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"Phase 5 split manifest not found at: {split_manifest_path}")

    with open(split_manifest_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    # Load dataset via Phase 1 loader
    p6_manifest = PROJECT_ROOT / "Hackathon-VIT/CBL04-Dataset/metadata/phase6_manifest.csv"
    p4_manifest = PROJECT_ROOT / "Hackathon-VIT/CBL04-Dataset/metadata/phase4_manifest.csv"
    manifest_to_use = p6_manifest if p6_manifest.exists() else p4_manifest

    loader = DatasetLoader(manifest_path=manifest_to_use)
    all_samples = loader.load(mode="SingleMode")
    sample_map = {getattr(s, "image_id"): s for s in all_samples}

    # Group samples by split
    val_samples: List[Any] = []
    test_samples: List[Any] = []
    holdout_samples: List[Any] = []

    for item in split_data["samples"]:
        img_id = item["image_id"]
        split_name = item["split"]
        s = sample_map.get(img_id)
        if s is None:
            continue
        if split_name == "val":
            val_samples.append(s)
        elif split_name == "test":
            test_samples.append(s)
        elif split_name == "holdout":
            holdout_samples.append(s)

    logger.info(
        f"Recovered split partitions: Validation={len(val_samples)}, "
        f"Test={len(test_samples)}, Holdout={len(holdout_samples)}"
    )

    # 2. Initialize Inferencer & Reconstructor
    logger.info("[Step 2/5] Initializing Heatmap Reconstructor & Model Inferencer...")
    reconstructor = HeatmapReconstructor(
        tile_size=cfg.get("tile_size", 256),
        window_type=cfg.get("window_type", "gaussian"),
        gaussian_sigma=cfg.get("gaussian_sigma", 64.0),
        hann_min_weight=cfg.get("hann_min_weight", 0.05),
    )

    ckpt_path = PROJECT_ROOT / cfg["checkpoint_path"]
    inferencer = TileInferencer(
        checkpoint_path=ckpt_path,
        device=torch.device(cfg.get("device", "cpu")),
        batch_size=cfg.get("batch_size", 32),
        reconstructor=reconstructor,
    )

    # 3. Validation-Only Threshold Optimization
    logger.info("[Step 3/5] Performing validation-only threshold optimization...")
    optimizer = ThresholdOptimizer(
        inferencer=inferencer,
        threshold_candidates=cfg.get("threshold_sweep"),
        default_threshold=cfg.get("default_threshold", 0.50),
    )

    max_val = 10 if fast_mode else cfg.get("num_val_images_for_threshold", 30)
    opt_res = optimizer.optimize_threshold(
        val_samples=val_samples,
        max_images=max_val,
        random_seed=cfg.get("random_seed", 42),
    )

    # Save threshold optimization results
    opt_path = artifacts_dir / "threshold_optimization.json"
    with open(opt_path, "w", encoding="utf-8") as f:
        json.dump(opt_res.to_dict(), f, indent=2)

    selected_threshold = opt_res.optimal_threshold
    logger.info(
        f"Validation optimization complete. Optimal threshold: {selected_threshold:.2f} "
        f"(Val Dice={opt_res.best_dice:.4f}, Val IoU={opt_res.best_iou:.4f})"
    )

    # 4. Held-Out Test Set Evaluation
    logger.info("[Step 4/5] Evaluating reconstructed heatmaps on held-out test split...")
    evaluator = Phase6Evaluator(threshold=selected_threshold)

    eval_test_samples = test_samples[:20] if fast_mode else test_samples
    eval_result = evaluator.evaluate_test_set(
        test_samples=eval_test_samples,
        inferencer=inferencer,
        top_k=cfg.get("image_level_top_k", 3),
    )

    # Save metrics artifacts
    evaluator.save_results(eval_result, artifacts_dir)

    # 5. Visual Output Artifacts Generation
    logger.info("[Step 5/5] Generating multi-panel visualization artifacts...")
    visualizer = Phase6Visualizer(colormap="inferno", overlay_alpha=0.5)

    # Render a representative subset of synthetic test images (clean and tampered)
    test_clean = [s for s in eval_test_samples if getattr(s, "label", 0) == 0]
    test_tampered = [s for s in eval_test_samples if getattr(s, "label", 0) == 1]

    vis_selected = test_clean[:3] + test_tampered[:5]
    for s in vis_selected:
        img_id = getattr(s, "image_id", "sample")
        img_path = getattr(s, "image_path")
        with Image.open(img_path) as im:
            rgb_arr = np.array(Preprocessor.convert_to_rgb(im), dtype=np.uint8)

        heatmap, _, (h, w) = inferencer.predict_image(img_path)
        gt_mask = evaluator._load_ground_truth_mask(s, (h, w))
        pred_mask = (heatmap >= selected_threshold).astype(np.uint8)

        # Look up per-image metrics
        img_rec = next((r.to_dict() for r in eval_result.per_image_records if r.image_id == img_id), {})
        vis_file = vis_dir / f"synthetic_{img_id}_diagnosis.png"
        visualizer.render_synthetic_evaluation(
            image_rgb=rgb_arr,
            gt_mask=gt_mask,
            heatmap=heatmap,
            pred_mask=pred_mask,
            output_path=vis_file,
            image_id=img_id,
            threshold=selected_threshold,
            metrics=img_rec,
        )

    logger.info(f"Generated {len(vis_selected)} synthetic test diagnostic figures in {vis_dir}")

    # Render exploratory heatmaps for real holdout screenshots
    vis_holdouts = holdout_samples[:4]
    for s in vis_holdouts:
        img_id = getattr(s, "image_id", "holdout")
        img_path = getattr(s, "image_path")
        with Image.open(img_path) as im:
            rgb_arr = np.array(Preprocessor.convert_to_rgb(im), dtype=np.uint8)

        heatmap, _, (h, w) = inferencer.predict_image(img_path)
        pred_mask = (heatmap >= selected_threshold).astype(np.uint8)

        vis_file = vis_dir / f"holdout_{img_id}_exploratory.png"
        visualizer.render_real_holdout_exploratory(
            image_rgb=rgb_arr,
            heatmap=heatmap,
            pred_mask=pred_mask,
            output_path=vis_file,
            image_id=img_id,
            threshold=selected_threshold,
        )

    logger.info(f"Generated {len(vis_holdouts)} exploratory real holdout figures in {vis_dir}")

    elapsed = time.time() - start_time

    # Save consolidated summary
    summary_path = artifacts_dir / "phase6_summary.json"
    summary_payload = {
        "status": "SUCCESS",
        "elapsed_seconds": elapsed,
        "config": cfg,
        "threshold_optimization": opt_res.to_dict(),
        "pixel_metrics": eval_result.pixel_metrics,
        "image_level_metrics": eval_result.image_level_metrics,
        "per_family_metrics": eval_result.per_family_metrics,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    logger.info("=" * 75)
    logger.info("PHASE 6 EVALUATION SUMMARY (SYNTHETIC HELD-OUT TEST SPLIT):")
    logger.info(f"  Evaluated Screenshots:  {eval_result.total_images_evaluated} ({eval_result.clean_images_count} clean, {eval_result.tampered_images_count} tampered)")
    logger.info(f"  Decision Threshold:     {selected_threshold:.2f} (Validation-optimized)")
    logger.info(f"  Global Pixel Accuracy:  {eval_result.pixel_metrics['global_pixel_accuracy'] * 100:.2f}%")
    logger.info(f"  Global Pixel Precision: {eval_result.pixel_metrics['global_pixel_precision']:.4f}")
    logger.info(f"  Global Pixel Recall:    {eval_result.pixel_metrics['global_pixel_recall']:.4f}")
    logger.info(f"  Global Pixel Dice (F1): {eval_result.pixel_metrics['global_dice']:.4f}")
    logger.info(f"  Global Pixel IoU:       {eval_result.pixel_metrics['global_iou']:.4f}")
    logger.info(f"  Pixel ROC-AUC:          {eval_result.pixel_metrics['pixel_roc_auc']:.4f}")
    logger.info(f"  Pixel PR-AUC:           {eval_result.pixel_metrics['pixel_pr_auc']:.4f}")
    logger.info(f"  Mean Image-Level Dice:  {eval_result.pixel_metrics['mean_image_dice']:.4f}")
    logger.info(f"  Image-Level ROC-AUC:    {eval_result.image_level_metrics['roc_auc']:.4f}")
    logger.info(f"  Elapsed Time:           {elapsed:.1f}s")
    logger.info("=" * 75)

    return summary_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 6 Heatmap Reconstruction & Evaluation")
    parser.add_argument(
        "--config",
        type=Path,
        default=HERE / "heatmap_config.json",
        help="Path to heatmap configuration JSON",
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
        help="Run in fast mode with reduced image counts for quick verification",
    )
    args = parser.parse_args()

    run_phase6_pipeline(args.config, args.artifacts_dir, fast_mode=args.fast)

