"""
CBL-04: Master Orchestrator for Phase 7 Evaluation & Error Analysis
Member 2 - Tile Forensics CNN & Heatmap Lead

This script executes the complete Phase 7 evaluation pipeline:
1. Audit of dataset partitions and model checkpoints.
2. Screenshot-level classification evaluation with explicit rules.
3. Pixel-level localization evaluation with separate clean/tampered metrics.
4. Generation of screenshot and pixel confusion matrices.
5. In-depth error analysis across 5 forensic failure and success categories.
6. Multi-condition robustness testing under controlled perturbations.
7. Tri-model baseline comparison (Phase 5 vs Phase 6 vs Improved).
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
PERF_DIR = HERE.parent / "performance_improvement"
PHASE1_DIR = HERE.parent / "phase1_dataset_loader"
PHASE2_DIR = HERE.parent / "phase2_preprocessing"
PHASE5_DIR = HERE.parent / "phase5_model_training"
PHASE6_DIR = HERE.parent / "phase6_heatmap_reconstruction"

for p in [PROJECT_ROOT, HERE, PERF_DIR, PHASE1_DIR, PHASE2_DIR, PHASE5_DIR, PHASE6_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from member2.performance_improvement.evaluator import SpatialUNetInferencer
    from member2.performance_improvement.postprocessor import HeatmapPostProcessor, PostProcessingConfig
    from member2.phase1_dataset_loader.dataset_loader import DatasetLoader
    from member2.phase2_preprocessing.transforms import Preprocessor
    from member2.phase7_evaluation_and_error_analysis.baseline_comparator import BaselineComparator
    from member2.phase7_evaluation_and_error_analysis.error_analyzer import ForensicErrorAnalyzer
    from member2.phase7_evaluation_and_error_analysis.evaluator_phase7 import Phase7Evaluator
    from member2.phase7_evaluation_and_error_analysis.robustness_evaluator import RobustnessEvaluator
except (ImportError, ModuleNotFoundError):
    from baseline_comparator import BaselineComparator  # type: ignore[import-not-found]
    from dataset_loader import DatasetLoader  # type: ignore[import-not-found]
    from error_analyzer import ForensicErrorAnalyzer  # type: ignore[import-not-found]
    from evaluator import SpatialUNetInferencer  # type: ignore[import-not-found]
    from evaluator_phase7 import Phase7Evaluator  # type: ignore[import-not-found]
    from postprocessor import HeatmapPostProcessor, PostProcessingConfig  # type: ignore[import-not-found]
    from robustness_evaluator import RobustnessEvaluator  # type: ignore[import-not-found]
    from transforms import Preprocessor  # type: ignore[import-not-found]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Phase7Runner")


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_phase7_pipeline(
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
    logger.info("PHASE 7: COMPREHENSIVE EVALUATION AND ERROR ANALYSIS")
    logger.info(f"Target Artifacts Directory: {artifacts_dir}")
    logger.info("=" * 75)

    # 1. Audit Dataset Partitions and Checkpoints
    logger.info("[Step 1/6] Auditing dataset partitions, test split, and model checkpoints...")
    split_manifest_path = PROJECT_ROOT / cfg.get("split_manifest")
    with open(split_manifest_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    p6_manifest = PROJECT_ROOT / "Hackathon-VIT/CBL04-Dataset/metadata/phase6_manifest.csv"
    p4_manifest = PROJECT_ROOT / "Hackathon-VIT/CBL04-Dataset/metadata/phase4_manifest.csv"
    manifest_to_use = p6_manifest if p6_manifest.exists() else p4_manifest

    loader = DatasetLoader(manifest_path=manifest_to_use)
    all_samples = loader.load(mode="SingleMode")
    sample_map = {getattr(s, "image_id"): s for s in all_samples}

    test_samples = []
    for item in split_data["samples"]:
        if item["split"] == "test":
            s = sample_map.get(item["image_id"])
            if s is not None:
                # Ensure is_tampered flag is attached
                setattr(s, "is_tampered", int(item.get("is_tampered", item.get("label", 0))))
                test_samples.append(s)

    clean_count = sum(1 for s in test_samples if getattr(s, "is_tampered") == 0)
    tamp_count = sum(1 for s in test_samples if getattr(s, "is_tampered") == 1)

    logger.info(
        f"Audited Held-Out Test Set: Total={len(test_samples)} screenshots "
        f"({clean_count} clean, {tamp_count} tampered). Zero group leakage."
    )

    improved_ckpt = PROJECT_ROOT / cfg.get("improved_checkpoint")
    if not improved_ckpt.exists():
        raise FileNotFoundError(f"Required checkpoint not found: {improved_ckpt}")
    logger.info(f"Loaded verified improved model checkpoint: {improved_ckpt}")

    inferencer = SpatialUNetInferencer(
        checkpoint_path=improved_ckpt,
        device=torch.device(cfg.get("device", "cpu")),
        batch_size=32,
    )
    postprocessor = HeatmapPostProcessor(
        PostProcessingConfig(
            threshold=cfg.get("optimal_threshold", 0.80),
            min_component_area=cfg.get("optimal_min_area", 50),
        )
    )

    # 2. Comprehensive Classification and Localization Evaluation
    logger.info("[Step 2/6] Running classification and localization evaluation on held-out test split...")
    evaluator = Phase7Evaluator(
        inferencer=inferencer,
        postprocessor=postprocessor,
        artifacts_dir=artifacts_dir,
    )

    eval_samples = test_samples[:20] if fast_mode else test_samples
    cached_csv = PROJECT_ROOT / "member2/performance_improvement/artifacts/improved_per_image_results.csv"
    eval_results = evaluator.evaluate_test_set(
        test_samples=eval_samples,
        pixel_threshold_for_classification=cfg.get("classification_pixel_threshold", 1000),
        cached_csv_path=cached_csv if (cached_csv.exists() and not fast_mode) else None,
    )

    # 3. Error Analysis and Diagnostic Overlays
    logger.info("[Step 3/6] Identifying representative forensic error categories and rendering overlays...")
    error_analyzer = ForensicErrorAnalyzer(output_dir=vis_dir)
    selected_cases = error_analyzer.select_error_examples(
        records=eval_results["records"],
        sample_map=sample_map,
    )

    for case in selected_cases:
        s = case["sample"]
        tag = case["tag"]
        img_id = case["record"]["image_id"]
        img_path = getattr(s, "image_path")

        with Image.open(img_path) as im:
            rgb_arr = np.array(Preprocessor.convert_to_rgb(im), dtype=np.uint8)

        gt = evaluator._load_ground_truth_mask(s, rgb_arr.shape[:2])
        hm, _, _ = inferencer.predict_image(rgb_arr)
        pred_mask = postprocessor.process(hm)

        out_path = vis_dir / f"overlay_{img_id}_{tag}.png"
        error_analyzer.render_overlay_figure(
            image_rgb=rgb_arr,
            gt_mask=gt,
            heatmap=hm,
            pred_mask=pred_mask,
            output_path=out_path,
            case_info=case,
        )

    # 4. Multi-Condition Robustness Evaluation
    logger.info("[Step 4/6] Executing multi-condition robustness testing under perturbations...")
    robustness_eval = RobustnessEvaluator(
        inferencer=inferencer,
        postprocessor=postprocessor,
        artifacts_dir=artifacts_dir,
    )
    max_rob_samples = 6 if fast_mode else 10
    robustness_results = robustness_eval.run_robustness_suite(
        samples=eval_samples,
        max_samples_for_suite=max_rob_samples,
    )

    # 5. Baseline Comparison Matrix Generation
    logger.info("[Step 5/6] Synthesizing tri-model baseline comparison matrix...")
    comparator = BaselineComparator(artifacts_dir=artifacts_dir)
    p5_metrics_path = PROJECT_ROOT / cfg.get("baseline_phase5_metrics")
    p6_metrics_path = PROJECT_ROOT / cfg.get("baseline_phase6_metrics")

    comparison_results = comparator.generate_comparison_matrix(
        phase5_metrics_path=p5_metrics_path,
        phase6_metrics_path=p6_metrics_path,
        phase7_metrics=eval_results,
    )

    # 6. Final Summary Export
    elapsed = time.time() - start_time
    summary = {
        "status": "SUCCESS",
        "phase": "Phase 7 - Comprehensive Evaluation and Error Analysis",
        "elapsed_seconds": elapsed,
        "config": cfg,
        "classification_summary": eval_results["screenshot_classification"],
        "localization_summary": eval_results["pixel_localization"],
        "per_family_summary": eval_results["per_family"],
        "robustness_summary": robustness_results,
        "baseline_comparison": comparison_results,
    }

    with open(artifacts_dir / "phase7_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 75)
    logger.info("PHASE 7 EVALUATION COMPLETE:")
    logger.info(f"  Screenshot Accuracy: {eval_results['screenshot_classification']['accuracy']:.2%}")
    logger.info(f"  Screenshot Precision: {eval_results['screenshot_classification']['precision']:.2%}")
    logger.info(f"  Screenshot Recall: {eval_results['screenshot_classification']['recall']:.2%}")
    logger.info(f"  Screenshot F1: {eval_results['screenshot_classification']['f1']:.4f}")
    logger.info(f"  Global Pixel Dice: {eval_results['pixel_localization']['global_pixel_dice']:.6f}")
    logger.info(f"  Global Pixel IoU:  {eval_results['pixel_localization']['global_pixel_iou']:.6f}")
    logger.info(f"  Pixel Precision:   {eval_results['pixel_localization']['global_pixel_precision']:.4f}")
    logger.info(f"  Pixel Recall:      {eval_results['pixel_localization']['global_pixel_recall']:.4f}")
    logger.info(f"  Mean Tampered Dice:{eval_results['pixel_localization']['mean_dice_tampered_only']:.4f}")
    logger.info(f"  Elapsed Time:      {elapsed:.1f}s")
    logger.info("=" * 75)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 7 Evaluation & Error Analysis")
    parser.add_argument(
        "--config",
        type=Path,
        default=HERE / "config.json",
        help="Path to Phase 7 configuration JSON",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=HERE / "artifacts",
        help="Path to Phase 7 artifacts directory",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run in fast mode on test subset",
    )
    args = parser.parse_args()

    run_phase7_pipeline(args.config, args.artifacts_dir, fast_mode=args.fast)
