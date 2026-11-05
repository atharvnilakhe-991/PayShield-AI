"""
CBL-04: Phase 7 Error Analysis & Visual Overlay Generator
Member 2 - Tile Forensics CNN & Heatmap Lead

This module identifies 5 key forensic error categories on synthetic test screenshots:
1. False-Positive Screenshots (clean image falsely classified)
2. False-Negative Screenshots (tampered edit missed / zero detection)
3. Excessive False-Positive Regions (tampered image with heavy spurious noise)
4. Poorly Localized Tampering (tampered image with low Dice / boundary bleed)
5. Successful Localization (high Dice, clean background isolation)

Generates 5-panel forensic overlay figures with:
- Panel 1: Original RGB Screenshot
- Panel 2: Ground-Truth Mask
- Panel 3: Predicted Continuous Heatmap
- Panel 4: Predicted Post-Processed Binary Mask
- Panel 5: RGB Overlay with Green=TP, Red=FP, Blue=FN
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
PHASE2_DIR = HERE.parent / "phase2_preprocessing"

try:
    from member2.phase2_preprocessing.transforms import Preprocessor
except (ImportError, ModuleNotFoundError):
    from transforms import Preprocessor  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


class ForensicErrorAnalyzer:
    """Identifies and visualizes representative forensic errors and successes."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or HERE / "artifacts" / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def select_error_examples(
        self,
        records: List[Dict[str, Any]],
        sample_map: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Categorize test records into the 5 required diagnostic cases.
        """
        clean_recs = [r for r in records if r["is_tampered"] == 0]
        tamp_recs = [r for r in records if r["is_tampered"] == 1]

        selected_cases: List[Dict[str, Any]] = []

        # 1. False-Positive Screenshot (clean screenshot with highest FP pixels)
        if clean_recs:
            fp_sorted = sorted(clean_recs, key=lambda r: r["fp_pixels"], reverse=True)
            fp_case = fp_sorted[0]
            selected_cases.append({
                "category": "False-Positive Screenshot",
                "tag": "false_positive_clean",
                "record": fp_case,
                "sample": sample_map.get(fp_case["image_id"]),
                "description": (
                    f"Clean screenshot with highest false-alarm pixel count "
                    f"({fp_case['fp_pixels']:,} FP pixels, detected {fp_case['detected_pixels']} px)."
                ),
            })

        # 2. False-Negative Screenshot (tampered screenshot with zero or lowest TP / lowest recall)
        if tamp_recs:
            fn_sorted = sorted(tamp_recs, key=lambda r: (r["pixel_recall"], r["tp_pixels"]))
            fn_case = fn_sorted[0]
            selected_cases.append({
                "category": "False-Negative Screenshot",
                "tag": "false_negative_missed",
                "record": fn_case,
                "sample": sample_map.get(fn_case["image_id"]),
                "description": (
                    f"Tampered screenshot with lowest recall ({fn_case['pixel_recall']:.2%}, "
                    f"{fn_case['tp_pixels']} TP px, {fn_case['fn_pixels']} missed FN px)."
                ),
            })

        # 3. Excessive False-Positive Regions (tampered screenshot with highest FP pixels)
        if tamp_recs:
            excess_fp_sorted = sorted(tamp_recs, key=lambda r: r["fp_pixels"], reverse=True)
            excess_fp_case = excess_fp_sorted[0]
            selected_cases.append({
                "category": "Excessive False-Positive Regions",
                "tag": "excessive_fp_regions",
                "record": excess_fp_case,
                "sample": sample_map.get(excess_fp_case["image_id"]),
                "description": (
                    f"Tampered screenshot with highest false-positive background noise "
                    f"({excess_fp_case['fp_pixels']:,} FP pixels, Precision={excess_fp_case['pixel_precision']:.2%})."
                ),
            })

        # 4. Poorly Localized Tampering (tampered with moderate recall but low Dice / IoU)
        if tamp_recs:
            # moderate recall (>0.15) but low dice
            cand = [r for r in tamp_recs if r["pixel_recall"] > 0.15]
            if cand:
                poor_loc_sorted = sorted(cand, key=lambda r: r["dice"])
                poor_loc_case = poor_loc_sorted[0]
            else:
                poor_loc_case = sorted(tamp_recs, key=lambda r: r["dice"])[len(tamp_recs) // 4]
            selected_cases.append({
                "category": "Poorly Localized Tampering",
                "tag": "poorly_localized_edit",
                "record": poor_loc_case,
                "sample": sample_map.get(poor_loc_case["image_id"]),
                "description": (
                    f"Tampered screenshot with loose spatial localization "
                    f"(Recall={poor_loc_case['pixel_recall']:.2%}, Dice={poor_loc_case['dice']:.4f}, IoU={poor_loc_case['iou']:.4f})."
                ),
            })

        # 5. Successful Localization (highest Dice / IoU among tampered)
        if tamp_recs:
            success_sorted = sorted(tamp_recs, key=lambda r: (r["dice"], r["pixel_precision"]), reverse=True)
            success_case = success_sorted[0]
            selected_cases.append({
                "category": "Successful Localization",
                "tag": "successful_localization",
                "record": success_case,
                "sample": sample_map.get(success_case["image_id"]),
                "description": (
                    f"High-fidelity localization of tampered text/digit "
                    f"(Dice={success_case['dice']:.4f}, IoU={success_case['iou']:.4f}, "
                    f"Precision={success_case['pixel_precision']:.2%}, Recall={success_case['pixel_recall']:.2%})."
                ),
            })

        return selected_cases

    def render_overlay_figure(
        self,
        image_rgb: np.ndarray,
        gt_mask: np.ndarray,
        heatmap: np.ndarray,
        pred_mask: np.ndarray,
        output_path: Path,
        case_info: Dict[str, Any],
    ) -> None:
        """
        Render 5-panel forensic overlay figure.
        """
        category = case_info["category"]
        rec = case_info["record"]
        desc = case_info["description"]
        img_id = rec["image_id"]

        h, w = image_rgb.shape[:2]

        # Prepare composite color-coded overlay
        # Green: TP (pred=1, gt=1)
        # Red: FP (pred=1, gt=0)
        # Blue: FN (pred=0, gt=1)
        p_bool = (pred_mask > 0)
        g_bool = (gt_mask > 0)

        tp_mask = p_bool & g_bool
        fp_mask = p_bool & (~g_bool)
        fn_mask = (~p_bool) & g_bool

        overlay = (image_rgb.astype(np.float32) * 0.65).astype(np.uint8)

        # Apply colored highlights with alpha blending
        alpha = 0.80
        # Green for TP
        overlay[tp_mask, 0] = (overlay[tp_mask, 0] * (1 - alpha) + 0 * alpha).astype(np.uint8)
        overlay[tp_mask, 1] = (overlay[tp_mask, 1] * (1 - alpha) + 255 * alpha).astype(np.uint8)
        overlay[tp_mask, 2] = (overlay[tp_mask, 2] * (1 - alpha) + 0 * alpha).astype(np.uint8)

        # Red for FP
        overlay[fp_mask, 0] = (overlay[fp_mask, 0] * (1 - alpha) + 255 * alpha).astype(np.uint8)
        overlay[fp_mask, 1] = (overlay[fp_mask, 1] * (1 - alpha) + 40 * alpha).astype(np.uint8)
        overlay[fp_mask, 2] = (overlay[fp_mask, 2] * (1 - alpha) + 40 * alpha).astype(np.uint8)

        # Cyan/Blue for FN
        overlay[fn_mask, 0] = (overlay[fn_mask, 0] * (1 - alpha) + 0 * alpha).astype(np.uint8)
        overlay[fn_mask, 1] = (overlay[fn_mask, 1] * (1 - alpha) + 160 * alpha).astype(np.uint8)
        overlay[fn_mask, 2] = (overlay[fn_mask, 2] * (1 - alpha) + 255 * alpha).astype(np.uint8)

        fig, axes = plt.subplots(1, 5, figsize=(22, 9), dpi=150)

        # 1. Original RGB
        axes[0].imshow(image_rgb)
        axes[0].set_title(f"1. Native Screenshot\n({w}x{h})", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # 2. Ground Truth Mask
        axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        gt_px = int(np.sum(gt_mask > 0))
        axes[1].set_title(f"2. Ground Truth Mask\n({gt_px:,} tampered px)", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        # 3. Heatmap
        hm_im = axes[2].imshow(heatmap, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[2].set_title(f"3. U-Net Heatmap\n(Peak: {rec.get('max_prob', 0):.3f})", fontsize=11, fontweight="bold")
        axes[2].axis("off")
        cbar = fig.colorbar(hm_im, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        # 4. Predicted Binary Mask
        axes[3].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
        axes[3].set_title(
            f"4. Binary Prediction\n({rec.get('detected_pixels', 0):,} px flagged)",
            fontsize=11,
            fontweight="bold",
        )
        axes[3].axis("off")

        # 5. Diagnostic Overlay
        axes[4].imshow(overlay)
        axes[4].set_title("5. Forensic Diagnostic Overlay\nGreen=TP, Red=FP, Blue=FN", fontsize=11, fontweight="bold")
        axes[4].axis("off")

        # Overall Title & Subtitle
        fig.suptitle(
            f"Phase 7 Error Analysis: [{category.upper()}] — Image ID: {img_id} (Family: {rec['layout_family']})\n"
            f"Metrics: Dice={rec.get('dice', 0):.4f} | IoU={rec.get('iou', 0):.4f} | "
            f"Prec={rec.get('pixel_precision', 0):.2%} | Rec={rec.get('pixel_recall', 0):.2%} | "
            f"TP={rec.get('tp_pixels', 0):,} px | FP={rec.get('fp_pixels', 0):,} px | FN={rec.get('fn_pixels', 0):,} px\n"
            f"Case Note: {desc}",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )

        fig.tight_layout(rect=[0, 0.02, 1, 0.94])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved forensic error overlay: {output_path}")

