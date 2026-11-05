"""
CBL-04: Comparative Multi-Panel Forensic Visualizer
Phase 6 Improvement - Member 2 (Tile Forensics CNN & Heatmap Lead)

Generates 7-panel side-by-side diagnostic figures comparing Baseline vs Improved:
1. Original Screenshot
2. Ground-Truth Mask
3. Baseline Probability Heatmap (Phase 6)
4. Improved Spatial Probability Heatmap
5. Baseline Predicted Mask
6. Improved Predicted Mask
7. Forensic Error Overlay (Green=TP, Red=FP, Blue=FN)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def create_error_overlay(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    gt_bool = (gt_mask > 0)
    pred_bool = (pred_mask > 0)

    tp = pred_bool & gt_bool
    fp = pred_bool & (~gt_bool)
    fn = (~pred_bool) & gt_bool

    # Dim background
    overlay = (image_rgb.astype(np.float32) * 0.7).astype(np.uint8)

    # TP: Bright Green
    overlay[tp] = np.clip(
        (1 - alpha) * overlay[tp] + alpha * np.array([0, 255, 0], dtype=np.float32), 0, 255
    ).astype(np.uint8)

    # FP: Red
    overlay[fp] = np.clip(
        (1 - alpha) * overlay[fp] + alpha * np.array([255, 50, 50], dtype=np.float32), 0, 255
    ).astype(np.uint8)

    # FN: Blue / Cyan
    overlay[fn] = np.clip(
        (1 - alpha) * overlay[fn] + alpha * np.array([0, 150, 255], dtype=np.float32), 0, 255
    ).astype(np.uint8)

    return overlay


class ComparativeVisualizer:
    """Renders 7-panel comparative forensic figures."""

    def __init__(self, colormap: str = "inferno") -> None:
        self.colormap = colormap

    def render_comparison(
        self,
        image_rgb: np.ndarray,
        gt_mask: np.ndarray,
        baseline_heatmap: np.ndarray,
        improved_heatmap: np.ndarray,
        baseline_pred_mask: np.ndarray,
        improved_pred_mask: np.ndarray,
        output_path: Union[str, Path],
        image_id: str,
        category_label: str = "General",
        metrics_dict: Optional[dict] = None,
    ) -> Path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        err_overlay = create_error_overlay(image_rgb, gt_mask, improved_pred_mask, alpha=0.6)

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 4, hspace=0.25, wspace=0.20)

        fig.suptitle(
            f"Forensic Comparison: {image_id} [{category_label.upper()}]\n"
            f"Baseline (Phase 6) vs Improved Spatial Localization",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        # (0, 0) Original Screenshot
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(image_rgb)
        ax0.set_title("(1) Original Screenshot", fontsize=10, fontweight="bold")
        ax0.axis("off")

        # (0, 1) Ground Truth Mask
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        ax1.set_title("(2) Ground-Truth Mask", fontsize=10, fontweight="bold")
        ax1.axis("off")

        # (0, 2) Baseline Heatmap
        ax2 = fig.add_subplot(gs[0, 2])
        im2 = ax2.imshow(baseline_heatmap, cmap=self.colormap, vmin=0.0, vmax=1.0)
        ax2.set_title("(3) Baseline Heatmap (Phase 6)", fontsize=10, fontweight="bold")
        ax2.axis("off")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # (0, 3) Improved Heatmap
        ax3 = fig.add_subplot(gs[0, 3])
        im3 = ax3.imshow(improved_heatmap, cmap=self.colormap, vmin=0.0, vmax=1.0)
        ax3.set_title("(4) Improved Heatmap (U-Net)", fontsize=10, fontweight="bold")
        ax3.axis("off")
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # (1, 0) Baseline Predicted Mask
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(baseline_pred_mask, cmap="gray", vmin=0, vmax=1)
        ax4.set_title("(5) Baseline Mask (τ=0.80)", fontsize=10, fontweight="bold")
        ax4.axis("off")

        # (1, 1) Improved Predicted Mask
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(improved_pred_mask, cmap="gray", vmin=0, vmax=1)
        ax5.set_title("(6) Improved Mask (Filtered)", fontsize=10, fontweight="bold")
        ax5.axis("off")

        # (1, 2:4) Diagnostic Error Overlay (spans 2 columns)
        ax6 = fig.add_subplot(gs[1, 2:])
        ax6.imshow(err_overlay)
        stat_text = ""
        if metrics_dict:
            stat_text = f" | Dice: {metrics_dict.get('dice', 0.0):.4f}, IoU: {metrics_dict.get('iou', 0.0):.4f}"
        ax6.set_title(
            f"(7) Improved Forensic Diagnostic{stat_text}\n"
            f"[Green: True Positive | Red: False Alarm | Blue: Missed Edit]",
            fontsize=10,
            fontweight="bold",
        )
        ax6.axis("off")

        plt.savefig(out_p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out_p

