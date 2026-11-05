"""
CBL-04: Forensic Heatmap & Mask Visualization Engine
Phase 6 - Member 2 (Tile Forensics CNN & Heatmap Lead)

This module generates publication-grade multi-panel forensic visualizations:
1. Synthetic Test Samples: 6-panel forensic diagnosis (Original, GT Mask, Heatmap,
   Thresholded Mask, Heatmap Overlay, Error Analysis Overlay).
2. Real Holdout Samples: 3-panel exploratory diagnosis with prominent
   UNLABELED / EXPLORATORY watermark.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def create_heatmap_overlay(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = "inferno",
) -> np.ndarray:
    """Alpha-composite a continuous [0, 1] probability heatmap onto an RGB image."""
    cm = plt.get_cmap(colormap_name)
    heatmap_colored = cm(heatmap)[:, :, :3]  # Drop alpha, in [0, 1]
    heatmap_rgb = (heatmap_colored * 255.0).astype(np.uint8)

    # Blend
    blended = (1.0 - alpha) * image_rgb.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def create_error_overlay(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Generate an RGB error overlay:
    - Green: True Positive (Hit)
    - Red: False Positive (False Alarm)
    - Blue: False Negative (Missed Tampering)
    """
    gt_bool = (gt_mask > 0)
    pred_bool = (pred_mask > 0)

    tp = pred_bool & gt_bool
    fp = pred_bool & (~gt_bool)
    fn = (~pred_bool) & gt_bool

    overlay = image_rgb.copy()

    # Dim background slightly for contrast
    dimmed = (overlay.astype(np.float32) * 0.7).astype(np.uint8)
    overlay = dimmed

    # Apply color overlays
    # TP -> Bright Green [0, 255, 0]
    overlay[tp] = np.clip(
        (1 - alpha) * overlay[tp] + alpha * np.array([0, 255, 0], dtype=np.float32),
        0, 255
    ).astype(np.uint8)

    # FP -> Red [255, 50, 50]
    overlay[fp] = np.clip(
        (1 - alpha) * overlay[fp] + alpha * np.array([255, 50, 50], dtype=np.float32),
        0, 255
    ).astype(np.uint8)

    # FN -> Cyan / Blue [0, 150, 255]
    overlay[fn] = np.clip(
        (1 - alpha) * overlay[fn] + alpha * np.array([0, 150, 255], dtype=np.float32),
        0, 255
    ).astype(np.uint8)

    return overlay


class Phase6Visualizer:
    """
    Orchestrates rendering and exporting multi-panel diagnostic PNGs.
    """

    def __init__(self, colormap: str = "inferno", overlay_alpha: float = 0.5) -> None:
        self.colormap = colormap
        self.overlay_alpha = overlay_alpha

    def render_synthetic_evaluation(
        self,
        image_rgb: np.ndarray,
        gt_mask: np.ndarray,
        heatmap: np.ndarray,
        pred_mask: np.ndarray,
        output_path: Union[str, Path],
        image_id: str,
        threshold: float,
        metrics: Optional[dict] = None,
    ) -> Path:
        """
        Render 6-panel composite visualization for synthetic test screenshots.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        overlay_hm = create_heatmap_overlay(
            image_rgb, heatmap, alpha=self.overlay_alpha, colormap_name=self.colormap
        )
        overlay_err = create_error_overlay(image_rgb, gt_mask, pred_mask, alpha=0.6)

        fig, axes = plt.subplots(2, 3, figsize=(15, 12))
        fig.suptitle(
            f"Phase 6 Forensic Tampering Localization | Image: {image_id} (Threshold: {threshold:.2f})",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        # 1. Original RGB
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title("(1) Original Screenshot", fontsize=11, fontweight="bold")
        axes[0, 0].axis("off")

        # 2. Ground-Truth Mask
        axes[0, 1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[0, 1].set_title("(2) Ground-Truth Mask (White=Tampered)", fontsize=11, fontweight="bold")
        axes[0, 1].axis("off")

        # 3. Probability Heatmap
        im3 = axes[0, 2].imshow(heatmap, cmap=self.colormap, vmin=0.0, vmax=1.0)
        axes[0, 2].set_title("(3) Predicted Probability Heatmap", fontsize=11, fontweight="bold")
        axes[0, 2].axis("off")
        fig.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04, label="P(Tampered)")

        # 4. Thresholded Mask
        axes[1, 0].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
        axes[1, 0].set_title(f"(4) Predicted Mask (>= {threshold:.2f})", fontsize=11, fontweight="bold")
        axes[1, 0].axis("off")

        # 5. Heatmap Overlay
        axes[1, 1].imshow(overlay_hm)
        axes[1, 1].set_title("(5) Heatmap Overlay", fontsize=11, fontweight="bold")
        axes[1, 1].axis("off")

        # 6. Error Diagnostic Overlay
        axes[1, 2].imshow(overlay_err)
        stat_str = ""
        if metrics:
            stat_str = f" [Dice={metrics.get('dice', 0.0):.3f}, IoU={metrics.get('iou', 0.0):.3f}]"
        axes[1, 2].set_title(f"(6) Forensic Diagnostic{stat_str}\nGreen:Hit | Red:FalseAlarm | Blue:Miss", fontsize=11, fontweight="bold")
        axes[1, 2].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(out_p, dpi=120)
        plt.close(fig)
        return out_p

    def render_real_holdout_exploratory(
        self,
        image_rgb: np.ndarray,
        heatmap: np.ndarray,
        pred_mask: np.ndarray,
        output_path: Union[str, Path],
        image_id: str,
        threshold: float,
    ) -> Path:
        """
        Render 3-panel composite visualization for unlabelled real holdout screenshots.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        overlay_hm = create_heatmap_overlay(
            image_rgb, heatmap, alpha=self.overlay_alpha, colormap_name=self.colormap
        )

        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.suptitle(
            f"EXPLORATORY ANALYSIS ONLY: Real Holdout Screenshot ({image_id})\n"
            f"[UNLABELED / NO GROUND TRUTH - NO METRICS ASSIGNED]",
            fontsize=13,
            fontweight="bold",
            color="darkred",
            y=0.98,
        )

        # 1. Original RGB
        axes[0].imshow(image_rgb)
        axes[0].set_title("(1) Original Holdout Screenshot", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # 2. Probability Heatmap
        im2 = axes[1].imshow(heatmap, cmap=self.colormap, vmin=0.0, vmax=1.0)
        axes[1].set_title(f"(2) Inferred Probability Heatmap", fontsize=11, fontweight="bold")
        axes[1].axis("off")
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="P(Tampered)")

        # 3. Heatmap Overlay
        axes[2].imshow(overlay_hm)
        axes[2].set_title("(3) Heatmap Overlay (Exploratory)", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(out_p, dpi=120)
        plt.close(fig)
        return out_p

