"""
CBL-04: Tile Forensics Evaluation & Baseline Comparison Engine
Phase 5 - Member 2 (Tile Forensics CNN & Heatmap Lead)

This module evaluates the trained tile forensics CNN on unseen test data:
- Primary Metrics: Accuracy, Precision, Recall, F1-score (Tampered class), ROC-AUC, PR-AUC.
- Detailed Diagnostics: Confusion matrix (TN, FP, FN, TP), False Positive Rate (FPR),
  False Negative Rate (FNR).
- Per-Family Breakdown: Stratified forensics performance for 'compact', 'enriched',
  'bank_progress', and 'recharge' layout families.
- Baseline Comparison: Direct comparative evaluation against Member 1's global-downsampling
  baseline (~50.5% accuracy).
- Artifacts: Generates JSON metrics, CSV predictions, and confusion matrix visualizations.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent

for p in [PROJECT_ROOT, HERE]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

logger = logging.getLogger(__name__)

# Member 1 Phase 7 Baseline Metrics for direct comparison
BASELINE_METRICS = {
    "model_name": "ResNet-18 Multi-Task (Global Downsampling)",
    "input_resolution": "160x320 downsampled",
    "overall_accuracy": 0.5046,
    "overall_precision": 0.5043,
    "overall_recall": 0.5461,
    "overall_f1": 0.5243,
    "compact_accuracy": 0.5000,
    "compact_recall": 1.0000,
    "compact_precision": 0.5000,
    "enriched_accuracy": 0.5093,
    "enriched_recall": 0.0880,
    "enriched_precision": 0.5588,
    "defect_analysis": (
        "Global downsampling destroyed text anti-aliasing and micro-artifacts, causing "
        "the model to collapse into degenerate majority-class or layout-memorization guessing."
    ),
}


@dataclass
class EvaluationResult:
    """Comprehensive test evaluation results and baseline comparison."""

    total_samples: int
    num_clean: int
    num_tampered: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: Dict[str, int]
    false_positive_rate: float
    false_negative_rate: float
    per_family: Dict[str, Dict[str, Any]]
    baseline_comparison: Dict[str, Any]
    artifacts: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Evaluator:
    """
    Executes forensic evaluation of trained models on tile datasets.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        artifacts_dir: Union[str, Path] = "artifacts",
    ) -> None:
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
        artifacts_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[Evaluator, Dict[str, Any]]:
        """Load model from saved PyTorch checkpoint."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        from .model import get_model

        dev = device or torch.device("cpu")
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)

        model_name = ckpt.get("model_name", "resnet18")
        num_classes = ckpt.get("config", {}).get("num_classes", 2)
        model = get_model(model_name=model_name, num_classes=num_classes)
        model.load_state_dict(ckpt["state_dict"])

        out_dir = artifacts_dir or ckpt_path.parent
        evaluator = cls(model=model, device=dev, artifacts_dir=out_dir)
        return evaluator, ckpt

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        save_artifacts: bool = True,
    ) -> EvaluationResult:
        """
        Run inference over test_loader and calculate comprehensive forensic metrics.
        """
        self.model.eval()

        all_true: List[int] = []
        all_pred: List[int] = []
        all_prob: List[float] = []
        tile_records: List[Dict[str, Any]] = []

        for batch_idx, (inputs, targets, infos) in enumerate(test_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            trues = targets.cpu().numpy().tolist()

            all_true.extend(trues)
            all_pred.extend(preds)
            all_prob.extend(probs)

            # Unpack metadata batch
            b_size = len(trues)
            for i in range(b_size):
                rec = {
                    "tile_id": infos["tile_id"][i] if "tile_id" in infos else f"tile_{len(tile_records)}",
                    "source_id": infos["source_id"][i] if "source_id" in infos else "",
                    "pair_id": infos["pair_id"][i] if "pair_id" in infos else "",
                    "split_group_id": infos["split_group_id"][i] if "split_group_id" in infos else "",
                    "layout_family": infos["layout_family"][i] if "layout_family" in infos else "unknown",
                    "true_label": trues[i],
                    "predicted_label": preds[i],
                    "tampered_prob": probs[i],
                }
                tile_records.append(rec)

        # Basic counts
        n_total = len(all_true)
        n_clean = sum(1 for yt in all_true if yt == 0)
        n_tamp = sum(1 for yt in all_true if yt == 1)

        # Standard metrics
        acc = float(accuracy_score(all_true, all_pred)) if n_total > 0 else 0.0
        prec = float(precision_score(all_true, all_pred, zero_division=0)) if n_total > 0 else 0.0
        rec = float(recall_score(all_true, all_pred, zero_division=0)) if n_total > 0 else 0.0
        f1 = float(f1_score(all_true, all_pred, zero_division=0)) if n_total > 0 else 0.0

        # ROC-AUC & PR-AUC
        try:
            roc_auc = float(roc_auc_score(all_true, all_prob)) if len(set(all_true)) > 1 else 0.5
        except Exception:
            roc_auc = 0.5

        try:
            pr_auc = float(average_precision_score(all_true, all_prob)) if len(set(all_true)) > 1 else 0.0
        except Exception:
            pr_auc = 0.0

        # Confusion matrix
        if n_total > 0:
            cm = confusion_matrix(all_true, all_pred, labels=[0, 1])
            tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        cm_dict = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

        # Breakdown per layout family
        family_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in tile_records:
            family_records[r["layout_family"]].append(r)

        per_family: Dict[str, Dict[str, Any]] = {}
        for fam, recs in sorted(family_records.items()):
            fam_true = [r["true_label"] for r in recs]
            fam_pred = [r["predicted_label"] for r in recs]
            fam_cm = confusion_matrix(fam_true, fam_pred, labels=[0, 1])

            per_family[fam] = {
                "support": len(recs),
                "clean_support": sum(1 for yt in fam_true if yt == 0),
                "tampered_support": sum(1 for yt in fam_true if yt == 1),
                "accuracy": float(accuracy_score(fam_true, fam_pred)),
                "precision": float(precision_score(fam_true, fam_pred, zero_division=0)),
                "recall": float(recall_score(fam_true, fam_pred, zero_division=0)),
                "f1": float(f1_score(fam_true, fam_pred, zero_division=0)),
                "confusion_matrix": {
                    "tn": int(fam_cm[0, 0]),
                    "fp": int(fam_cm[0, 1]),
                    "fn": int(fam_cm[1, 0]),
                    "tp": int(fam_cm[1, 1]),
                },
            }

        # Comparative Analysis with Baseline
        baseline_comp = {
            "baseline_overall_accuracy": BASELINE_METRICS["overall_accuracy"],
            "tile_model_accuracy": acc,
            "accuracy_improvement": acc - BASELINE_METRICS["overall_accuracy"],
            "baseline_overall_f1": BASELINE_METRICS["overall_f1"],
            "tile_model_f1": f1,
            "f1_improvement": f1 - BASELINE_METRICS["overall_f1"],
            "baseline_model": BASELINE_METRICS["model_name"],
            "baseline_defect": BASELINE_METRICS["defect_analysis"],
            "tile_advantage": (
                "Preserving native resolution (256x256 uncompressed patches) allows the CNN "
                "to recognize anti-aliasing distortions, JPEG grid discontinuity, and splice boundaries."
            ),
        }

        artifacts: Dict[str, str] = {}

        if save_artifacts:
            # 1. Save metrics JSON
            metrics_path = self.artifacts_dir / "metrics_test.json"
            result_dict = {
                "total_samples": n_total,
                "clean_samples": n_clean,
                "tampered_samples": n_tamp,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "confusion_matrix": cm_dict,
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
                "per_family": per_family,
                "baseline_comparison": baseline_comp,
            }
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2)
            artifacts["metrics_json"] = str(metrics_path)

            # 2. Save confusion matrix JSON
            cm_path = self.artifacts_dir / "confusion_matrix.json"
            with open(cm_path, "w", encoding="utf-8") as f:
                json.dump(cm_dict, f, indent=2)
            artifacts["confusion_matrix_json"] = str(cm_path)

            # 3. Draw confusion matrix visualization
            cm_plot_path = self.artifacts_dir / "confusion_matrix.png"
            self._plot_confusion_matrix(cm_dict, cm_plot_path)
            artifacts["confusion_matrix_png"] = str(cm_plot_path)

            # 4. Save comparison JSON
            comp_path = self.artifacts_dir / "baseline_vs_tile_comparison.json"
            with open(comp_path, "w", encoding="utf-8") as f:
                json.dump(baseline_comp, f, indent=2)
            artifacts["comparison_json"] = str(comp_path)

            # 5. Save predictions CSV
            pred_path = self.artifacts_dir / "predictions_test.csv"
            if tile_records:
                with open(pred_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(tile_records[0].keys()))
                    writer.writeheader()
                    writer.writerows(tile_records)
            artifacts["predictions_csv"] = str(pred_path)

        return EvaluationResult(
            total_samples=n_total,
            num_clean=n_clean,
            num_tampered=n_tamp,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            confusion_matrix=cm_dict,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            per_family=per_family,
            baseline_comparison=baseline_comp,
            artifacts=artifacts,
        )

    def _plot_confusion_matrix(
        self, cm_dict: Dict[str, int], output_path: Union[str, Path]
    ) -> None:
        """Render publication-grade 2x2 confusion matrix heatmap."""
        matrix = np.array([
            [cm_dict["tn"], cm_dict["fp"]],
            [cm_dict["fn"], cm_dict["tp"]],
        ])

        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(matrix, cmap="Blues", alpha=0.8)

        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                total = np.sum(matrix[i, :])
                pct = (val / total * 100.0) if total > 0 else 0.0
                ax.text(
                    j,
                    i,
                    f"{val}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    color="white" if val > matrix.max() / 2 else "black",
                    fontsize=12,
                    fontweight="bold",
                )

        fig.colorbar(cax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Clean (0)", "Tampered (1)"], fontsize=11)
        ax.set_yticklabels(["Clean (0)", "Tampered (1)"], fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
        ax.set_title("Tile Forensics Confusion Matrix", fontsize=13, fontweight="bold", pad=15)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved confusion matrix plot to {output_path}")

