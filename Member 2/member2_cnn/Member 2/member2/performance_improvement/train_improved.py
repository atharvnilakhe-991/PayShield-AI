"""
CBL-04: Improved Model Training Pipeline (Tile Spatial Segmentation)
Phase 6 Improvement - Member 2 (Tile Forensics CNN & Heatmap Lead)

This module executes supervised training of TileSpatialUNet:
- Joint loss: CombinedBCEDiceLoss (BCE + Soft Dice).
- Balanced sampling: 50% tampered tiles, 50% clean tiles per epoch.
- Optimizer: AdamW with Cosine Annealing learning rate schedule.
- Early stopping monitored on validation Dice.
- Saves best checkpoint to artifacts/best_improved_model.pt.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent

for p in [PROJECT_ROOT, HERE]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from member2.performance_improvement.models import CombinedBCEDiceLoss, TileSpatialUNet
    from member2.performance_improvement.segmentation_dataset import TileSpatialSegmentationDataset
except (ImportError, ModuleNotFoundError):
    from models import CombinedBCEDiceLoss, TileSpatialUNet  # type: ignore[import-not-found]
    from segmentation_dataset import TileSpatialSegmentationDataset  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


class ImprovedTrainer:
    """
    Supervised trainer for tile spatial segmentation.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        num_epochs: int = 8,
        early_stopping_patience: int = 3,
        checkpoint_dir: Union[str, Path] = "artifacts",
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: List[Dict[str, Any]] = []
        self.best_val_dice: float = -1.0
        self.best_epoch: int = 0

    def _compute_pixel_metrics(
        self, probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5
    ) -> Dict[str, float]:
        preds = (probs >= threshold).astype(bool)
        gts = (targets > 0.5).astype(bool)

        tp = int(np.sum(preds & gts))
        fp = int(np.sum(preds & (~gts)))
        fn = int(np.sum((~preds) & gts))
        tn = int(np.sum((~preds) & (~gts)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        return {
            "dice": float(dice),
            "iou": float(iou),
            "precision": float(prec),
            "recall": float(rec),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0
        n_samples = 0

        for batch_idx, (inputs, targets, _) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            b_size = inputs.size(0)
            total_loss += loss.item() * b_size
            n_samples += b_size

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5)
            gts = (targets > 0.5)

            total_tp += int((preds & gts).sum().item())
            total_fp += int((preds & (~gts)).sum().item())
            total_fn += int(((~preds) & gts).sum().item())
            total_tn += int(((~preds) & (~gts)).sum().item())

        avg_loss = total_loss / max(n_samples, 1)
        dice = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn) if (2.0 * total_tp + total_fp + total_fn) > 0 else 0.0

        return {
            "loss": float(avg_loss),
            "dice": float(dice),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        }

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0
        n_samples = 0

        for batch_idx, (inputs, targets, _) in enumerate(self.val_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            logits = self.model(inputs)
            loss = self.criterion(logits, targets)

            b_size = inputs.size(0)
            total_loss += loss.item() * b_size
            n_samples += b_size

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5)
            gts = (targets > 0.5)

            total_tp += int((preds & gts).sum().item())
            total_fp += int((preds & (~gts)).sum().item())
            total_fn += int(((~preds) & gts).sum().item())
            total_tn += int(((~preds) & (~gts)).sum().item())

        avg_loss = total_loss / max(n_samples, 1)
        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        dice = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn) if (2.0 * total_tp + total_fp + total_fn) > 0 else 0.0
        iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

        return {
            "loss": float(avg_loss),
            "dice": float(dice),
            "iou": float(iou),
            "precision": float(prec),
            "recall": float(rec),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        }

    def fit(self) -> Dict[str, Any]:
        patience_counter = 0
        best_ckpt_path = self.checkpoint_dir / "best_improved_model.pt"

        logger.info(
            f"Starting U-Net training on {self.device}: epochs={self.num_epochs}, "
            f"patience={self.early_stopping_patience}"
        )

        for epoch in range(1, self.num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            record = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_dice": train_metrics["dice"],
                "val_loss": val_metrics["loss"],
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
            }
            self.history.append(record)

            logger.info(
                f"Epoch {epoch:02d}/{self.num_epochs:02d} | "
                f"Train Loss: {train_metrics['loss']:.4f}, Train Dice: {train_metrics['dice']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Dice: {val_metrics['dice']:.4f}, "
                f"Val Prec: {val_metrics['precision']:.4f}, Val Rec: {val_metrics['recall']:.4f}"
            )

            # Check improvement on validation Dice
            if val_metrics["dice"] > self.best_val_dice:
                self.best_val_dice = val_metrics["dice"]
                self.best_epoch = epoch
                patience_counter = 0

                # Save best checkpoint
                torch.save(
                    {
                        "epoch": epoch,
                        "model_name": "spatial_unet",
                        "state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_metrics["loss"],
                        "val_dice": val_metrics["dice"],
                        "val_iou": val_metrics["iou"],
                        "val_precision": val_metrics["precision"],
                        "val_recall": val_metrics["recall"],
                        "training_history": self.history,
                    },
                    best_ckpt_path,
                )
                logger.info(f"[*] New best validation Dice ({val_metrics['dice']:.4f})! Saved best_improved_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        # Save history JSON
        with open(self.checkpoint_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        return {
            "best_epoch": self.best_epoch,
            "best_val_dice": self.best_val_dice,
            "checkpoint_path": str(best_ckpt_path),
            "history": self.history,
        }
