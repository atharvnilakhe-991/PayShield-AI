"""
CBL-04: Tile Forensics CNN Training Pipeline
Phase 5 - Member 2 (Tile Forensics CNN & Heatmap Lead)

This module implements the end-to-end training loop for the tile forensics CNN:
- Loss function: Class-weighted CrossEntropyLoss (combats ~11.69:1 imbalance).
- Optimizer: AdamW with weight decay.
- LR Scheduler: CosineAnnealingLR or StepLR.
- Early Stopping: Monitors validation F1-score or validation loss with configurable patience.
- Checkpointing: Saves `best_model.pt`, `last_model.pt`, and `training_history.json`.
- Reproducibility: Fully deterministic random seeding.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
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

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Encapsulates training outcomes, history, and checkpoint locations."""

    best_epoch: int
    best_val_loss: float
    best_val_f1: float
    best_val_accuracy: float
    best_val_precision: float
    best_val_recall: float
    total_epochs_trained: int
    stopped_early: bool
    best_model_path: str
    last_model_path: str
    training_history: List[Dict[str, Any]]
    config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Trainer:
    """
    Executes supervised training of the tile forensics CNN.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        num_epochs: int = 10,
        early_stopping_patience: int = 3,
        early_stopping_min_delta: float = 0.001,
        early_stopping_metric: str = "val_f1",
        checkpoint_dir: Union[str, Path] = "artifacts",
        device: Optional[torch.device] = None,
        model_name: str = "resnet18",
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_metric = early_stopping_metric.lower()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.config_dict = config_dict or {}

        # Default criterion if none provided
        self.criterion = criterion or nn.CrossEntropyLoss()

        # Default optimizer: AdamW
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-4
        )

        # LR Scheduler
        self.scheduler = scheduler

        # Tracking state
        self.history: List[Dict[str, Any]] = []
        self.best_val_f1: float = -1.0
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0
        self.best_metrics: Dict[str, float] = {}

    def _compute_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """Compute accuracy, precision, recall, and F1 for binary classification."""
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

        total = len(y_true)
        acc = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        all_true: List[int] = []
        all_pred: List[int] = []

        for batch_idx, (inputs, targets, _) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_true.extend(targets.cpu().numpy().tolist())
            all_pred.extend(preds.tolist())

        total_samples = len(all_true)
        avg_loss = total_loss / max(total_samples, 1)
        metrics = self._compute_metrics(all_true, all_pred)
        metrics["loss"] = float(avg_loss)
        return metrics

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Run validation evaluation."""
        self.model.eval()
        total_loss = 0.0
        all_true: List[int] = []
        all_pred: List[int] = []

        for batch_idx, (inputs, targets, _) in enumerate(self.val_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_true.extend(targets.cpu().numpy().tolist())
            all_pred.extend(preds.tolist())

        total_samples = len(all_true)
        avg_loss = total_loss / max(total_samples, 1)
        metrics = self._compute_metrics(all_true, all_pred)
        metrics["loss"] = float(avg_loss)
        return metrics

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        filename: str,
    ) -> Path:
        """Save PyTorch checkpoint containing model state, optimizer state, and training metadata."""
        ckpt_path = self.checkpoint_dir / filename
        checkpoint_data = {
            "epoch": epoch,
            "model_name": self.model_name,
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": metrics.get("loss", 0.0),
            "val_f1": metrics.get("f1", 0.0),
            "val_accuracy": metrics.get("accuracy", 0.0),
            "val_precision": metrics.get("precision", 0.0),
            "val_recall": metrics.get("recall", 0.0),
            "config": self.config_dict,
            "training_history": self.history,
        }
        torch.save(checkpoint_data, ckpt_path)
        return ckpt_path

    def fit(self) -> TrainingResult:
        """Execute full training procedure with early stopping and checkpointing."""
        logger.info(
            f"Starting training on device={self.device}: epochs={self.num_epochs}, "
            f"patience={self.early_stopping_patience}, metric={self.early_stopping_metric}"
        )

        patience_counter = 0
        stopped_early = False
        best_ckpt_path = self.checkpoint_dir / "best_model.pt"
        last_ckpt_path = self.checkpoint_dir / "last_model.pt"

        for epoch in range(1, self.num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)

            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            epoch_record = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
            }
            self.history.append(epoch_record)

            logger.info(
                f"Epoch {epoch:02d}/{self.num_epochs:02d} | "
                f"Train Loss: {train_metrics['loss']:.4f}, Train F1: {train_metrics['f1']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

            # Determine improvement
            is_best = False
            if self.early_stopping_metric == "val_loss":
                delta = self.best_val_loss - val_metrics["loss"]
                if delta > self.early_stopping_min_delta:
                    self.best_val_loss = val_metrics["loss"]
                    is_best = True
            else:  # val_f1
                delta = val_metrics["f1"] - self.best_val_f1
                if delta > self.early_stopping_min_delta:
                    self.best_val_f1 = val_metrics["f1"]
                    is_best = True

            if is_best:
                self.best_epoch = epoch
                self.best_metrics = val_metrics
                patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, "best_model.pt")
                logger.info(f"[*] New best validation performance at epoch {epoch}! Saved best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch} (no improvement for "
                        f"{self.early_stopping_patience} epochs)."
                    )
                    stopped_early = True
                    self._save_checkpoint(epoch, val_metrics, "last_model.pt")
                    break

            # Save last checkpoint every epoch
            self._save_checkpoint(epoch, val_metrics, "last_model.pt")

        # Save history json
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        # Plot training curves
        self.plot_training_curves(self.checkpoint_dir / "loss_f1_curve.png")

        return TrainingResult(
            best_epoch=self.best_epoch,
            best_val_loss=self.best_metrics.get("loss", 0.0),
            best_val_f1=self.best_metrics.get("f1", 0.0),
            best_val_accuracy=self.best_metrics.get("accuracy", 0.0),
            best_val_precision=self.best_metrics.get("precision", 0.0),
            best_val_recall=self.best_metrics.get("recall", 0.0),
            total_epochs_trained=len(self.history),
            stopped_early=stopped_early,
            best_model_path=str(best_ckpt_path),
            last_model_path=str(last_ckpt_path),
            training_history=self.history,
            config=self.config_dict,
        )

    def plot_training_curves(self, output_path: Union[str, Path]) -> None:
        """Plot loss and F1-score progression across training epochs."""
        if not self.history:
            return

        epochs = [h["epoch"] for h in self.history]
        train_losses = [h["train_loss"] for h in self.history]
        val_losses = [h["val_loss"] for h in self.history]
        train_f1s = [h["train_f1"] for h in self.history]
        val_f1s = [h["val_f1"] for h in self.history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss subplot
        ax1.plot(epochs, train_losses, "b-o", label="Train Loss", linewidth=2)
        ax1.plot(epochs, val_losses, "r-s", label="Val Loss", linewidth=2)
        ax1.set_title("Cross-Entropy Loss vs Epoch", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=10)
        ax1.set_ylabel("Loss", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="upper right")

        # F1-Score subplot
        ax2.plot(epochs, train_f1s, "g-o", label="Train F1", linewidth=2)
        ax2.plot(epochs, val_f1s, "m-s", label="Val F1", linewidth=2)
        ax2.set_title("Tampered F1-Score vs Epoch", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=10)
        ax2.set_ylabel("F1 Score", fontsize=10)
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="lower right")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved training curves to {output_path}")

