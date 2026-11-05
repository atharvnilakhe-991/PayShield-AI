"""
Phase 17 Experimental Local Branch Training Script.
Trains a lightweight ResNet-18 model on native-resolution text crops.
Uses pair-grouped development data (434 train pairs), validates strictly on 54 validation pairs.
Saves checkpoint to experiments/phase17_visual_forensics/checkpoints/ (completely isolated from production models).
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase17.train")

SEED = 20261104
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4

# Deterministic seeding
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def roc_auc_score(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    y_scores_sorted = y_scores[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_scores_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if fps[-1] <= 0 or tps[-1] <= 0:
        return 0.5

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


class CropDataset(Dataset):
    def __init__(self, rows: list[dict], transform: T.Compose) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        r = self.rows[idx]
        img_path = PROJECT_ROOT / r["crop_path"]
        with Image.open(img_path) as src:
            img = src.convert("RGB")
        tensor = self.transform(img)
        label = int(r["true_label"])
        return tensor, label, r["sample_id"]


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    manifest_path = PROJECT_ROOT / "experiments" / "phase17_visual_forensics" / "data" / "crop_dataset_manifest.csv"
    with open(manifest_path) as fp:
        all_rows = list(csv.DictReader(fp))

    train_rows = [r for r in all_rows if r["split"] == "train"]
    val_rows = [r for r in all_rows if r["split"] == "val"]
    logger.info("Train crops: %d, Val crops: %d", len(train_rows), len(val_rows))

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(CropDataset(train_rows, transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CropDataset(val_rows, transform), batch_size=BATCH_SIZE, shuffle=False)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    ckpt_dir = PROJECT_ROOT / "experiments" / "phase17_visual_forensics" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best_crop_resnet18.pt"

    best_val_roc = 0.0
    history = []

    start_train_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            preds = torch.argmax(out, dim=1)
            train_correct += (preds == y).sum().item()
            train_total += len(y)

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / train_total

        # Validation
        model.eval()
        val_probs = []
        val_labels = []
        val_correct = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                out = model(x)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_labels.extend(y.numpy())
                preds = np.argmax(out.cpu().numpy(), axis=1)
                val_correct += np.sum(preds == y.numpy())

        val_labels_arr = np.array(val_labels)
        val_probs_arr = np.array(val_probs)
        val_acc = val_correct / len(val_labels)
        val_roc = roc_auc_score(val_labels_arr, val_probs_arr)

        logger.info(
            "Epoch %2d/%d | Train Loss: %.4f, Train Acc: %.4f | Val Acc: %.4f, Val ROC-AUC: %.4f",
            epoch, EPOCHS, avg_train_loss, train_acc, val_acc, val_roc
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "val_roc_auc": round(val_roc, 4),
        })

        if val_roc > best_val_roc or epoch == 1:
            best_val_roc = val_roc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_roc_auc": val_roc,
                "seed": SEED,
                "architecture": "ResNet18_Crop_Classifier",
            }, best_ckpt_path)

    total_time = time.time() - start_train_time
    logger.info("Training completed in %.2fs. Best Val ROC-AUC: %.4f", total_time, best_val_roc)

    # Compute SHA-256 hash of saved checkpoint
    ckpt_bytes = best_ckpt_path.read_bytes()
    ckpt_hash = hashlib.sha256(ckpt_bytes).hexdigest()
    logger.info("Saved best checkpoint to %s (SHA256: %s)", best_ckpt_path, ckpt_hash)

    # Save training log
    summary_path = ckpt_dir / "training_summary.json"
    with open(summary_path, "w") as fp:
        json.dump({
            "training_time_seconds": round(total_time, 2),
            "best_epoch": max(history, key=lambda x: x["val_roc_auc"])["epoch"],
            "best_val_roc_auc": round(best_val_roc, 4),
            "checkpoint_path": str(best_ckpt_path.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": ckpt_hash,
            "history": history,
        }, fp, indent=2)


if __name__ == "__main__":
    main()

