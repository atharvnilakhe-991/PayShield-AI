"""
Phase 19: Grouped Stratified 5-Fold Local Visual Branch Training.
Trains ResNet-18 on candidate crops across 5 folds.
- Clean and edited images of the same pair are strictly in the same fold (GroupKFold by pair_id).
- Edit categories are stratified across folds.
- Checkpoints are saved strictly based on fold validation performance (NEVER on bank_progress validation set).
- Generates full Out-of-Fold (OOF) predictions for all training crops and images.
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
from typing import Any, Dict, List, Tuple

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
logger = logging.getLogger("phase19.train_5fold")

EXP_DIR = PROJECT_ROOT / "experiments" / "phase19_local_branch_redevelopment"
DATA_DIR = EXP_DIR / "data"
CKPT_DIR = EXP_DIR / "checkpoints"
RESULTS_DIR = EXP_DIR / "results"

SEED = 20261104
BATCH_SIZE = 32
MAX_EPOCHS = 15
PATIENCE = 5
LR = 1e-4
WEIGHT_DECAY = 1e-2


def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    desc_idx = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_idx]
    y_scores_sorted = y_scores[desc_idx]

    distinct_idx = np.where(np.diff(y_scores_sorted))[0]
    threshold_idxs = np.r_[distinct_idx, y_true.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if fps[-1] <= 0 or tps[-1] <= 0:
        return 0.5

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def compute_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    desc_idx = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_idx]
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    recalls = tps / max(1, tps[-1])
    precisions = tps / np.maximum(1, tps + fps)
    recalls = np.r_[0.0, recalls]
    precisions = np.r_[1.0, precisions]
    return float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))


class CropDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], transform: T.Compose) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        r = self.rows[idx]
        img_path = EXP_DIR / r["crop_rel_path"]
        with Image.open(img_path) as src:
            img = src.convert("RGB")
        tensor = self.transform(img)
        label = int(r["crop_label"])
        return tensor, label, idx


def get_transforms():
    train_transform = T.Compose([
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def build_model(device: torch.device) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while chunk := fp.read(65536):
            h.update(chunk)
    return h.hexdigest()


def train_single_fold(
    fold_idx: int,
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    device: torch.device,
) -> Tuple[Path, List[Dict[str, Any]], Dict[str, Any]]:
    logger.info("=== Training Fold %d: %d train crops, %d val crops ===", fold_idx, len(train_rows), len(val_rows))

    train_labels = [int(r["crop_label"]) for r in train_rows]
    n_clean = sum(1 for l in train_labels if l == 0)
    n_tamp = sum(1 for l in train_labels if l == 1)
    weight_pos = float(n_clean) / max(1.0, float(n_tamp))
    logger.info("Fold %d Class counts: clean=%d, tampered=%d (pos_weight=%.2f)", fold_idx, n_clean, n_tamp, weight_pos)

    weights = torch.tensor([1.0, weight_pos], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_trans, val_trans = get_transforms()
    train_loader = DataLoader(CropDataset(train_rows, train_trans), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CropDataset(val_rows, val_trans), batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    ckpt_path = CKPT_DIR / f"best_crop_resnet18_fold{fold_idx}.pt"
    best_val_auc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = []
    t_start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        t_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(y)
        scheduler.step()
        train_loss = t_loss / len(train_rows)

        # Evaluate on fold validation
        model.eval()
        v_loss = 0.0
        val_preds_prob = []
        val_targets = []

        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                v_loss += loss.item() * len(y)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                val_preds_prob.extend(probs.tolist())
                val_targets.extend(y.cpu().numpy().tolist())

        val_loss = v_loss / len(val_rows)
        val_auc = compute_roc_auc(np.array(val_targets), np.array(val_preds_prob))
        val_pr_auc = compute_pr_auc(np.array(val_targets), np.array(val_preds_prob))

        logger.info(
            "Fold %d Epoch %02d: train_loss=%.4f, val_loss=%.4f, val_auc=%.4f, val_pr_auc=%.4f",
            fold_idx, epoch, train_loss, val_loss, val_auc, val_pr_auc
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_roc_auc": round(val_auc, 4),
            "val_pr_auc": round(val_pr_auc, 4),
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_roc_auc": val_auc,
                "val_loss": val_loss,
                "val_pr_auc": val_pr_auc,
                "fold": fold_idx,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("Fold %d early stopping at epoch %d (best epoch %d, AUC=%.4f)", fold_idx, epoch, best_epoch, best_val_auc)
                break

    # Load best model for OOF inference
    logger.info("Fold %d loading best checkpoint from epoch %d (val AUC=%.4f)", fold_idx, best_epoch, best_val_auc)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    oof_crop_records = []
    with torch.no_grad():
        for x, y, idx_batch in val_loader:
            x = x.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            for p, lbl, idx_val in zip(probs, y.numpy(), idx_batch.numpy()):
                meta_r = val_rows[int(idx_val)]
                oof_crop_records.append({
                    "crop_id": meta_r["crop_id"],
                    "sample_id": meta_r["sample_id"],
                    "pair_id": meta_r["pair_id"],
                    "fold": fold_idx,
                    "image_label": meta_r["image_label"],
                    "crop_label": int(meta_r["crop_label"]),
                    "field_name": meta_r["field_name"],
                    "edit_category": meta_r["edit_category"],
                    "layout_family": meta_r["layout_family"],
                    "p_crop_tampered": round(float(p), 5),
                    "is_routing_hit": meta_r["is_routing_hit"],
                })

    training_duration_s = round(time.time() - t_start, 2)
    val_labels = [int(r["crop_label"]) for r in val_rows]
    fold_metrics = {
        "fold": fold_idx,
        "train_sample_count": len(train_rows),
        "held_out_sample_count": len(val_rows),
        "train_positive_count": sum(1 for l in train_labels if l == 1),
        "train_negative_count": sum(1 for l in train_labels if l == 0),
        "held_out_positive_count": sum(1 for l in val_labels if l == 1),
        "held_out_negative_count": sum(1 for l in val_labels if l == 0),
        "epoch_count": len(history),
        "best_epoch": best_epoch,
        "best_held_out_val_loss": round(best_val_loss, 4),
        "best_held_out_val_roc_auc": round(best_val_auc, 4),
        "checkpoint_path": str(ckpt_path.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(ckpt_path),
        "training_duration_seconds": training_duration_s,
        "failed_or_skipped_samples": 0,
        "history": history,
    }
    return ckpt_path, oof_crop_records, fold_metrics


def aggregate_image_predictions(oof_crops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_sample: Dict[str, List[Dict[str, Any]]] = {}
    for c in oof_crops:
        by_sample.setdefault(c["sample_id"], []).append(c)

    image_records = []
    for sample_id, crops in by_sample.items():
        first = crops[0]
        probs = [c["p_crop_tampered"] for c in crops]
        probs_sorted = sorted(probs, reverse=True)

        p_max = float(np.max(probs))
        p_top2 = float(np.mean(probs_sorted[:2])) if len(probs_sorted) >= 2 else p_max

        # Field-specific probs
        field_probs = {c["field_name"]: c["p_crop_tampered"] for c in crops}
        p_amt = field_probs.get("amount", 0.0)
        p_rec = field_probs.get("receiver", 0.0)
        p_txn = field_probs.get("transaction_id", 0.0)
        p_date = field_probs.get("date", 0.0)

        # Field weighted: amount and receiver have higher prior tampering probability
        p_weighted = 0.35 * p_amt + 0.30 * p_rec + 0.20 * p_txn + 0.15 * p_date

        image_records.append({
            "sample_id": sample_id,
            "pair_id": first["pair_id"],
            "fold": first["fold"],
            "true_label": 0 if first["image_label"] == "clean" else 1,
            "image_label": first["image_label"],
            "edit_category": first["edit_category"],
            "layout_family": first["layout_family"],
            "crop_count": len(crops),
            "p_visual_max": round(p_max, 5),
            "p_visual_top2": round(p_top2, 5),
            "p_visual_weighted": round(p_weighted, 5),
            "p_amt": round(p_amt, 5),
            "p_rec": round(p_rec, 5),
            "p_txn": round(p_txn, 5),
            "p_date": round(p_date, 5),
        })
    return image_records


def main():
    set_seed(SEED)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    manifest_path = DATA_DIR / "crop_dataset_manifest.csv"
    with open(manifest_path) as fp:
        all_crops = list(csv.DictReader(fp))

    # Strictly exclude ambiguous crops from training and fold validation
    train_crops = [r for r in all_crops if r["split"] == "train" and r.get("is_ambiguous") != "True" and int(r["crop_label"]) in (0, 1)]
    logger.info("Loaded %d unambiguous training candidate crops across 5 folds", len(train_crops))

    all_oof_crops = []
    all_fold_metrics = []

    for fold_idx in range(5):
        f_train = [r for r in train_crops if int(r["fold"]) != fold_idx]
        f_val = [r for r in train_crops if int(r["fold"]) == fold_idx]

        ckpt_path, oof_crops, f_metrics = train_single_fold(fold_idx, f_train, f_val, device)
        all_oof_crops.extend(oof_crops)
        all_fold_metrics.append(f_metrics)

    # Save OOF Crop Predictions
    oof_crop_csv = RESULTS_DIR / "oof_crop_predictions.csv"
    with open(oof_crop_csv, "w", newline="") as fp:
        fields = list(all_oof_crops[0].keys())
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_oof_crops)
    logger.info("Saved OOF crop predictions to %s", oof_crop_csv)

    # Aggregate Image Predictions
    image_records = aggregate_image_predictions(all_oof_crops)
    oof_img_csv = RESULTS_DIR / "oof_image_predictions.csv"
    with open(oof_img_csv, "w", newline="") as fp:
        fields = list(image_records[0].keys())
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(image_records)
    logger.info("Saved OOF image predictions to %s (%d images)", oof_img_csv, len(image_records))

    # Compute overall OOF metrics for image-level visual aggregation
    y_true_img = np.array([r["true_label"] for r in image_records])
    for pool_strategy in ["p_visual_max", "p_visual_top2", "p_visual_weighted"]:
        y_scores = np.array([r[pool_strategy] for r in image_records])
        auc = compute_roc_auc(y_true_img, y_scores)
        pr_auc = compute_pr_auc(y_true_img, y_scores)
        logger.info("OOF Image-Level Aggregation [%s]: ROC-AUC=%.4f, PR-AUC=%.4f", pool_strategy, auc, pr_auc)

    # Save provenance JSON
    prov_file = CKPT_DIR / "fold_checkpoint_provenance.json"
    with open(prov_file, "w") as fp:
        json.dump(all_fold_metrics, fp, indent=2)
    logger.info("Saved checkpoint provenance to %s", prov_file)


if __name__ == "__main__":
    main()

