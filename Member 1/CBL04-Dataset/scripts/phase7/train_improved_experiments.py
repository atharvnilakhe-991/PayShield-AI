"""
Controlled CNN Improvement Experiments for CBL-04 Phase 4.
Evaluates baseline deficiencies, trains improved candidate models using leave-one-family-out folds,
and records comprehensive metrics in models/phase9_improved/ and reports/phase9_improved/.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import build_model  # noqa: E402

SEED = 20261207
HEIGHT, WIDTH = 320, 160
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


class AugmentedScreenshotDataset(Dataset):
    """
    Dataset loader matching Phase 7 preprocessing with optional subtle photometric augmentation.
    """

    def __init__(self, rows: list[dict[str, str]], augment: bool = False) -> None:
        self.rows = rows
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with Image.open(ROOT / row["image_path"]) as src:
            image = src.convert("RGB")
        with Image.open(ROOT / row["mask_path"]) as src:
            mask = src.convert("L")

        # Subtle photometric augmentation for training (contrast/brightness)
        if self.augment:
            if random.random() > 0.5:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(random.uniform(0.95, 1.05))
            if random.random() > 0.5:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(random.uniform(0.95, 1.05))

        scale = min(WIDTH / image.width, HEIGHT / image.height)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
        mask = mask.resize(size, Image.Resampling.BOX).point(lambda v: 255 if v > 0 else 0)

        canvas = Image.new("RGB", (WIDTH, HEIGHT), (18, 18, 18))
        canvas.paste(image, (0, 0))
        mask_canvas = Image.new("L", (WIDTH, HEIGHT), 0)
        mask_canvas.paste(mask, (0, 0))

        rgb = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8).reshape(HEIGHT, WIDTH, 3)
        x = rgb.permute(2, 0, 1).float() / 255.0
        x = (x - MEAN) / STD

        y_mask = torch.frombuffer(bytearray(mask_canvas.tobytes()), dtype=torch.uint8).reshape(HEIGHT, WIDTH)
        y_mask = (y_mask > 127).float().unsqueeze(0)

        label = 0 if row["label"] == "clean" else 1
        return {
            "image": x,
            "mask": y_mask,
            "label": torch.tensor(label, dtype=torch.long),
            "source_id": row["source_id"],
            "pair_id": row["pair_id"],
            "family": row["layout_family"],
            "edit_category": row["edit_category"],
            "image_path": row["image_path"],
        }


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()


def calculate_loss(
    out: dict[str, torch.Tensor],
    labels: torch.Tensor,
    masks: torch.Tensor,
    seg_weight: float = 0.5,
    pos_weight: float = 12.0,
    device: torch.device = DEVICE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cls_loss = F.cross_entropy(out["logits"], labels)
    bce = F.binary_cross_entropy_with_logits(
        out["mask_logits"], masks, pos_weight=torch.tensor(pos_weight, device=device)
    )
    dice = dice_loss(out["mask_logits"], masks)
    seg_loss = bce + dice
    total_loss = cls_loss + seg_weight * seg_loss
    return total_loss, cls_loss, seg_loss


def compute_metrics(y_true: list[int], y_pred: list[int], scores: list[float] | None = None) -> dict[str, Any]:
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    tn = sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_edited = 2 * precision * recall / max(1e-12, precision + recall)

    # Clean class metrics (to calculate balanced Macro-F1)
    precision_clean = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_clean = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_clean = 2 * precision_clean * recall_clean / max(1e-12, precision_clean + recall_clean)

    macro_f1 = (f1_clean + f1_edited) / 2.0
    accuracy = (tp + tn) / max(1, len(y_true))
    specificity = recall_clean
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    result = {
        "support": len(y_true),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_edited,
        "f1_clean": f1_clean,
        "macro_f1": macro_f1,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }

    if scores and len(scores) == len(y_true):
        clean_scores = [s for s, y in zip(scores, y_true) if y == 0]
        edited_scores = [s for s, y in zip(scores, y_true) if y == 1]
        mean_clean = sum(clean_scores) / len(clean_scores) if clean_scores else 0.0
        mean_edited = sum(edited_scores) / len(edited_scores) if edited_scores else 0.0
        result["mean_clean_score"] = mean_clean
        result["mean_edited_score"] = mean_edited
        result["score_separation"] = mean_edited - mean_clean

    return result


def compute_localization_metrics(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dice = 2 * tp / max(1, 2 * tp + fp + fn)
    iou = tp / max(1, tp + fp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
    }


def evaluate_dataset(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    pix: dict[str, int] = defaultdict(int)

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            out = model(x)
            probs = torch.softmax(out["logits"], dim=1)[:, 1].cpu().tolist()
            mask_pred = torch.sigmoid(out["mask_logits"]) >= 0.5
            target = (batch["mask"] >= 0.5).to(device)

            for i, prob in enumerate(probs):
                true_label = int(batch["label"][i].item())
                pred_label = int(prob >= 0.5)
                a, b = mask_pred[i, 0], target[i, 0]
                tp = int((a & b).sum().item())
                fp = int((a & ~b).sum().item())
                fn = int((~a & b).sum().item())
                pix["tp"] += tp
                pix["fp"] += fp
                pix["fn"] += fn
                predictions.append({
                    "source_id": batch["source_id"][i],
                    "pair_id": batch["pair_id"][i],
                    "layout_family": batch["family"][i],
                    "edit_category": batch["edit_category"][i],
                    "image_path": batch["image_path"][i],
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "edited_probability": prob,
                    "mask_tp_pixels": tp,
                    "mask_fp_pixels": fp,
                    "mask_fn_pixels": fn,
                })
    return predictions, pix


def run_experiment(
    exp_id: str,
    description: str,
    fold: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seg_weight: float,
    pos_weight: float,
    augment: bool,
    device: torch.device = DEVICE,
) -> dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Starting {exp_id} (Fold {fold}): {description}")
    print(f"Device: {device}, Epochs: {epochs}, BatchSize: {batch_size}, LR: {lr}, SegWeight: {seg_weight}")
    print(f"=======================================================")

    random.seed(SEED + fold)
    torch.manual_seed(SEED + fold)

    fold_dir = ROOT / "metadata/phase7/folds"
    train_rows = read_csv(fold_dir / f"fold_{fold}_train.csv")
    val_rows = read_csv(fold_dir / f"fold_{fold}_validation.csv")
    test_rows = read_csv(fold_dir / f"fold_{fold}_test.csv")

    train_ds = AugmentedScreenshotDataset(train_rows, augment=augment)
    val_ds = AugmentedScreenshotDataset(val_rows, augment=False)
    test_ds = AugmentedScreenshotDataset(test_rows, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    cfg_path = ROOT / "templates/phase7/model_config.json"
    model_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    model_cfg["architecture"]["pretrained"] = False

    model = build_model(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_macro_f1 = -1.0
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_epoch = 0
    history = []

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        # Training loop
        model.train()
        train_loss = train_cls_loss = train_seg_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            m = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss, cls_loss, seg_loss = calculate_loss(
                out, y, m, seg_weight=seg_weight, pos_weight=pos_weight, device=device
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            bs = y.shape[0]
            train_loss += loss.item() * bs
            train_cls_loss += cls_loss.item() * bs
            train_seg_loss += seg_loss.item() * bs
            n_batches += bs

        scheduler.step()
        train_loss /= max(1, n_batches)
        train_cls_loss /= max(1, n_batches)
        train_seg_loss /= max(1, n_batches)

        # Validation loop
        val_preds, val_pix = evaluate_dataset(model, val_loader, device)
        val_y_true = [p["true_label"] for p in val_preds]
        val_y_pred = [p["predicted_label"] for p in val_preds]
        val_scores = [p["edited_probability"] for p in val_preds]
        vm = compute_metrics(val_y_true, val_y_pred, val_scores)
        v_loc = compute_localization_metrics(val_pix)

        val_score = vm["macro_f1"]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_cls_loss": train_cls_loss,
            "val_accuracy": vm["accuracy"],
            "val_f1_edited": vm["f1"],
            "val_f1_clean": vm["f1_clean"],
            "val_macro_f1": val_score,
            "val_confusion": vm["confusion_matrix"],
            "val_loc_iou": v_loc["iou"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(
            f"Epoch {epoch:2d}/{epochs:2d} | Train Cls: {train_cls_loss:.4f} | "
            f"Val Acc: {vm['accuracy']:.4f} | Macro F1: {val_score:.4f} (F1_ed: {vm['f1']:.3f}, F1_cl: {vm['f1_clean']:.3f}) | "
            f"Val CM: {vm['confusion_matrix']}"
        )

        # Macro-F1 Model Selection (Prevents single-class collapse!)
        if val_score > best_val_macro_f1:
            best_val_macro_f1 = val_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_duration = round(time.time() - start_time, 2)
    print(f"Training completed in {train_duration}s. Best Epoch: {best_epoch} with Val Macro-F1: {best_val_macro_f1:.4f}")

    if best_state is None:
        raise RuntimeError("No model state was captured during training.")

    # Save candidate checkpoint to models/phase9_improved/ (NEVER models/phase7/)
    ckpt_dir = ROOT / "models/phase9_improved"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_filename = f"{exp_id}_fold_{fold}_best.pt"
    ckpt_path = ckpt_dir / ckpt_filename

    checkpoint_data = {
        "experiment_id": exp_id,
        "description": description,
        "fold": fold,
        "seed": SEED + fold,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "model_config": model_cfg,
        "state_dict": best_state,
        "training_history": history,
        "training_hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seg_weight": seg_weight,
            "pos_weight": pos_weight,
            "augment": augment,
            "selection_metric": "val_macro_f1",
        },
    }
    torch.save(checkpoint_data, ckpt_path)
    print(f"Saved candidate checkpoint to {ckpt_path}")

    # Evaluate on held-out Test Set
    model.load_state_dict(best_state)
    test_preds, test_pix = evaluate_dataset(model, test_loader, device)
    test_y_true = [p["true_label"] for p in test_preds]
    test_y_pred = [p["predicted_label"] for p in test_preds]
    test_scores = [p["edited_probability"] for p in test_preds]

    test_metrics = compute_metrics(test_y_true, test_y_pred, test_scores)
    test_loc = compute_localization_metrics(test_pix)
    test_family = test_rows[0]["layout_family"]

    print(f"\n--- HELD-OUT TEST EVALUATION (Family: {test_family}, N={len(test_preds)}) ---")
    print(f"Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"Precision:   {test_metrics['precision']:.4f}")
    print(f"Recall:      {test_metrics['recall']:.4f}")
    print(f"F1 (Edited): {test_metrics['f1']:.4f}")
    print(f"F1 (Clean):  {test_metrics['f1_clean']:.4f}")
    print(f"Macro F1:    {test_metrics['macro_f1']:.4f}")
    print(f"FPR:         {test_metrics['fpr']:.4f}")
    print(f"FNR:         {test_metrics['fnr']:.4f}")
    print(f"Confusion:   {test_metrics['confusion_matrix']}")
    print(f"Mask IoU:    {test_loc['iou']:.4f}")
    print(f"Clean Score: {test_metrics.get('mean_clean_score', 0):.4f} | Edited Score: {test_metrics.get('mean_edited_score', 0):.4f}")
    print(f"Separation:  {test_metrics.get('score_separation', 0):.4f}")

    return {
        "experiment_id": exp_id,
        "description": description,
        "fold": fold,
        "test_family": test_family,
        "checkpoint_file": ckpt_filename,
        "best_epoch": best_epoch,
        "epochs": epochs,
        "best_val_macro_f1": round(best_val_macro_f1, 4),
        "test_accuracy": round(test_metrics["accuracy"], 4),
        "test_precision": round(test_metrics["precision"], 4),
        "test_recall": round(test_metrics["recall"], 4),
        "test_f1": round(test_metrics["f1"], 4),
        "test_macro_f1": round(test_metrics["macro_f1"], 4),
        "test_fpr": round(test_metrics["fpr"], 4),
        "test_fnr": round(test_metrics["fnr"], 4),
        "test_tn": test_metrics["confusion_matrix"]["tn"],
        "test_fp": test_metrics["confusion_matrix"]["fp"],
        "test_fn": test_metrics["confusion_matrix"]["fn"],
        "test_tp": test_metrics["confusion_matrix"]["tp"],
        "test_mask_iou": round(test_loc["iou"], 4),
        "test_mask_dice": round(test_loc["dice"], 4),
        "mean_clean_score": round(test_metrics.get("mean_clean_score", 0), 4),
        "mean_edited_score": round(test_metrics.get("mean_edited_score", 0), 4),
        "score_separation": round(test_metrics.get("score_separation", 0), 4),
        "training_duration_s": train_duration,
        "predictions": test_preds,
    }


def main() -> None:
    print(f"Starting Phase 4 Controlled Experiments on device: {DEVICE}")

    experiments_to_run = [
        # Experiment 1: Extended epochs (12) + Macro-F1 model selection (Fixes false-positive collapse)
        {
            "exp_id": "EXP1_MACRO_F1",
            "description": "12 Epochs + Macro-F1 Checkpoint Selection + Cosine LR",
            "fold": 1,
            "epochs": 12,
            "batch_size": 16,
            "lr": 2e-4,
            "seg_weight": 0.5,
            "pos_weight": 12.0,
            "augment": False,
        },
        {
            "exp_id": "EXP1_MACRO_F1",
            "description": "12 Epochs + Macro-F1 Checkpoint Selection + Cosine LR",
            "fold": 2,
            "epochs": 12,
            "batch_size": 16,
            "lr": 2e-4,
            "seg_weight": 0.5,
            "pos_weight": 12.0,
            "augment": False,
        },
        # Experiment 2: Classification Priority (Lower seg weight 0.20, pos_weight 4.0)
        {
            "exp_id": "EXP2_CLS_PRIORITY",
            "description": "Classification-Prioritized Loss (seg_weight=0.20, pos_weight=4.0)",
            "fold": 1,
            "epochs": 12,
            "batch_size": 16,
            "lr": 2e-4,
            "seg_weight": 0.20,
            "pos_weight": 4.0,
            "augment": False,
        },
        {
            "exp_id": "EXP2_CLS_PRIORITY",
            "description": "Classification-Prioritized Loss (seg_weight=0.20, pos_weight=4.0)",
            "fold": 2,
            "epochs": 12,
            "batch_size": 16,
            "lr": 2e-4,
            "seg_weight": 0.20,
            "pos_weight": 4.0,
            "augment": False,
        },
    ]

    all_results = []
    all_test_preds = []

    for spec in experiments_to_run:
        res = run_experiment(**spec)
        preds = res.pop("predictions")
        all_results.append(res)
        all_test_preds.extend(preds)

    # Save summary CSV
    reports_dir = ROOT / "reports/phase9_improved"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "experiment_results.csv"
    write_csv(csv_path, all_results)
    print(f"\nSaved experiment summary to {csv_path}")

    json_path = reports_dir / "experiment_results.json"
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Saved experiment details to {json_path}")


if __name__ == "__main__":
    main()

