#!/usr/bin/env python3
"""
Phase 20: Train and Evaluate Dense MIL Forensics Models
Evaluates:
- Model C: Dense 4x2 Patch Mean Pooling
- Model D: Dense 4x2 Patch Max Pooling
- Model E: Dense 4x2 Patch Attention MIL (Ilse et al. 2018)
- Model F: Multi-Scale Attention MIL (Scale 1 [8 patches] + Scale 2 [18 patches] = 26 patches)

Implements Grouped Stratified 5-Fold Cross-Validation over 434 pairs (868 screenshots),
generates complete OOF predictions, extracts attention weights, and evaluates on
frozen 108 bank_progress validation screenshots.
"""

import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WORKSPACE_ROOT = Path("/Users/heemanshupjain/Downloads/Member 3")
OUTPUT_DIR = WORKSPACE_ROOT / "experiments" / "phase20_dense_mil_forensics"
DATA_DIR = OUTPUT_DIR / "data"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "results"
MANIFEST_PATH = DATA_DIR / "patch_manifest.csv"
FOLD_ASSIGNMENTS_PATH = WORKSPACE_ROOT / "experiments" / "phase19_local_branch_redevelopment" / "data" / "fold_assignments.csv"
TRAIN_MANIFEST_PATH = WORKSPACE_ROOT / "Hackathon-VIT" / "CBL04-Dataset" / "metadata" / "phase6" / "train_manifest.csv"
VAL_MANIFEST_PATH = WORKSPACE_ROOT / "Hackathon-VIT" / "CBL04-Dataset" / "metadata" / "phase6" / "validation_manifest.csv"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------
# Architectures
# ---------------------------------------------------------

class MeanPoolingMIL(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feats, scales=None):
        # feats: [B, K, 512]
        bag_feat = torch.mean(feats, dim=1)  # [B, 512]
        logits = self.classifier(bag_feat)  # [B, 1]
        # Uniform attention for logging
        B, K, _ = feats.shape
        attn = torch.full((B, K), 1.0 / K, device=feats.device)
        return logits, attn


class MaxPoolingMIL(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feats, scales=None):
        # feats: [B, K, 512]
        bag_feat, max_indices = torch.max(feats, dim=1)  # [B, 512]
        logits = self.classifier(bag_feat)  # [B, 1]
        B, K, _ = feats.shape
        attn = torch.zeros((B, K), device=feats.device)
        return logits, attn


class GatedAttentionMIL(nn.Module):
    """
    Gated Attention Multiple Instance Learning (Ilse et al., ICML 2018).
    a_i = softmax( w^T (tanh(V h_i) * sigm(U h_i)) )
    """
    def __init__(self, in_dim=512, hidden_dim=128):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feats, scales=None):
        # feats: [B, K, 512]
        v = self.attention_V(feats)  # [B, K, hidden_dim]
        u = self.attention_U(feats)  # [B, K, hidden_dim]
        gated = v * u  # [B, K, hidden_dim]
        attn_logits = self.attention_w(gated).squeeze(-1)  # [B, K]
        attn_weights = F.softmax(attn_logits, dim=1)  # [B, K]

        bag_feat = torch.bmm(attn_weights.unsqueeze(1), feats).squeeze(1)  # [B, 512]
        logits = self.classifier(bag_feat)  # [B, 1]
        return logits, attn_weights


class MultiScaleAttentionMIL(nn.Module):
    """
    Multi-Scale Gated Attention MIL with learned scale conditioning embeddings.
    Embeds coarse (Scale 1: 4x2) and fine (Scale 2: 6x3) patches into a unified bag.
    """
    def __init__(self, in_dim=512, hidden_dim=128, n_scales=2):
        super().__init__()
        self.scale_embed = nn.Embedding(n_scales, in_dim)
        self.attention_V = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feats, scales):
        # feats: [B, K, 512], scales: [B, K]
        s_emb = self.scale_embed(scales)  # [B, K, 512]
        conditioned_feats = feats + s_emb

        v = self.attention_V(conditioned_feats)
        u = self.attention_U(conditioned_feats)
        gated = v * u
        attn_logits = self.attention_w(gated).squeeze(-1)  # [B, K]
        attn_weights = F.softmax(attn_logits, dim=1)  # [B, K]

        bag_feat = torch.bmm(attn_weights.unsqueeze(1), conditioned_feats).squeeze(1)
        logits = self.classifier(bag_feat)
        return logits, attn_weights


# ---------------------------------------------------------
# Metrics Calculation (Pure Python / Numpy)
# ---------------------------------------------------------

def compute_roc_pr_auc(y_true, y_scores):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_scores = np.asarray(y_scores, dtype=np.float64)

    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5

    # Sort descending
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_score_indices]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    # ROC AUC
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    tpr = tps / n_pos
    fpr = fps / n_neg

    tpr = np.r_[0.0, tpr, 1.0]
    fpr = np.r_[0.0, fpr, 1.0]
    # Pure numpy trapezoid integration (compatible with NumPy 1.x and 2.x)
    roc_auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))

    # PR AUC
    precision = tps / (tps + fps)
    recall = tps / n_pos
    recall_full = np.r_[0.0, recall]
    precision_full = np.r_[1.0, precision]
    pr_auc = float(np.sum((recall_full[1:] - recall_full[:-1]) * (precision_full[1:] + precision_full[:-1]) / 2.0))

    return roc_auc, pr_auc


def compute_binary_metrics_at_threshold(y_true, y_scores, thresh=0.5):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    y_pred = (y_scores >= thresh).astype(np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    n_pos = tp + fn
    n_neg = tn + fp
    total = n_pos + n_neg

    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / n_pos if n_pos > 0 else 0.0
    spec = tn / n_neg if n_neg > 0 else 0.0
    fpr = fp / n_neg if n_neg > 0 else 0.0
    fnr = fn / n_pos if n_pos > 0 else 0.0

    prec_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_neg = tn / n_neg if n_neg > 0 else 0.0
    f1_pos = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    f1_neg = 2 * prec_neg * rec_neg / (prec_neg + rec_neg) if (prec_neg + rec_neg) > 0 else 0.0
    macro_f1 = (f1_pos + f1_neg) / 2.0

    return {
        "threshold": thresh,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total": total, "n_pos": n_pos, "n_neg": n_neg,
        "accuracy": acc, "precision": prec, "recall": rec,
        "specificity": spec, "fpr": fpr, "fnr": fnr,
        "macro_f1": macro_f1
    }


def compute_decisive_coverage(y_scores, lower=0.45, upper=0.55):
    y_scores = np.asarray(y_scores, dtype=np.float64)
    total = len(y_scores)
    decisive = np.sum((y_scores < lower) | (y_scores > upper))
    coverage = float(decisive / total) if total > 0 else 0.0
    return decisive, total, coverage


print("Module train_and_evaluate_mil imported successfully.")

# ---------------------------------------------------------
# Bag Construction and DataLoader
# ---------------------------------------------------------

class ScreenshotBagDataset(torch.utils.data.Dataset):
    def __init__(self, items, features, is_multiscale=False):
        """
        items: list of screenshot metadata dicts
        features: Tensor of shape [TotalPatches, 512]
        is_multiscale: bool, whether to include Scale-1 + Scale-2 (26 patches) or only Scale-1 (8 patches)
        """
        self.items = items
        self.features = features
        self.is_multiscale = is_multiscale

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        if self.is_multiscale:
            indices = item["scale1_patch_indices"] + item["scale2_patch_indices"]
            scales = torch.tensor([0]*len(item["scale1_patch_indices"]) + [1]*len(item["scale2_patch_indices"]), dtype=torch.long)
        else:
            indices = item["scale1_patch_indices"]
            scales = torch.zeros(len(indices), dtype=torch.long)

        bag_feats = self.features[indices]  # [K, 512]
        label = torch.tensor([item["label_num"]], dtype=torch.float32)
        return bag_feats, scales, label, idx


# ---------------------------------------------------------
# Training and Evaluation Harness
# ---------------------------------------------------------

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for bag_feats, scales, labels, _ in dataloader:
        bag_feats = bag_feats.to(device)
        scales = scales.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, attn = model(bag_feats, scales)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
    return total_loss / len(dataloader.dataset)


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_scores = []
    all_labels = []
    all_attns = []

    with torch.no_grad():
        for bag_feats, scales, labels, _ in dataloader:
            bag_feats = bag_feats.to(device)
            scales = scales.to(device)
            labels = labels.to(device)

            logits, attn = model(bag_feats, scales)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            all_scores.extend(probs.tolist())
            all_labels.extend(labels.squeeze(-1).cpu().numpy().tolist())
            all_attns.extend(attn.cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    roc_auc, pr_auc = compute_roc_pr_auc(all_labels, all_scores)
    return avg_loss, roc_auc, pr_auc, np.array(all_scores), np.array(all_labels), np.array(all_attns)


def run_experiment_suite():
    set_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Executing MIL Experiment Suite on: {device}")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load manifest and organize screenshots
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        patch_rows = list(csv.DictReader(f))

    # Map image_id -> list of patch indices
    img_s1_indices = {}
    img_s2_indices = {}
    img_meta = {}

    for idx, r in enumerate(patch_rows):
        img_id = r["image_id"]
        if img_id not in img_meta:
            img_meta[img_id] = {
                "image_id": img_id,
                "pair_id": r["pair_id"],
                "split": r["split"],
                "fold": int(r["fold"]),
                "scale1_patch_indices": [],
                "scale2_patch_indices": [],
            }
        if r["scale"] == "scale_1":
            img_meta[img_id]["scale1_patch_indices"].append(idx)
        else:
            img_meta[img_id]["scale2_patch_indices"].append(idx)

    # Merge labels from train and val manifests
    train_info = {}
    with open(TRAIN_MANIFEST_PATH, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            train_info[r["image_id"]] = {
                "label": r.get("label", "clean" if "_clean" in r["image_id"] else "edited"),
                "edit_category": r.get("edit_category", ""),
                "layout_family": r.get("layout_family", ""),
            }

    val_info = {}
    with open(VAL_MANIFEST_PATH, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            val_info[r["image_id"]] = {
                "label": r.get("label", "clean" if "_clean" in r["image_id"] else "edited"),
                "edit_category": r.get("edit_category", ""),
                "layout_family": r.get("layout_family", "bank_progress"),
            }

    all_screenshots = []
    for img_id, meta in img_meta.items():
        info = train_info.get(img_id) or val_info.get(img_id, {})
        label_str = info.get("label", "clean" if "_clean" in img_id else "edited")
        meta["label_str"] = label_str
        meta["label_num"] = 1.0 if label_str == "edited" else 0.0
        meta["edit_category"] = info.get("edit_category", "")
        meta["layout_family"] = info.get("layout_family", "")
        all_screenshots.append(meta)

    # Separate train (868) and val (108)
    train_screenshots = [s for s in all_screenshots if s["split"] == "train"]
    val_screenshots = [s for s in all_screenshots if s["split"] == "val"]

    print(f"Total train screenshots: {len(train_screenshots)} (Pairs: {len(set(s['pair_id'] for s in train_screenshots))})")
    print(f"Total val screenshots: {len(val_screenshots)} (Pairs: {len(set(s['pair_id'] for s in val_screenshots))})")

    models_to_train = [
        ("model_c_mean_pooling", MeanPoolingMIL, False),
        ("model_d_max_pooling", MaxPoolingMIL, False),
        ("model_e_attention_mil", GatedAttentionMIL, False),
        ("model_f_multiscale_mil", MultiScaleAttentionMIL, True),
    ]

    all_oof_results = {m_name: {} for m_name, _, _ in models_to_train}
    all_val_predictions = {m_name: np.zeros(len(val_screenshots), dtype=np.float64) for m_name, _, _ in models_to_train}
    model_summaries = {}

    criterion = nn.BCEWithLogitsLoss()

    for m_name, model_cls, is_multiscale in models_to_train:
        print(f"\n=======================================================")
        print(f" TRAINING: {m_name.upper()} (MultiScale={is_multiscale})")
        print(f"=======================================================")

        fold_checkpoints = []
        oof_scores = {}
        oof_attns = {}

        for fold_idx in range(5):
            print(f"\n--- Fold {fold_idx} ---")
            feat_cache_path = DATA_DIR / f"patch_features_fold{fold_idx}.pt"
            if not feat_cache_path.exists():
                raise FileNotFoundError(f"Missing feature cache: {feat_cache_path}")

            feat_data = torch.load(feat_cache_path, map_location="cpu", weights_only=False)
            features = feat_data["features"]
            ckpt_provenance = feat_data["feature_extractor_checkpoint"]
            ckpt_sha256 = feat_data["checkpoint_sha256"]

            # Train / Held-out split for this fold
            train_items = [s for s in train_screenshots if s["fold"] != fold_idx]
            val_items = [s for s in train_screenshots if s["fold"] == fold_idx]

            train_ds = ScreenshotBagDataset(train_items, features, is_multiscale=is_multiscale)
            val_ds = ScreenshotBagDataset(val_items, features, is_multiscale=is_multiscale)

            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)

            model = model_cls().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)

            best_val_loss = float("inf")
            best_epoch = -1
            best_roc_auc = 0.0
            best_pr_auc = 0.0
            best_state = None
            patience = 7
            patience_counter = 0

            for epoch in range(1, 26):
                tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
                v_loss, v_roc, v_pr, _, _, _ = evaluate_model(model, val_loader, criterion, device)
                scheduler.step()

                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    best_roc_auc = v_roc
                    best_pr_auc = v_pr
                    best_epoch = epoch
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            # Save best checkpoint
            ckpt_filename = f"best_{m_name}_fold{fold_idx}.pt"
            ckpt_path = CHECKPOINTS_DIR / ckpt_filename
            torch.save({
                "model_name": m_name,
                "fold": fold_idx,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "val_roc_auc": best_roc_auc,
                "val_pr_auc": best_pr_auc,
                "state_dict": best_state,
                "feature_extractor_checkpoint": ckpt_provenance,
                "feature_extractor_sha256": ckpt_sha256,
            }, ckpt_path)

            saved_hash = compute_sha256(ckpt_path)
            fold_checkpoints.append({
                "fold": fold_idx,
                "checkpoint": ckpt_filename,
                "sha256": saved_hash,
                "best_epoch": best_epoch,
                "val_loss": best_val_loss,
                "val_roc_auc": best_roc_auc,
                "val_pr_auc": best_pr_auc,
            })
            print(f"Fold {fold_idx} Best Epoch {best_epoch} | Loss: {best_val_loss:.4f} | ROC-AUC: {best_roc_auc:.4f} | PR-AUC: {best_pr_auc:.4f} | Hash: {saved_hash[:16]}")

            # Predict on held-out fold for OOF
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            _, _, _, val_scores, _, val_attns = evaluate_model(model, val_loader, criterion, device)
            for idx, item in enumerate(val_items):
                oof_scores[item["image_id"]] = {
                    "score": float(val_scores[idx]),
                    "fold": fold_idx,
                    "checkpoint_sha256": saved_hash,
                    "attention": val_attns[idx].tolist() if len(val_attns) > 0 else []
                }

            # Predict on frozen 108 validation set
            val_frozen_ds = ScreenshotBagDataset(val_screenshots, features, is_multiscale=is_multiscale)
            val_frozen_loader = torch.utils.data.DataLoader(val_frozen_ds, batch_size=32, shuffle=False)
            _, _, _, frozen_scores, _, _ = evaluate_model(model, val_frozen_loader, criterion, device)
            all_val_predictions[m_name] += frozen_scores / 5.0

        all_oof_results[m_name] = oof_scores

        # Compute full OOF metrics across all 868 screenshots
        oof_y_true = [s["label_num"] for s in train_screenshots]
        oof_y_score = [oof_scores[s["image_id"]]["score"] for s in train_screenshots]
        oof_roc, oof_pr = compute_roc_pr_auc(oof_y_true, oof_y_score)
        dec_cnt, dec_tot, dec_cov = compute_decisive_coverage(oof_y_score)
        m05 = compute_binary_metrics_at_threshold(oof_y_true, oof_y_score, thresh=0.5)

        model_summaries[m_name] = {
            "oof_roc_auc": oof_roc,
            "oof_pr_auc": oof_pr,
            "oof_accuracy": m05["accuracy"],
            "oof_precision": m05["precision"],
            "oof_recall": m05["recall"],
            "oof_specificity": m05["specificity"],
            "oof_fpr": m05["fpr"],
            "oof_fnr": m05["fnr"],
            "oof_macro_f1": m05["macro_f1"],
            "oof_decisive_coverage": dec_cov,
            "fold_checkpoints": fold_checkpoints,
        }
        print(f"\n{m_name.upper()} Overall OOF: ROC-AUC = {oof_roc:.4f} | PR-AUC = {oof_pr:.4f} | Acc = {m05['accuracy']:.4f} | Decisive Coverage = {dec_cov*100:.2f}%")

    # ---------------------------------------------------------
    # Save OOF Predictions Manifest
    # ---------------------------------------------------------
    oof_csv_path = RESULTS_DIR / "oof_mil_predictions.csv"
    with open(oof_csv_path, "w", newline="") as f:
        fieldnames = [
            "image_id", "pair_id", "fold", "split", "label",
            "edit_category", "layout_family",
            "score_model_c_mean", "score_model_d_max",
            "score_model_e_attn", "score_model_f_multiscale",
            "attn_weights_model_e", "attn_weights_model_f"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in train_screenshots:
            iid = s["image_id"]
            writer.writerow({
                "image_id": iid,
                "pair_id": s["pair_id"],
                "fold": s["fold"],
                "split": s["split"],
                "label": s["label_str"],
                "edit_category": s["edit_category"],
                "layout_family": s["layout_family"],
                "score_model_c_mean": all_oof_results["model_c_mean_pooling"][iid]["score"],
                "score_model_d_max": all_oof_results["model_d_max_pooling"][iid]["score"],
                "score_model_e_attn": all_oof_results["model_e_attention_mil"][iid]["score"],
                "score_model_f_multiscale": all_oof_results["model_f_multiscale_mil"][iid]["score"],
                "attn_weights_model_e": json.dumps(all_oof_results["model_e_attention_mil"][iid]["attention"]),
                "attn_weights_model_f": json.dumps(all_oof_results["model_f_multiscale_mil"][iid]["attention"]),
            })
    print(f"\nSaved OOF Predictions to {oof_csv_path} ({compute_sha256(oof_csv_path)})")

    # ---------------------------------------------------------
    # Save Frozen Validation Predictions Manifest
    # ---------------------------------------------------------
    val_csv_path = RESULTS_DIR / "val_mil_predictions.csv"
    val_y_true = [s["label_num"] for s in val_screenshots]
    val_summaries = {}

    with open(val_csv_path, "w", newline="") as f:
        fieldnames = [
            "image_id", "pair_id", "split", "label",
            "edit_category", "layout_family",
            "score_model_c_mean", "score_model_d_max",
            "score_model_e_attn", "score_model_f_multiscale"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, s in enumerate(val_screenshots):
            writer.writerow({
                "image_id": s["image_id"],
                "pair_id": s["pair_id"],
                "split": s["split"],
                "label": s["label_str"],
                "edit_category": s["edit_category"],
                "layout_family": s["layout_family"],
                "score_model_c_mean": float(all_val_predictions["model_c_mean_pooling"][idx]),
                "score_model_d_max": float(all_val_predictions["model_d_max_pooling"][idx]),
                "score_model_e_attn": float(all_val_predictions["model_e_attention_mil"][idx]),
                "score_model_f_multiscale": float(all_val_predictions["model_f_multiscale_mil"][idx]),
            })

    for m_name in [m[0] for m in models_to_train]:
        v_scores = all_val_predictions[m_name]
        v_roc, v_pr = compute_roc_pr_auc(val_y_true, v_scores)
        m05 = compute_binary_metrics_at_threshold(val_y_true, v_scores, thresh=0.5)
        dec_cnt, dec_tot, dec_cov = compute_decisive_coverage(v_scores)
        val_summaries[m_name] = {
            "val_roc_auc": v_roc,
            "val_pr_auc": v_pr,
            "val_accuracy": m05["accuracy"],
            "val_precision": m05["precision"],
            "val_recall": m05["recall"],
            "val_specificity": m05["specificity"],
            "val_fpr": m05["fpr"],
            "val_fnr": m05["fnr"],
            "val_macro_f1": m05["macro_f1"],
            "val_decisive_coverage": dec_cov,
        }

    # Save summary JSON
    summary_path = RESULTS_DIR / "phase20_mil_evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model_summaries_oof": model_summaries,
            "model_summaries_val": val_summaries,
        }, f, indent=2)
    print(f"Saved evaluation summary to {summary_path} ({compute_sha256(summary_path)})")


if __name__ == "__main__":
    run_experiment_suite()
