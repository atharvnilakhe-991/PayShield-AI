#!/usr/bin/env python3
"""
Phase 20: Pre-extract Dense Patch Embeddings across 5 Folds
Extracts 512-dimensional feature vectors from ResNet-18 for all 25,376 patches
across both Scale 1 (4x2 = 8 patches) and Scale 2 (6x3 = 18 patches).

For each fold f in {0..4}:
- Uses checkpoint best_crop_resnet18_fold{f}.pt (trained strictly on Folds \ {f}).
- Embeds all training screenshots (868) and frozen validation screenshots (108).
- Saves features to patch_features_fold{f}.pt with full metadata and tensor provenance.
"""

import csv
import hashlib
import json
import time
from pathlib import Path
import torch
from torchvision import models
from PIL import Image

WORKSPACE_ROOT = Path("/Users/heemanshupjain/Downloads/Member 3")
OUTPUT_DIR = WORKSPACE_ROOT / "experiments" / "phase20_dense_mil_forensics"
PATCHES_DIR = OUTPUT_DIR / "patches"
MANIFEST_PATH = OUTPUT_DIR / "data" / "patch_manifest.csv"
DATA_DIR = OUTPUT_DIR / "data"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        patch_path = PATCHES_DIR / f"{r['patch_id']}.png"
        with Image.open(patch_path) as im:
            rgb = im.convert("RGB")
            # Convert to tensor and normalize
            t = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8).reshape(224, 224, 3)
            t = t.permute(2, 0, 1).float() / 255.0
            norm_t = (t - IMAGENET_MEAN) / IMAGENET_STD
        return norm_t, idx


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using extraction device: {device}")

    # Load patch manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    print(f"Loaded {len(manifest_rows)} patch records from manifest.")

    dataset = PatchDataset(manifest_rows)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=False
    )

    # Process each fold
    for fold_idx in range(5):
        ckpt_path = WORKSPACE_ROOT / "experiments" / "phase19_local_branch_redevelopment" / "checkpoints" / f"best_crop_resnet18_fold{fold_idx}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing fold checkpoint: {ckpt_path}")

        ckpt_hash = compute_sha256(ckpt_path)
        print(f"\n--- Extracting Features for Fold {fold_idx} using {ckpt_path.name} ({ckpt_hash[:16]}...) ---")

        # Build feature extractor
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(512, 2)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device).eval()

        all_feats = torch.zeros((len(manifest_rows), 512), dtype=torch.float32)

        t0 = time.perf_counter()
        with torch.no_grad():
            for batch_t, batch_indices in loader:
                batch_t = batch_t.to(device)
                feats = feature_extractor(batch_t).squeeze(-1).squeeze(-1)  # [B, 512]
                all_feats[batch_indices] = feats.cpu()

        if device.type == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        print(f"Fold {fold_idx} extraction completed in {t1 - t0:.2f}s ({len(manifest_rows)/(t1 - t0):.1f} patches/s).")

        out_path = DATA_DIR / f"patch_features_fold{fold_idx}.pt"
        torch.save({
            "fold": fold_idx,
            "feature_extractor_checkpoint": ckpt_path.name,
            "checkpoint_sha256": ckpt_hash,
            "features": all_feats,
            "manifest_sha256": compute_sha256(MANIFEST_PATH),
        }, out_path)
        print(f"Saved: {out_path} ({compute_sha256(out_path)})")


if __name__ == "__main__":
    main()
