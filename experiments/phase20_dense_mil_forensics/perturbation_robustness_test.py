#!/usr/bin/env python3
"""
Phase 20: Harmless Perturbation Robustness Diagnostic
Evaluates stability of Dense MIL model predictions against non-tampering transformations:
1. Recompression: JPEG Q=95, Q=85
2. Rescaling: 95% (-5%), 105% (+5%)

Measures mean absolute score deviation |Delta S| and prediction stability on validation screenshots.
"""

import csv
import io
import json
import os
from pathlib import Path
import numpy as np
import torch
from torchvision import models
from PIL import Image

WORKSPACE_ROOT = Path("/Users/heemanshupjain/Downloads/Member 3")
PHASE20_DIR = WORKSPACE_ROOT / "experiments" / "phase20_dense_mil_forensics"
RESULTS_DIR = PHASE20_DIR / "results"
DATASET_ROOT = WORKSPACE_ROOT / "Hackathon-VIT" / "CBL04-Dataset"
VAL_MANIFEST = DATASET_ROOT / "metadata" / "phase6" / "validation_manifest.csv"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def letterbox_and_resize(patch_img, target_size=224, bg_color=(18, 18, 18)):
    w, h = patch_img.size
    max_dim = max(w, h)
    canvas = Image.new("RGB", (max_dim, max_dim), bg_color)
    offset_x = (max_dim - w) // 2
    offset_y = (max_dim - h) // 2
    canvas.paste(patch_img, (offset_x, offset_y))
    return canvas.resize((target_size, target_size), Image.Resampling.LANCZOS)


def extract_4x2_patches_from_pil(img_rgb):
    w, h = img_rgb.size
    patches = []
    for r in range(4):
        y1 = int(r * h / 4)
        y2 = int((r + 1) * h / 4) if r < 3 else h
        for c in range(2):
            x1 = int(c * w / 2)
            x2 = int((c + 1) * w / 2) if c < 1 else w
            patch_crop = img_rgb.crop([x1, y1, x2, y2])
            proc = letterbox_and_resize(patch_crop)
            t = torch.frombuffer(bytearray(proc.tobytes()), dtype=torch.uint8).reshape(224, 224, 3)
            t = t.permute(2, 0, 1).float() / 255.0
            norm_t = (t - IMAGENET_MEAN) / IMAGENET_STD
            patches.append(norm_t)
    return torch.stack(patches, dim=0)  # [8, 3, 224, 224]


def run_perturbation_test():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Perturbation Test on: {device}")

    # Load Model E checkpoint from Fold 0
    ckpt_path = PHASE20_DIR / "checkpoints" / "best_model_e_attention_mil_fold0.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not ready yet: {ckpt_path}. Exiting.")
        return

    # Load ResNet-18 feature extractor from Phase 19 Fold 0
    fe_ckpt = WORKSPACE_ROOT / "experiments" / "phase19_local_branch_redevelopment" / "checkpoints" / "best_crop_resnet18_fold0.pt"
    fe_model = models.resnet18(weights=None)
    fe_model.fc = torch.nn.Linear(512, 2)
    fe_data = torch.load(fe_ckpt, map_location="cpu", weights_only=False)
    fe_model.load_state_dict(fe_data["model_state_dict"])
    extractor = torch.nn.Sequential(*list(fe_model.children())[:-1]).to(device).eval()

    # Load Model E
    from train_and_evaluate_mil import GatedAttentionMIL
    mil_model = GatedAttentionMIL().to(device)
    mil_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mil_model.load_state_dict({k: v.to(device) for k, v in mil_data["state_dict"].items()})
    mil_model.eval()

    def score_image(pil_img):
        patch_t = extract_4x2_patches_from_pil(pil_img).to(device)  # [8, 3, 224, 224]
        with torch.no_grad():
            feats = extractor(patch_t).squeeze(-1).squeeze(-1).unsqueeze(0)  # [1, 8, 512]
            logits, attn = mil_model(feats)
            prob = torch.sigmoid(logits).item()
        return prob

    # Load sample of validation screenshots (10 clean, 10 edited)
    with open(VAL_MANIFEST, "r", encoding="utf-8-sig") as f:
        val_items = list(csv.DictReader(f))

    clean_samples = [r for r in val_items if r["label"] == "clean"][:10]
    edited_samples = [r for r in val_items if r["label"] == "edited"][:10]
    test_samples = clean_samples + edited_samples

    perturbations = ["baseline", "jpeg_q95", "jpeg_q85", "scale_95", "scale_105"]
    results = []

    for item in test_samples:
        img_path = DATASET_ROOT / item["image_path"]
        with Image.open(img_path) as im:
            orig_img = im.convert("RGB")
            base_score = score_image(orig_img)

            # JPEG Q95
            buf_95 = io.BytesIO()
            orig_img.save(buf_95, format="JPEG", quality=95)
            buf_95.seek(0)
            img_q95 = Image.open(buf_95)
            s_q95 = score_image(img_q95)

            # JPEG Q85
            buf_85 = io.BytesIO()
            orig_img.save(buf_85, format="JPEG", quality=85)
            buf_85.seek(0)
            img_q85 = Image.open(buf_85)
            s_q85 = score_image(img_q85)

            # Scale 95%
            w, h = orig_img.size
            img_s95 = orig_img.resize((int(w * 0.95), int(h * 0.95)), Image.Resampling.BILINEAR)
            s_s95 = score_image(img_s95)

            # Scale 105%
            img_s105 = orig_img.resize((int(w * 1.05), int(h * 1.05)), Image.Resampling.BILINEAR)
            s_s105 = score_image(img_s105)

            results.append({
                "image_id": item["image_id"],
                "label": item["label"],
                "edit_category": item.get("edit_category", "none"),
                "base_score": base_score,
                "score_jpeg_q95": s_q95,
                "delta_q95": s_q95 - base_score,
                "score_jpeg_q85": s_q85,
                "delta_q85": s_q85 - base_score,
                "score_scale_95": s_s95,
                "delta_s95": s_s95 - base_score,
                "score_scale_105": s_s105,
                "delta_s105": s_s105 - base_score,
            })

    # Summary statistics
    delta_q95 = [abs(r["delta_q95"]) for r in results]
    delta_q85 = [abs(r["delta_q85"]) for r in results]
    delta_s95 = [abs(r["delta_s95"]) for r in results]
    delta_s105 = [abs(r["delta_s105"]) for r in results]

    summary = {
        "num_samples_evaluated": len(results),
        "mean_abs_delta_jpeg_q95": float(np.mean(delta_q95)),
        "max_abs_delta_jpeg_q95": float(np.max(delta_q95)),
        "mean_abs_delta_jpeg_q85": float(np.mean(delta_q85)),
        "max_abs_delta_jpeg_q85": float(np.max(delta_q85)),
        "mean_abs_delta_scale_95": float(np.mean(delta_s95)),
        "max_abs_delta_scale_95": float(np.max(delta_s95)),
        "mean_abs_delta_scale_105": float(np.mean(delta_s105)),
        "max_abs_delta_scale_105": float(np.max(delta_s105)),
        "sample_level_results": results
    }

    out_file = RESULTS_DIR / "perturbation_robustness_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Perturbation diagnostic completed and saved to {out_file}")


if __name__ == "__main__":
    run_perturbation_test()

