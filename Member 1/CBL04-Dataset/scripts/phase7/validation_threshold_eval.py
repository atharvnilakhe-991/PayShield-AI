from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_evaluate import ScreenshotDataset, classification_metrics, read_csv, infer  # noqa: E402
from model import build_model  # noqa: E402


def choose_threshold(labels: list[int], scores: list[float]) -> tuple[float, dict]:
    candidates = sorted({0.0, 1.0, *scores})
    scored = []
    for threshold in candidates:
        metrics = classification_metrics(labels, [int(score >= threshold) for score in scores])
        scored.append((metrics["f1"], metrics["accuracy"], -abs(threshold - 0.5), threshold, metrics))
    best = max(scored, key=lambda item: item[:4])
    return best[3], best[4]


def main() -> None:
    torch.set_num_threads(4)
    out_dir = ROOT / "reports/phase8"
    out_dir.mkdir(parents=True, exist_ok=True)
    results, predictions = [], []
    for fold in (1, 2):
        ckpt_path = ROOT / f"models/phase7/fold_{fold}_resnet18_multitask.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model(ckpt["model_config"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        fold_dir = ROOT / "metadata/phase7/folds"
        fold_rows = {}
        all_rows = {}
        for role in ("validation", "test"):
            rows = read_csv(fold_dir / f"fold_{fold}_{role}.csv")
            loader = DataLoader(ScreenshotDataset(rows), batch_size=8, shuffle=False, num_workers=0)
            pred_rows, _ = infer(model, loader)
            # infer() uses 0.5 labels; probabilities stay unchanged so we can
            # select a threshold only on validation predictions below.
            fold_rows[role] = pred_rows
            all_rows[role] = rows
        val_y = [p["true_label"] for p in fold_rows["validation"]]
        val_scores = [p["edited_probability"] for p in fold_rows["validation"]]
        threshold, val_metrics = choose_threshold(val_y, val_scores)
        test_preds = fold_rows["test"]
        test_y = [p["true_label"] for p in test_preds]
        test_scores = [p["edited_probability"] for p in test_preds]
        test_base = classification_metrics(test_y, [int(p >= 0.5) for p in test_scores])
        test_tuned = classification_metrics(test_y, [int(p >= threshold) for p in test_scores])
        results.append({"fold": fold, "test_family": test_preds[0]["layout_family"],
                        "validation_family": fold_rows["validation"][0]["layout_family"],
                        "validation_support": len(val_y), "threshold_selected_on_validation": threshold,
                        "validation_metrics_at_selected_threshold": val_metrics,
                        "test_support": len(test_y), "baseline_threshold_0_5": test_base,
                        "validation_selected_threshold_test": test_tuned,
                        "no_test_tuning": True})
        for pred in test_preds:
            predictions.append({"fold": fold, "source_id": pred["source_id"], "pair_id": pred["pair_id"],
                                "layout_family": pred["layout_family"], "edit_category": pred["edit_category"],
                                "image_path": pred["image_path"], "true_label": pred["true_label"],
                                "edited_probability_raw_uncalibrated": pred["edited_probability"],
                                "baseline_prediction_0_5": int(pred["edited_probability"] >= .5),
                                "validation_selected_threshold": threshold,
                                "validation_selected_prediction": int(pred["edited_probability"] >= threshold)})

    pooled_y = [p["true_label"] for p in predictions]
    pooled_base = classification_metrics(pooled_y, [p["baseline_prediction_0_5"] for p in predictions])
    pooled_tuned = classification_metrics(pooled_y, [p["validation_selected_prediction"] for p in predictions])
    payload = {"experiment": "validation-only per-fold probability threshold selection",
               "scope": "completed folds 1 and 2 only; synthetic held-out tests; no training or test tuning",
               "folds": results, "pooled_two_model_baseline_at_0_5": pooled_base,
               "pooled_two_model_validation_selected_thresholds": pooled_tuned,
               "interpretation": "The pooled values combine distinct fold checkpoints and are not one model score. Thresholds are not calibrated probabilities."}
    (out_dir / "threshold_tuning_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if predictions:
        with (out_dir / "threshold_tuning_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
            writer.writeheader(); writer.writerows(predictions)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
