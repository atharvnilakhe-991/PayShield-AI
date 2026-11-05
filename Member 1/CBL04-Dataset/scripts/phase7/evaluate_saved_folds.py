from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import build_model  # noqa: E402
from train_evaluate import (ScreenshotDataset, classification_metrics, infer,
                            localization_from_counts, draw_confusion, draw_bar,
                            read_csv, write_csv)  # noqa: E402


def main() -> None:
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    reports = ROOT / "reports/phase7"
    predictions = []
    fold_results = []
    localization_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for fold in (1, 2):
        checkpoint_path = ROOT / f"models/phase7/fold_{fold}_resnet18_multitask.pt"
        if not checkpoint_path.is_file():
            continue
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = build_model(checkpoint["model_config"]).cpu()
        model.load_state_dict(checkpoint["state_dict"])
        rows = read_csv(ROOT / f"metadata/phase7/folds/fold_{fold}_test.csv")
        loader = DataLoader(ScreenshotDataset(rows), batch_size=8, shuffle=False, num_workers=0)
        pred, pixels = infer(model, loader)
        for p in pred:
            p["fold"] = fold
            predictions.append(p)
        metrics = classification_metrics([p["true_label"] for p in pred], [p["predicted_label"] for p in pred])
        fold_loc = localization_from_counts(pixels)
        family = rows[0]["layout_family"]
        localization_counts[family] = {"tp": pixels["tp"], "fp": pixels["fp"], "fn": pixels["fn"]}
        fold_results.append({"fold": fold, "test_family": family, "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
                             "training_epochs": len(checkpoint["training_history"]),
                             "best_validation_f1": checkpoint["best_validation_f1"],
                             "metrics": metrics, "localization": fold_loc})
    if not predictions:
        raise RuntimeError("No completed checkpoints available for evaluation")
    overall = classification_metrics([p["true_label"] for p in predictions], [p["predicted_label"] for p in predictions])
    family_predictions = defaultdict(list)
    for p in predictions:
        family_predictions[p["layout_family"]].append(p)
    per_family = {f: classification_metrics([p["true_label"] for p in rows], [p["predicted_label"] for p in rows])
                  for f, rows in family_predictions.items()}
    per_family_loc = {f: localization_from_counts(c) for f, c in localization_counts.items()}
    categories = {}
    for category in sorted({p["edit_category"] for p in predictions if p["true_label"] == 1}):
        positives = [p for p in predictions if p["true_label"] == 1 and p["edit_category"] == category]
        fams = {p["layout_family"] for p in positives}
        negatives = [p for p in predictions if p["true_label"] == 0 and p["layout_family"] in fams]
        selected = negatives + positives
        categories[category] = classification_metrics([p["true_label"] for p in selected], [p["predicted_label"] for p in selected])
        categories[category]["edited_support"] = len(positives)
        categories[category]["clean_negative_support"] = len(negatives)
    false_positives = [p for p in predictions if p["true_label"] == 0 and p["predicted_label"] == 1]
    false_negatives = [p for p in predictions if p["true_label"] == 1 and p["predicted_label"] == 0]
    for p in false_positives: p["error_type"] = "false_positive"
    for p in false_negatives: p["error_type"] = "false_negative"
    write_csv(reports / "available_folds_1_2_predictions.csv", predictions)
    write_csv(reports / "available_false_positives.csv", false_positives)
    write_csv(reports / "available_false_negatives.csv", false_negatives)
    result = {"scope": "completed fold 1 and fold 2 test families only; folds 3 and 4 have no checkpoints",
              "threshold": 0.5, "evaluated_samples": len(predictions), "evaluated_families": sorted(per_family),
              "not_evaluated_families": ["bank_progress", "recharge"], "overall": overall,
              "per_family": per_family, "per_family_localization": per_family_loc,
              "per_edit_category": categories, "per_fold": fold_results,
              "false_positive_count": len(false_positives), "false_negative_count": len(false_negatives),
              "synthetic_only": True, "real_images_used": False, "pretrained": False}
    (reports / "available_folds_1_2_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plots = reports / "plots"
    draw_confusion(plots / "available_folds_1_2_confusion_matrix.png", overall["confusion_matrix"])
    draw_bar(plots / "available_folds_1_2_per_family_f1.png", "F1 by evaluated held-out family",
             [(k,v["f1"]) for k,v in per_family.items()], (51,125,190))
    draw_bar(plots / "available_folds_1_2_per_edit_f1.png", "Edit-category F1 (available families)",
             [(k,v["f1"]) for k,v in categories.items()], (54,150,90))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
