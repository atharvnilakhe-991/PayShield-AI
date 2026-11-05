"""
Phase 9 Held-Out Test Set Evaluation Script.
Evaluates the frozen pipeline (CNN, OCR, Semantic, Stacker, Final Verdict)
EXACTLY ONCE on the untouched held-out test partition (24 samples, recharge family).
Computes accuracy, precision, recall, F1, Macro-F1, confusion matrix, and inconclusive rates.
Documents statistical uncertainty due to N=24 sample size.
Strictly prohibits tuning, threshold adjustment, or retraining after evaluation.
"""

import csv
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

DATASET_ROOT = WORKSPACE_ROOT / "Hackathon-VIT" / "CBL04-Dataset"
sys.path.insert(0, str(DATASET_ROOT / "scripts" / "phase7"))

import torch
from torch.utils.data import DataLoader
from app.services.ocr_service import get_ocr_service
from app.services.semantic_feature_service import get_semantic_service
from app.services.stacker_service import get_stacker_service
from app.services.final_verdict_service import get_final_verdict_service
from app.schemas.enums import VerdictEnum
from app.schemas.cnn import CNNModelOutput
from model import build_model
from train_evaluate import ScreenshotDataset, infer, read_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase9_final_test_evaluation")


def compute_test_cnn_predictions() -> Dict[str, Dict[str, Any]]:
    """
    Computes Fold 2 CNN predictions for the 24 test samples (recharge layout family).
    Fold 2 was trained on bank_progress and compact; recharge was its held-out validation set.
    """
    ckpt_path = DATASET_ROOT / "models" / "phase7" / "fold_2_resnet18_multitask.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing Fold 2 checkpoint at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test_csv = DATASET_ROOT / "metadata" / "phase7" / "folds" / "fold_2_validation.csv"
    test_rows = read_csv(test_csv)
    loader = DataLoader(ScreenshotDataset(test_rows), batch_size=16, shuffle=False)
    preds, _ = infer(model, loader)

    pred_map = {}
    for p in preds:
        img_p = p["image_path"].replace("\\", "/").strip()
        pred_map[img_p] = p
    logger.info("Computed %d test CNN predictions with Fold 2 model", len(pred_map))
    return pred_map


def evaluate_final_test():
    logger.info("Starting STRICT ONE-TIME Held-Out Test Set Evaluation...")

    test_manifest_path = DATASET_ROOT / "metadata" / "phase7_branch_validation" / "untouched_test_manifest.csv"
    if not test_manifest_path.exists():
        raise FileNotFoundError(f"Untouched test manifest missing at {test_manifest_path}")

    with open(test_manifest_path, "r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))

    assert len(test_rows) == 24, f"CRITICAL: Expected exactly 24 test samples, found {len(test_rows)}"
    logger.info("Verified exactly 24 test samples in %s", test_manifest_path)

    # Services
    ocr_service = get_ocr_service()
    semantic_service = get_semantic_service()
    stacker_service = get_stacker_service()
    verdict_service = get_final_verdict_service()

    cnn_preds_map = compute_test_cnn_predictions()

    clean_dir = DATASET_ROOT / "data" / "synthetic_gpay_phase4" / "clean"
    edited_dir = DATASET_ROOT / "data" / "synthetic_gpay_phase4" / "edited"

    predictions_log = []
    y_true = []
    stacker_y_pred = []
    stacker_scores = []
    verdict_y_pred = []  # 1 for EDITED, 0 for GENUINE, -1 for INCONCLUSIVE

    branch_latencies = {"ocr_ms": [], "semantic_ms": [], "stacker_ms": [], "total_ms": []}
    branch_failures = {"cnn": 0, "ocr": 0, "semantic": 0}

    for idx, r in enumerate(test_rows):
        sample_id = r["sample_id"]
        true_lbl = 1 if r["true_label"] in ["1", "edited"] else 0
        y_true.append(true_lbl)

        is_clean = true_lbl == 0
        img_name = f"{sample_id}.png"
        img_path = (clean_dir if is_clean else edited_dir) / img_name
        rel_img_path = f"data/synthetic_gpay_phase4/{'clean' if is_clean else 'edited'}/{img_name}"

        t_start = time.perf_counter()

        # 1. CNN
        cnn_p = cnn_preds_map.get(rel_img_path)
        if cnn_p:
            cnn_score = float(cnn_p["edited_probability"])
            cnn_obj = CNNModelOutput(manipulation_probability=cnn_score)
        else:
            cnn_score = 0.50
            cnn_obj = None
            branch_failures["cnn"] += 1

        # 2. OCR
        with open(img_path, "rb") as fp:
            img_bytes = fp.read()

        t_ocr = time.perf_counter()
        try:
            ocr_res = ocr_service.extract(img_bytes)
            ocr_ms = (time.perf_counter() - t_ocr) * 1000.0
            branch_latencies["ocr_ms"].append(ocr_ms)
        except Exception as ocr_err:
            logger.error("OCR failed for %s: %s", sample_id, ocr_err)
            ocr_res = None
            branch_failures["ocr"] += 1

        # 3. Semantic
        if ocr_res is not None:
            t_sem = time.perf_counter()
            try:
                sem_res = semantic_service.extract_features(ocr_res)
                sem_ms = (time.perf_counter() - t_sem) * 1000.0
                branch_latencies["semantic_ms"].append(sem_ms)
            except Exception as sem_err:
                logger.error("Semantic analysis failed for %s: %s", sample_id, sem_err)
                sem_res = None
                branch_failures["semantic"] += 1
        else:
            sem_res = None
            branch_failures["semantic"] += 1

        # 4. Stacker
        t_stk = time.perf_counter()
        stacker_res = stacker_service.predict_branches(
            cnn_result=cnn_obj,
            ocr_result=ocr_res,
            semantic_result=sem_res,
        )
        stk_ms = (time.perf_counter() - t_stk) * 1000.0
        branch_latencies["stacker_ms"].append(stk_ms)

        stk_score = stacker_res.final_manipulation_probability
        stacker_scores.append(stk_score)
        stk_cls = 1 if stk_score >= 0.50 else 0
        stacker_y_pred.append(stk_cls)

        # 5. Final Verdict Engine
        verdict_res = verdict_service.evaluate_verdict(
            stacker_output=stacker_res,
            cnn_result=cnn_obj,
            ocr_result=ocr_res,
            semantic_result=sem_res,
        )

        v_cls = -1
        if verdict_res.verdict == VerdictEnum.EDITED:
            v_cls = 1
        elif verdict_res.verdict == VerdictEnum.GENUINE:
            v_cls = 0
        verdict_y_pred.append(v_cls)

        total_ms = (time.perf_counter() - t_start) * 1000.0
        branch_latencies["total_ms"].append(total_ms)

        predictions_log.append({
            "sample_id": sample_id,
            "pair_id": r["pair_id"],
            "layout_family": r["layout_family"],
            "true_label": true_lbl,
            "cnn_score": round(cnn_score, 4),
            "ocr_confidence": round(ocr_res.overall_confidence if ocr_res else 0.0, 4),
            "semantic_score": round(sem_res.semantic_manipulation_score if sem_res else 0.0, 4),
            "stacker_score": round(stk_score, 4),
            "stacker_pred_class": stk_cls,
            "final_verdict": verdict_res.verdict.value,
            "verdict_confidence": round(verdict_res.confidence, 4),
            "verdict_rule": verdict_res.decision_rule,
            "reason_codes": [rc.value for rc in verdict_res.reason_codes],
        })

    # Metrics computation
    y_true_np = np.array(y_true)
    stk_pred_np = np.array(stacker_y_pred)

    tp = int(np.sum((y_true_np == 1) & (stk_pred_np == 1)))
    fp = int(np.sum((y_true_np == 0) & (stk_pred_np == 1)))
    tn = int(np.sum((y_true_np == 0) & (stk_pred_np == 0)))
    fn = int(np.sum((y_true_np == 1) & (stk_pred_np == 0)))

    n = len(y_true)
    acc = (tp + tn) / n if n > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0
    macro_f1 = (f1 + f1_0) / 2.0

    # Verdict metrics (excluding inconclusive)
    inconclusive_count = sum(1 for v in verdict_y_pred if v == -1)
    inconclusive_rate = inconclusive_count / n

    decisive_mask = [v != -1 for v in verdict_y_pred]
    if sum(decisive_mask) > 0:
        dec_true = np.array([y for i, y in enumerate(y_true) if decisive_mask[i]])
        dec_pred = np.array([v for i, v in enumerate(verdict_y_pred) if decisive_mask[i]])
        dec_acc = float(np.mean(dec_true == dec_pred))
    else:
        dec_acc = 0.0

    # Wilson score 95% confidence interval for accuracy: p +- z * sqrt(p(1-p)/n)
    z = 1.96
    ci_margin = z * math.sqrt((acc * (1.0 - acc)) / n) if n > 0 else 0.0
    acc_ci_low = max(0.0, acc - ci_margin)
    acc_ci_high = min(1.0, acc + ci_margin)

    results_payload = {
        "evaluation_name": "Phase 9 Held-Out Test Evaluation",
        "evaluation_date": "2026-11-03",
        "dataset_split": "test",
        "layout_family": "recharge",
        "sample_count": n,
        "clean_samples_count": int(np.sum(y_true_np == 0)),
        "edited_samples_count": int(np.sum(y_true_np == 1)),
        "evaluation_provenance": "Strictly one-time held-out evaluation. No tuning occurred after this run.",
        "statistical_limitation_statement": (
            f"The test set comprises exactly N={n} samples (12 clean, 12 edited) from a single layout family (recharge). "
            f"Observed accuracy of {acc:.4f} carries an asymptotic 95% confidence interval of [{acc_ci_low:.4f}, {acc_ci_high:.4f}]. "
            "Metrics reflect performance on this specific synthetic template and should not be extrapolated to unconstrained real-world payment screenshots."
        ),
        "stacker_metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "confidence_interval_95": {
                "lower": round(acc_ci_low, 4),
                "upper": round(acc_ci_high, 4),
                "margin": round(ci_margin, 4),
            },
            "edited_class": {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "support": 12,
            },
            "clean_class": {
                "precision": round(prec_0, 4),
                "recall": round(rec_0, 4),
                "f1": round(f1_0, 4),
                "support": 12,
            },
            "confusion_matrix": {
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
            },
        },
        "final_verdict_engine_metrics": {
            "total_decisions": n,
            "inconclusive_count": inconclusive_count,
            "inconclusive_rate": round(inconclusive_rate, 4),
            "decisive_decisions_count": n - inconclusive_count,
            "decisive_accuracy": round(dec_acc, 4),
        },
        "branch_failure_rates": {
            "cnn_failures": branch_failures["cnn"],
            "ocr_failures": branch_failures["ocr"],
            "semantic_failures": branch_failures["semantic"],
        },
        "latency_stats_ms": {
            "mean_total": round(float(np.mean(branch_latencies["total_ms"])), 2),
            "median_total": round(float(np.median(branch_latencies["total_ms"])), 2),
            "mean_ocr": round(float(np.mean(branch_latencies["ocr_ms"])), 2),
            "mean_semantic": round(float(np.mean(branch_latencies["semantic_ms"])), 2),
            "mean_stacker": round(float(np.mean(branch_latencies["stacker_ms"])), 2),
        },
    }

    # Save outputs
    out_dir = DATASET_ROOT / "reports" / "phase9"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_p = out_dir / "final_test_evaluation_results.json"
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    csv_p = out_dir / "final_test_predictions.csv"
    with open(csv_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions_log[0].keys()))
        writer.writeheader()
        writer.writerows(predictions_log)

    logger.info("Phase 9 Test Evaluation successfully written to %s and %s", json_p, csv_p)
    return results_payload


if __name__ == "__main__":
    evaluate_final_test()

