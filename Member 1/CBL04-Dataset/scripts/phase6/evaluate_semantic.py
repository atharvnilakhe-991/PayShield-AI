"""
Phase 6 Semantic Consistency Evaluation Script.
Evaluates SemanticFeatureService on paired clean/edited GPay dark-theme receipts
from training and validation splits (strictly excluding test set).
Measures per-feature coverage, format validity correctness, false-positive rule activation
on clean receipts, anomaly detection on edited receipts, and execution latency.
"""

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Set workspace paths
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from app.services.ocr_service import get_ocr_service
from app.services.semantic_feature_service import get_semantic_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_semantic")


def evaluate_semantic_benchmark(sample_size: int = 50) -> Dict[str, Any]:
    """
    Evaluates semantic feature extraction across paired clean/edited receipts
    strictly drawn from train/val splits (excluding held-out test set).
    """
    dataset_dir = WORKSPACE_ROOT / "Hackathon-VIT" / "CBL04-Dataset"
    ann_dir = dataset_dir / "annotations" / "phase4" / "field_annotations"
    clean_dir = dataset_dir / "data" / "synthetic_gpay_phase4" / "clean"
    edited_dir = dataset_dir / "data" / "synthetic_gpay_phase4" / "edited"
    test_manifest_p = dataset_dir / "metadata" / "phase6" / "test_manifest.csv"

    # 1. Load test manifest to strictly exclude test samples
    test_sids = set()
    if test_manifest_p.exists():
        with open(test_manifest_p) as fp:
            r = csv.DictReader(fp)
            for row in r:
                sid = row.get("source_id") or row.get("sample_id")
                if sid:
                    test_sids.add(sid)
    logger.info("Loaded %d test sample IDs for strict test-set exclusion", len(test_sids))

    # 2. Collect available non-test annotation files
    ann_files = sorted(ann_dir.glob("P4-*.json"))
    categories = ["amount", "transaction_id", "recipient", "date_time", "other"]
    cat_bins: Dict[str, List[Path]] = {c: [] for c in categories}

    for f in ann_files:
        try:
            with open(f) as fp:
                meta = json.load(fp)
            sid = meta["source_id"]
            if sid in test_sids:
                continue  # STRICT EXCLUSION OF TEST SAMPLES

            cat = meta.get("edited_field", "other")
            if cat in cat_bins and len(cat_bins[cat]) < (sample_size // len(categories)):
                clean_p = clean_dir / f"{sid}_clean.png"
                edited_p = edited_dir / f"{sid}_edited.png"
                if clean_p.exists() and edited_p.exists():
                    cat_bins[cat].append(f)
        except Exception:
            continue

    selected_pairs: List[Path] = []
    for cat, flist in cat_bins.items():
        selected_pairs.extend(flist)

    logger.info("Selected %d paired samples (%d total images) across %d categories for semantic evaluation",
                len(selected_pairs), len(selected_pairs) * 2, len(categories))

    ocr_service = get_ocr_service()
    semantic_service = get_semantic_service()

    results = {
        "total_images_evaluated": 0,
        "clean_images_evaluated": 0,
        "edited_images_evaluated": 0,
        "test_samples_excluded_count": len(test_sids),
        "latencies_ms": [],
        "clean_stats": {
            "total_clean": 0,
            "false_positive_anomalies": 0,
            "fp_by_field": {
                "amount_anomaly": 0,
                "recipient_anomaly": 0,
                "date_anomaly": 0,
                "ref_id_anomaly": 0,
                "status_anomaly": 0,
                "upi_handle_anomaly": 0,
            },
            "mean_manipulation_score": 0.0,
        },
        "edited_stats": {
            "total_edited": 0,
            "anomalies_detected": 0,
            "detected_by_target_field": {
                "amount": 0,
                "transaction_id": 0,
                "recipient": 0,
                "date_time": 0,
                "other": 0,
            },
            "mean_manipulation_score": 0.0,
        },
        "feature_metrics": {
            "amount_valid_format": {"available": 0, "not_available": 0, "invalid_format": 0, "ambiguous": 0},
            "amount_currency_symbol_valid": {"available": 0, "not_available": 0, "invalid_format": 0, "ambiguous": 0},
            "amount_numeric_value_reasonable": {"available": 0, "not_available": 0, "invalid_format": 0, "ambiguous": 0},
            "upi_handle_valid_format": {"available": 0, "not_available": 0, "invalid_format": 0, "ambiguous": 0},
            "date_chronology_valid": {"available": 0, "not_available": 0, "invalid_format": 0, "ambiguous": 0},
            "time_format_valid": {"available": 0, "not_available": 0, "invalid_format": 0, "ambiguous": 0},
            "ref_id_length_valid": {"available": 0, "not_available": 0, "invalid_format": 0, "ambiguous": 0},
            "status_text_authentic": {"available": 0, "not_available": 0, "invalid_format": 0, "ambiguous": 0},
        },
        "latency_stats_ms": {},
    }

    eval_items = []
    for ann_file in selected_pairs:
        with open(ann_file) as fp:
            meta = json.load(fp)
        sid = meta["source_id"]
        eval_items.append((sid, "clean", clean_dir / f"{sid}_clean.png", meta))
        eval_items.append((sid, "edited", edited_dir / f"{sid}_edited.png", meta))

    total_items = len(eval_items)
    clean_scores = []
    edited_scores = []

    for idx, (sid, variant, img_path, meta) in enumerate(eval_items):
        with open(img_path, "rb") as fp:
            img_bytes = fp.read()

        ocr_res = ocr_service.extract(img_bytes)

        t0 = time.perf_counter()
        sem_res = semantic_service.extract_features(ocr_res)
        dur_ms = (time.perf_counter() - t0) * 1000.0

        results["latencies_ms"].append(dur_ms)
        results["total_images_evaluated"] += 1

        # Track per-feature status
        for feat_name, ev in sem_res.feature_evidence.items():
            if feat_name in results["feature_metrics"]:
                st = ev.feature_status
                if st in results["feature_metrics"][feat_name]:
                    results["feature_metrics"][feat_name][st] += 1

        target_field = meta.get("edited_field", "other")
        anom = sem_res.field_anomalies
        has_any_anomaly = (
            anom.amount_anomaly or anom.recipient_anomaly or anom.date_anomaly
            or anom.ref_id_anomaly or anom.status_anomaly or anom.upi_handle_anomaly
        )

        if variant == "clean":
            results["clean_images_evaluated"] += 1
            results["clean_stats"]["total_clean"] += 1
            clean_scores.append(sem_res.semantic_manipulation_score)
            if has_any_anomaly:
                results["clean_stats"]["false_positive_anomalies"] += 1
                if anom.amount_anomaly:
                    results["clean_stats"]["fp_by_field"]["amount_anomaly"] += 1
                if anom.recipient_anomaly:
                    results["clean_stats"]["fp_by_field"]["recipient_anomaly"] += 1
                if anom.date_anomaly:
                    results["clean_stats"]["fp_by_field"]["date_anomaly"] += 1
                if anom.ref_id_anomaly:
                    results["clean_stats"]["fp_by_field"]["ref_id_anomaly"] += 1
                if anom.status_anomaly:
                    results["clean_stats"]["fp_by_field"]["status_anomaly"] += 1
                if anom.upi_handle_anomaly:
                    results["clean_stats"]["fp_by_field"]["upi_handle_anomaly"] += 1
        else:
            results["edited_images_evaluated"] += 1
            results["edited_stats"]["total_edited"] += 1
            edited_scores.append(sem_res.semantic_manipulation_score)
            if has_any_anomaly:
                results["edited_stats"]["anomalies_detected"] += 1
                if target_field in results["edited_stats"]["detected_by_target_field"]:
                    results["edited_stats"]["detected_by_target_field"][target_field] += 1

        if (idx + 1) % 10 == 0 or (idx + 1) == total_items:
            logger.info("Processed %d / %d images (%.1f%%)...", idx + 1, total_items, (idx + 1) / total_items * 100)

    # Aggregate statistics
    results["clean_stats"]["mean_manipulation_score"] = round(float(np.mean(clean_scores)), 4) if clean_scores else 0.0
    results["edited_stats"]["mean_manipulation_score"] = round(float(np.mean(edited_scores)), 4) if edited_scores else 0.0

    lats = results["latencies_ms"]
    results["latency_stats_ms"] = {
        "mean": round(float(np.mean(lats)), 2),
        "median": round(float(np.median(lats)), 2),
        "p90": round(float(np.percentile(lats, 90)), 2),
        "p95": round(float(np.percentile(lats, 95)), 2),
        "min": round(float(np.min(lats)), 2),
        "max": round(float(np.max(lats)), 2),
    }

    # Save output report
    out_dir = WORKSPACE_ROOT / "Hackathon-VIT" / "CBL04-Dataset" / "reports" / "phase6"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "semantic_evaluation_results.json"
    with open(out_json, "w") as fp:
        json.dump(results, fp, indent=2)

    logger.info("Phase 6 evaluation complete! Report saved to %s", out_json)
    return results


if __name__ == "__main__":
    res = evaluate_semantic_benchmark(sample_size=50)
    print("\n" + "=" * 60)
    print("PHASE 6 SEMANTIC CONSISTENCY EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Evaluated: {res['total_images_evaluated']} ({res['clean_images_evaluated']} Clean, {res['edited_images_evaluated']} Edited)")
    print(f"Strict Test Exclusion: {res['test_samples_excluded_count']} test samples excluded")
    print(f"Mean Latency: {res['latency_stats_ms']['mean']} ms | Median: {res['latency_stats_ms']['median']} ms | P95: {res['latency_stats_ms']['p95']} ms")
    print(f"\nClean Receipts (False Positive Rate):")
    tot_c = res['clean_stats']['total_clean']
    fp_c = res['clean_stats']['false_positive_anomalies']
    print(f"  - FP Anomaly Count: {fp_c} / {tot_c} ({fp_c / tot_c * 100:.1f}%)")
    print(f"  - Mean Clean Manipulation Score: {res['clean_stats']['mean_manipulation_score']}")
    print(f"\nEdited Receipts (Semantic Rule Activation):")
    tot_e = res['edited_stats']['total_edited']
    det_e = res['edited_stats']['anomalies_detected']
    print(f"  - Anomaly Activation: {det_e} / {tot_e} ({det_e / tot_e * 100:.1f}%)")
    print(f"  - Mean Edited Manipulation Score: {res['edited_stats']['mean_manipulation_score']}")
    print("\nFeature Status Breakdown across all evaluated receipts:")
    for f, st in res["feature_metrics"].items():
        print(f"  - {f:32s}: Avail={st['available']}, Missing={st['not_available']}, Invalid={st['invalid_format']}")
    print("=" * 60)

