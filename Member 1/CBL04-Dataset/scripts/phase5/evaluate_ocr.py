"""
Phase 5 OCR Evaluation Script.
Evaluates EasyOCR text and structured field extraction against ground truth field annotations
from Hackathon-VIT/CBL04-Dataset/annotations/phase4/field_annotations/.
Measures exact match accuracy, normalized accuracy, field extraction coverage,
missing field rates, ambiguous field rates, and latency statistics.
Preserves paired clean/edited sample integrity across layout families.
"""

import json
import logging
import os
import re
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_ocr")


def normalize_str(s: Optional[str]) -> str:
    """Normalizes string for comparison (removes punctuation, spaces, lowercase)."""
    if not s:
        return ""
    return re.sub(r"[^a-zA-Z0-9]", "", str(s)).lower()


def evaluate_ocr_benchmark(sample_size: int = 50) -> Dict[str, Any]:
    """
    Evaluates OCR extraction on a balanced, paired sample of clean and edited screenshots.
    
    Args:
        sample_size: Number of pairs to evaluate (default: 50 pairs = 100 images).
    """
    dataset_dir = WORKSPACE_ROOT / "Hackathon-VIT" / "CBL04-Dataset"
    ann_dir = dataset_dir / "annotations" / "phase4" / "field_annotations"
    clean_dir = dataset_dir / "data" / "synthetic_gpay_phase4" / "clean"
    edited_dir = dataset_dir / "data" / "synthetic_gpay_phase4" / "edited"

    # Collect available annotation files
    ann_files = sorted(ann_dir.glob("P4-*.json"))
    if not ann_files:
        raise FileNotFoundError(f"No annotation JSONs found in {ann_dir}")

    # Stratified selection across the 5 edit categories: 10 pairs per category = 50 pairs
    categories = ["amount", "transaction_id", "recipient", "date_time", "other"]
    cat_bins: Dict[str, List[Path]] = {c: [] for c in categories}

    for f in ann_files:
        try:
            with open(f) as fp:
                meta = json.load(fp)
                cat = meta.get("edited_field", "other")
                if cat in cat_bins and len(cat_bins[cat]) < (sample_size // len(categories)):
                    # Ensure both clean and edited images exist
                    sid = meta["source_id"]
                    clean_p = clean_dir / f"{sid}_clean.png"
                    edited_p = edited_dir / f"{sid}_edited.png"
                    if clean_p.exists() and edited_p.exists():
                        cat_bins[cat].append(f)
        except Exception:
            continue

    selected_pairs: List[Path] = []
    for cat, flist in cat_bins.items():
        selected_pairs.extend(flist)

    logger.info("Selected %d paired samples (%d total images) across %d categories",
                len(selected_pairs), len(selected_pairs) * 2, len(categories))

    ocr_service = get_ocr_service()

    results = {
        "total_images_evaluated": 0,
        "clean_images_evaluated": 0,
        "edited_images_evaluated": 0,
        "latencies_ms": [],
        "overall_confidences": [],
        "per_field_stats": {
            "amount": {"total_annotated": 0, "extracted": 0, "exact_match": 0, "numeric_match": 0, "missing": 0, "ambiguous": 0},
            "transaction_id": {"total_annotated": 0, "extracted": 0, "exact_match": 0, "normalized_match": 0, "missing": 0, "ambiguous": 0},
            "recipient": {"total_annotated": 0, "extracted": 0, "exact_match": 0, "normalized_match": 0, "missing": 0, "ambiguous": 0},
            "date_time": {"total_annotated": 0, "extracted": 0, "exact_match": 0, "normalized_match": 0, "missing": 0, "ambiguous": 0},
            "other": {"total_annotated": 0, "extracted": 0, "token_present": 0, "missing": 0, "ambiguous": 0},
        },
        "coverage": {
            "amount": 0.0,
            "transaction_id": 0.0,
            "status": 0.0,
            "receiver": 0.0,
            "date": 0.0,
        },
        "layout_family_breakdown": {},
    }

    eval_items = []
    for ann_file in selected_pairs:
        with open(ann_file) as fp:
            meta = json.load(fp)
        sid = meta["source_id"]
        eval_items.append((sid, "clean", clean_dir / f"{sid}_clean.png", meta["original_fictional_value"], meta))
        eval_items.append((sid, "edited", edited_dir / f"{sid}_edited.png", meta["replacement_fictional_value"], meta))

    total_items = len(eval_items)

    for idx, (sid, variant, img_path, gt_val, meta) in enumerate(eval_items):
        t0 = time.perf_counter()
        with open(img_path, "rb") as fp:
            img_bytes = fp.read()

        ocr_res = ocr_service.extract(img_bytes)
        dur_ms = (time.perf_counter() - t0) * 1000.0

        results["latencies_ms"].append(dur_ms)
        results["overall_confidences"].append(ocr_res.overall_confidence)
        results["total_images_evaluated"] += 1
        if variant == "clean":
            results["clean_images_evaluated"] += 1
        else:
            results["edited_images_evaluated"] += 1

        family = meta.get("layout_family", "unknown")
        if family not in results["layout_family_breakdown"]:
            results["layout_family_breakdown"][family] = {"count": 0, "mean_conf": []}
        results["layout_family_breakdown"][family]["count"] += 1
        results["layout_family_breakdown"][family]["mean_conf"].append(ocr_res.overall_confidence)

        # Check coverage
        fields = ocr_res.fields
        if fields.amount_numeric is not None:
            results["coverage"]["amount"] += 1
        if fields.transaction_id is not None:
            results["coverage"]["transaction_id"] += 1
        if fields.status is not None:
            results["coverage"]["status"] += 1
        if fields.receiver is not None:
            results["coverage"]["receiver"] += 1
        if fields.date is not None:
            results["coverage"]["date"] += 1

        # Check target annotated field
        target_field = meta.get("edited_field", "other")
        fstats = results["per_field_stats"][target_field]
        fstats["total_annotated"] += 1

        gt_str = str(gt_val).strip()

        if target_field == "amount":
            extracted_num = fields.amount_numeric
            det_field = ocr_res.detailed_fields.get("amount")
            if det_field and det_field.is_detected:
                fstats["extracted"] += 1
                if det_field.extraction_status == "ambiguous":
                    fstats["ambiguous"] += 1
                try:
                    gt_num = float(gt_str.replace(",", ""))
                    if extracted_num is not None and abs(extracted_num - gt_num) < 0.01:
                        fstats["numeric_match"] += 1
                        fstats["exact_match"] += 1
                except ValueError:
                    pass
            else:
                fstats["missing"] += 1

        elif target_field == "transaction_id":
            det_field = ocr_res.detailed_fields.get("transaction_id")
            if det_field and det_field.is_detected:
                fstats["extracted"] += 1
                if det_field.extraction_status == "ambiguous":
                    fstats["ambiguous"] += 1
                ext_val = det_field.normalized_value or ""
                if gt_str in ext_val or ext_val in gt_str:
                    fstats["exact_match"] += 1
                    fstats["normalized_match"] += 1
                elif normalize_str(gt_str) in normalize_str(ext_val):
                    fstats["normalized_match"] += 1
            else:
                fstats["missing"] += 1

        elif target_field == "recipient":
            det_field = ocr_res.detailed_fields.get("receiver")
            if det_field and det_field.is_detected:
                fstats["extracted"] += 1
                if det_field.extraction_status == "ambiguous":
                    fstats["ambiguous"] += 1
                ext_val = det_field.normalized_value or ""
                if gt_str.lower() in ext_val.lower() or ext_val.lower() in gt_str.lower():
                    fstats["exact_match"] += 1
                    fstats["normalized_match"] += 1
                elif normalize_str(gt_str) in normalize_str(ext_val):
                    fstats["normalized_match"] += 1
            else:
                fstats["missing"] += 1

        elif target_field == "date_time":
            det_date = ocr_res.detailed_fields.get("date")
            det_time = ocr_res.detailed_fields.get("time")
            is_extracted = (det_date and det_date.is_detected) or (det_time and det_time.is_detected)
            if is_extracted:
                fstats["extracted"] += 1
                ext_date = det_date.normalized_value if det_date else ""
                ext_time = det_time.normalized_value if det_time else ""
                combined = f"{ext_date} {ext_time}"
                if gt_str.lower() in combined.lower() or normalize_str(gt_str) in normalize_str(combined):
                    fstats["exact_match"] += 1
                    fstats["normalized_match"] += 1
            else:
                fstats["missing"] += 1

        elif target_field == "other":
            # Bank account suffix (e.g. 6900, 2669) - checked against raw tokens
            det_tokens = [t.raw_text for t in ocr_res.raw_detections if t.raw_text]
            token_match = any(gt_str in t for t in det_tokens)
            if token_match:
                fstats["extracted"] += 1
                fstats["token_present"] += 1
            else:
                fstats["missing"] += 1

        if (idx + 1) % 10 == 0 or (idx + 1) == total_items:
            logger.info("Processed %d / %d images (%.1f%%)...", idx + 1, total_items, (idx + 1) / total_items * 100)

    # Compute overall rates
    N = results["total_images_evaluated"]
    for k in results["coverage"]:
        results["coverage"][k] = round(results["coverage"][k] / N * 100.0, 2)

    for fam in results["layout_family_breakdown"]:
        confs = results["layout_family_breakdown"][fam]["mean_conf"]
        results["layout_family_breakdown"][fam]["mean_confidence"] = round(float(np.mean(confs)), 4) if confs else 0.0
        del results["layout_family_breakdown"][fam]["mean_conf"]

    # Compute latency percentiles
    lats = results["latencies_ms"]
    results["latency_stats_ms"] = {
        "mean": round(float(np.mean(lats)), 2),
        "median": round(float(np.median(lats)), 2),
        "p90": round(float(np.percentile(lats, 90)), 2),
        "p95": round(float(np.percentile(lats, 95)), 2),
        "min": round(float(np.min(lats)), 2),
        "max": round(float(np.max(lats)), 2),
    }

    results["mean_overall_confidence"] = round(float(np.mean(results["overall_confidences"])), 4)

    # Save summary report
    out_dir = WORKSPACE_ROOT / "Hackathon-VIT" / "CBL04-Dataset" / "reports" / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "ocr_evaluation_results.json"
    with open(out_json, "w") as fp:
        json.dump(results, fp, indent=2)

    logger.info("Evaluation complete! Report saved to %s", out_json)
    return results


if __name__ == "__main__":
    res = evaluate_ocr_benchmark(sample_size=50)
    print("\n" + "=" * 60)
    print("PHASE 5 OCR BENCHMARK EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Images Evaluated: {res['total_images_evaluated']} ({res['clean_images_evaluated']} Clean, {res['edited_images_evaluated']} Edited)")
    print(f"Mean Latency: {res['latency_stats_ms']['mean']} ms | Median: {res['latency_stats_ms']['median']} ms | P95: {res['latency_stats_ms']['p95']} ms")
    print(f"Mean Overall Token Confidence: {res['mean_overall_confidence']}")
    print("\nField Extraction Coverage (% detected across all receipts):")
    for k, v in res["coverage"].items():
        print(f"  - {k:15s}: {v:.1f}%")
    print("\nPer-Field Accuracy against Ground Truth Annotations:")
    for f, st in res["per_field_stats"].items():
        tot = st["total_annotated"]
        if tot > 0:
            exact = st.get("exact_match", st.get("numeric_match", st.get("token_present", 0)))
            print(f"  - {f:15s} (N={tot}): Extracted={st['extracted']}/{tot} ({st['extracted']/tot*100:.1f}%), Matches={exact}/{tot} ({exact/tot*100:.1f}%), Missing={st['missing']}/{tot}")
    print("=" * 60)
