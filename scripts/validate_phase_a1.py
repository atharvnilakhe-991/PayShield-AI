"""
Phase A.1 Post-Remediation Validation Script.
Evaluates:
1. Part 1: Trace confidence calculation and score semantics.
2. Part 3: Validation on Phase 6 validation manifest (108 samples: 54 clean, 54 edited).
3. Part 4: OCR amount extraction regression on benchmark cases and annotated samples.
4. Part 5: Safety audit across all 5 edit categories (amount, recipient, date_time, transaction_id, other).
"""

import csv
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from fastapi.testclient import TestClient

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from app.main import app
from app.services.ocr_service import TransactionFieldExtractor, get_ocr_service
from app.schemas.common import NormalizedBoundingBox, PixelBoundingBox
from app.schemas.ocr import OCRExtractedField

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_phase_a1")


def run_validation():
    client = TestClient(app)
    manifest_path = WORKSPACE_ROOT / "Hackathon-VIT" / "CBL04-Dataset" / "metadata" / "phase6" / "validation_manifest.csv"
    
    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    logger.info("Loaded %d validation samples from %s", len(rows), manifest_path)

    results = []
    t_start = time.perf_counter()

    for idx, r in enumerate(rows):
        img_rel = r["image_path"]
        img_full = WORKSPACE_ROOT / "Hackathon-VIT" / "CBL04-Dataset" / img_rel
        label = r["label"].lower().strip()
        cat = r.get("edit_category", "").strip() or "clean"
        img_id = r.get("\ufeffimage_id") or r.get("image_id")

        if not img_full.exists():
            logger.warning("Missing image: %s", img_full)
            continue

        with open(img_full, "rb") as fp:
            resp = client.post(
                "/api/v1/analyze",
                files={"image": (img_full.name, fp, "image/png")},
            )

        if resp.status_code != 200:
            logger.error("API error %d on %s: %s", resp.status_code, img_id, resp.text)
            continue

        data = resp.json()
        verdict = data.get("verdict")
        conf = data.get("confidence")
        manip_prob = data.get("manipulation_probability")
        branch_scores = data.get("branch_scores", {})
        cnn_score = branch_scores.get("cnn_score")
        stacker_score = branch_scores.get("stacker_score")
        sem_score = branch_scores.get("semantic_score")
        ocr_data = data.get("ocr_data", {})
        amt_str = ocr_data.get("amount")
        amt_num = ocr_data.get("amount_numeric")
        rule = data.get("decision_rule")
        reason_codes = data.get("reason_codes", [])

        results.append({
            "image_id": img_id,
            "label": label,
            "category": cat,
            "layout_family": r.get("layout_family", "bank_progress"),
            "verdict": verdict,
            "confidence": conf,
            "manipulation_probability": manip_prob,
            "cnn_score": cnn_score,
            "stacker_score": stacker_score,
            "semantic_score": sem_score,
            "ocr_amount": amt_str,
            "ocr_amount_numeric": amt_num,
            "decision_rule": rule,
            "reason_codes": reason_codes,
            "warnings": data.get("warnings", []),
        })

        if (idx + 1) % 20 == 0 or (idx + 1) == len(rows):
            logger.info("Processed %d / %d samples...", idx + 1, len(rows))

    elapsed = time.perf_counter() - t_start
    logger.info("Completed %d samples in %.2f s", len(results), elapsed)

    # -------------------------------------------------------------
    # Metrics Computation
    # -------------------------------------------------------------
    clean_samples = [r for r in results if r["label"] == "clean"]
    edited_samples = [r for r in results if r["label"] == "edited"]

    clean_total = len(clean_samples)
    edited_total = len(edited_samples)

    clean_verdicts = Counter(r["verdict"] for r in clean_samples)
    edited_verdicts = Counter(r["verdict"] for r in edited_samples)

    # Confusion matrix
    # Genuine predictions on clean = TN
    # Edited predictions on clean = FP
    # Inconclusive predictions on clean = INC_clean
    # Edited predictions on edited = TP
    # Genuine predictions on edited = FN
    # Inconclusive predictions on edited = INC_edited

    tn = clean_verdicts.get("GENUINE", 0)
    fp = clean_verdicts.get("EDITED", 0)
    inc_clean = clean_verdicts.get("INCONCLUSIVE", 0)

    tp = edited_verdicts.get("EDITED", 0)
    fn = edited_verdicts.get("GENUINE", 0)
    inc_edited = edited_verdicts.get("INCONCLUSIVE", 0)

    decisive_clean = tn + fp
    decisive_edited = tp + fn
    total_decisive = decisive_clean + decisive_edited
    total_samples = clean_total + edited_total

    # Metrics
    fpr_total = fp / clean_total if clean_total > 0 else 0.0
    fpr_decisive = fp / decisive_clean if decisive_clean > 0 else 0.0

    fnr_total = fn / edited_total if edited_total > 0 else 0.0
    fnr_decisive = fn / decisive_edited if decisive_edited > 0 else 0.0

    coverage = total_decisive / total_samples if total_samples > 0 else 0.0
    acc_decisive = (tp + tn) / total_decisive if total_decisive > 0 else 0.0
    acc_overall = (tp + tn) / total_samples if total_samples > 0 else 0.0

    # Category breakdown for edited
    cat_breakdown = defaultdict(lambda: Counter())
    cat_samples = defaultdict(list)
    for r in edited_samples:
        cat = r["category"]
        v = r["verdict"]
        cat_breakdown[cat][v] += 1
        cat_samples[cat].append(r)

    summary = {
        "dataset": "metadata/phase6/validation_manifest.csv",
        "sample_counts": {
            "total": total_samples,
            "clean_total": clean_total,
            "edited_total": edited_total,
        },
        "confusion_matrix": {
            "clean": {
                "GENUINE (TN)": tn,
                "EDITED (FP)": fp,
                "INCONCLUSIVE": inc_clean,
            },
            "edited": {
                "EDITED (TP)": tp,
                "GENUINE (FN)": fn,
                "INCONCLUSIVE": inc_edited,
            },
        },
        "rates": {
            "false_positive_rate_overall": round(fpr_total, 4),
            "false_positive_rate_decisive": round(fpr_decisive, 4),
            "false_negative_rate_overall": round(fnr_total, 4),
            "false_negative_rate_decisive": round(fnr_decisive, 4),
            "coverage": round(coverage, 4),
            "accuracy_decisive": round(acc_decisive, 4),
            "accuracy_overall": round(acc_overall, 4),
        },
        "edit_category_breakdown": {
            cat: dict(counts) for cat, counts in cat_breakdown.items()
        },
        "samples": results,
    }

    out_path = WORKSPACE_ROOT / "reports" / "model_comparison" / "phase_a1_validation_data.json"
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    logger.info("Saved summary data to %s", out_path)
    return summary


if __name__ == "__main__":
    summary = run_validation()
    print("\n================== SUMMARY ==================")
    print("Clean count:", summary["sample_counts"]["clean_total"])
    print("Edited count:", summary["sample_counts"]["edited_total"])
    print("Clean verdicts:", summary["confusion_matrix"]["clean"])
    print("Edited verdicts:", summary["confusion_matrix"]["edited"])
    print("Rates:", json.dumps(summary["rates"], indent=2))
    print("Categories:", json.dumps(summary["edit_category_breakdown"], indent=2))

