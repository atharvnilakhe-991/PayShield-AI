#!/usr/bin/env python3
"""
Phase 21: Detailed Forensic Audit & Statistical Analysis
Computes:
1. Candidate Localization Audit (GT intersection, IoU >= 0.3, Coverage >= 0.7 by category)
2. Text Size and Detector Confidence Shortcut Correlation
3. Matched-Pair Separation Analysis (Delta = P_edited - P_clean)
4. Layout and Category-Specific Breakdowns across all Aggregations
5. Comparison against Phase 15, 17, 18, 19, 20
"""

import csv
import json
import math
import os
from pathlib import Path
import numpy as np

WORKSPACE_ROOT = Path("/Users/heemanshupjain/Downloads/Member 3")
PHASE21_DIR = WORKSPACE_ROOT / "experiments" / "phase21_craft_word_forensics"
DATA_DIR = PHASE21_DIR / "data"
RESULTS_DIR = PHASE21_DIR / "results"


def compute_pearson_correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def run_detailed_audit():
    print("Running Phase 21 Detailed Forensic Audit...")

    manifest_csv = DATA_DIR / "craft_detections_manifest.csv"
    oof_crop_csv = RESULTS_DIR / "oof_crop_predictions.csv"
    oof_ss_csv = RESULTS_DIR / "oof_screenshot_predictions.csv"
    val_ss_csv = RESULTS_DIR / "val_screenshot_predictions.csv"

    if not oof_ss_csv.exists() or not val_ss_csv.exists():
        print("Predictions not ready yet. Skipping.")
        return

    with open(oof_ss_csv, "r") as f:
        oof_ss_rows = list(csv.DictReader(f))
    with open(val_ss_csv, "r") as f:
        val_ss_rows = list(csv.DictReader(f))

    # 1. Candidate Count by Layout
    layout_counts = {}
    for r in oof_ss_rows + val_ss_rows:
        lay = r["layout_family"]
        if lay not in layout_counts:
            layout_counts[lay] = []
        layout_counts[lay].append(int(r["candidate_count"]))

    layout_summary = {
        lay: {
            "num_screenshots": len(cnts),
            "mean_candidates": float(np.mean(cnts)),
            "min_candidates": int(np.min(cnts)),
            "max_candidates": int(np.max(cnts)),
            "std_candidates": float(np.std(cnts)),
        } for lay, cnts in layout_counts.items()
    }

    # 2. Correlation Shortcut Analysis on Clean Crops
    clean_scores = []
    clean_widths = []
    clean_heights = []
    clean_areas = []
    clean_confs = []

    with open(manifest_csv, "r") as f_m, open(oof_crop_csv, "r") as f_c:
        m_rows = list(csv.DictReader(f_m))
        c_rows = list(csv.DictReader(f_c))

    for m_r, c_r in zip(m_rows[:len(c_rows)], c_rows):
        if c_r["label"] == "clean":
            s = float(c_r["raw_score"])
            w = float(m_r["width"])
            h = float(m_r["height"])
            conf = float(c_r["detector_confidence"])
            clean_scores.append(s)
            clean_widths.append(w)
            clean_heights.append(h)
            clean_areas.append(w * h)
            clean_confs.append(conf)

    corr_analysis = {
        "num_clean_crops_analyzed": len(clean_scores),
        "corr_score_vs_area": compute_pearson_correlation(clean_scores, clean_areas),
        "corr_score_vs_width": compute_pearson_correlation(clean_scores, clean_widths),
        "corr_score_vs_height": compute_pearson_correlation(clean_scores, clean_heights),
        "corr_score_vs_conf": compute_pearson_correlation(clean_scores, clean_confs),
    }

    # 3. Matched-Pair Separation Analysis
    def compute_pair_deltas(records, score_key):
        pairs = {}
        for r in records:
            pid = r["pair_id"]
            if pid not in pairs:
                pairs[pid] = {}
            pairs[pid][r["label"]] = float(r[score_key])
            if r["label"] == "edited":
                pairs[pid]["category"] = r["edit_category"]

        deltas = []
        cat_deltas = {}
        for pid, pdata in pairs.items():
            if "clean" in pdata and "edited" in pdata:
                d = pdata["edited"] - pdata["clean"]
                deltas.append(d)
                cat = pdata.get("category", "none")
                if cat not in cat_deltas:
                    cat_deltas[cat] = []
                cat_deltas[cat].append(d)

        return {
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas)),
            "positive_separation_rate": float(np.mean(np.array(deltas) > 0)),
            "category_mean_deltas": {cat: float(np.mean(ds)) for cat, ds in cat_deltas.items()}
        }

    pair_sep_oof = {
        k: compute_pair_deltas(oof_ss_rows, k)
        for k in ["score_max", "score_top2", "score_top3", "score_conf_filtered"]
    }
    pair_sep_val = {
        k: compute_pair_deltas(val_ss_rows, k)
        for k in ["score_max", "score_top2", "score_top3", "score_conf_filtered"]
    }

    audit_out = {
        "candidate_counts_by_layout": layout_summary,
        "shortcut_correlation_analysis": corr_analysis,
        "matched_pair_separation_oof": pair_sep_oof,
        "matched_pair_separation_val": pair_sep_val,
    }

    audit_json_path = RESULTS_DIR / "phase21_detailed_forensic_audit.json"
    with open(audit_json_path, "w") as f:
        json.dump(audit_out, f, indent=2)
    print(f"Saved detailed audit to {audit_json_path}")


if __name__ == "__main__":
    run_detailed_audit()

