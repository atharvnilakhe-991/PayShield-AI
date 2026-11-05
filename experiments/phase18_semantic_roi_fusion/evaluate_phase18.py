"""
Phase 18 Comprehensive Evaluation Script.
Evaluates all 6 inference-time fusion strategies on the 108 validation images:
  Strategy A: OCR-only baseline
  Strategy B: Semantic/geometric-only baseline
  Strategy C: Candidate crop-model-only baseline (inference-selected crops)
  Strategy D: Conservative semantic-guided ROI fusion (proposed system)
  Strategy E: Ablation without semantic routing (blind maximum pooling)
  Strategy F: Ablation without crop-model scores (semantic + OCR only)

Outputs exact numerators, denominators, confusion matrices, decisive coverage,
category breakdowns, paired differences, localization recall, and comparison against
Phase 15 Global CNN and Phase 17 Oracle Upper Bound.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.phase18_semantic_roi_fusion.candidate_selector import InferenceCandidateSelector
from experiments.phase18_semantic_roi_fusion.crop_evaluator import InferenceCropEvaluator
from experiments.phase18_semantic_roi_fusion.fusion_engine import Phase18FusionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase18.eval")


def roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    desc = np.argsort(y_scores)[::-1]
    y_t = y_true[desc]
    y_s = y_scores[desc]
    distinct = np.where(np.diff(y_s))[0]
    th_idxs = np.r_[distinct, y_t.size - 1]
    tps = np.cumsum(y_t)[th_idxs]
    fps = 1 + th_idxs - tps
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    if fps[-1] <= 0 or tps[-1] <= 0:
        return 0.5
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    desc = np.argsort(y_scores)[::-1]
    y_t = y_true[desc]
    y_s = y_scores[desc]
    distinct = np.where(np.diff(y_s))[0]
    th_idxs = np.r_[distinct, y_t.size - 1]
    tps = np.cumsum(y_t)[th_idxs]
    fps = 1 + th_idxs - tps
    prec = tps / (tps + fps)
    rec = tps / tps[-1] if tps[-1] > 0 else np.zeros_like(tps)
    rec = np.r_[0, rec]
    prec = np.r_[1, prec]
    return float(np.sum((rec[1:] - rec[:-1]) * (prec[1:] + prec[:-1]) / 2.0))


def compute_discrete_metrics(y_true: np.ndarray, verdicts: List[str], y_scores: np.ndarray) -> Dict[str, Any]:
    n = len(y_true)
    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))

    decisive_mask = np.array([v != "INCONCLUSIVE" for v in verdicts])
    decisive_cnt = int(np.sum(decisive_mask))
    coverage = decisive_cnt / n if n > 0 else 0.0

    # Counts based on binary verdicts (where INCONCLUSIVE is treated as non-decision / negative for strict detection)
    tp = sum(1 for i, v in enumerate(verdicts) if v == "EDITED" and y_true[i] == 1)
    fp = sum(1 for i, v in enumerate(verdicts) if v == "EDITED" and y_true[i] == 0)
    tn = sum(1 for i, v in enumerate(verdicts) if v == "CLEAN" and y_true[i] == 0)
    fn = sum(1 for i, v in enumerate(verdicts) if v == "CLEAN" and y_true[i] == 1)
    inconclusive_clean = sum(1 for i, v in enumerate(verdicts) if v == "INCONCLUSIVE" and y_true[i] == 0)
    inconclusive_edited = sum(1 for i, v in enumerate(verdicts) if v == "INCONCLUSIVE" and y_true[i] == 1)
    total_inconclusive = inconclusive_clean + inconclusive_edited

    # Decisive accuracy: accuracy among decisive decisions only
    decisive_correct = tp + tn
    decisive_acc = decisive_correct / decisive_cnt if decisive_cnt > 0 else 0.0

    # Overall accuracy (decisive correct / total population)
    overall_acc = (tp + tn) / n if n > 0 else 0.0

    # Clean FPR: FP / Negatives
    clean_fpr = fp / neg if neg > 0 else 0.0
    # Edited Recall: TP / Positives
    edited_recall = tp / pos if pos > 0 else 0.0
    # Specificity: TN / Negatives
    specificity = tn / neg if neg > 0 else 0.0
    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # FNR: FN / Positives
    fnr = fn / pos if pos > 0 else 0.0

    f1_pos = 2 * precision * edited_recall / (precision + edited_recall) if (precision + edited_recall) > 0 else 0.0
    f1_neg = 2 * specificity * (tn / (tn + fn) if (tn + fn) > 0 else 0.0) / (specificity + (tn / (tn + fn) if (tn + fn) > 0 else 0.0)) if (specificity + (tn / (tn + fn) if (tn + fn) > 0 else 0.0)) > 0 else 0.0
    macro_f1 = (f1_pos + f1_neg) / 2.0

    auc_r = roc_auc(y_true, y_scores)
    auc_p = pr_auc(y_true, y_scores)

    clean_scores = y_scores[y_true == 0]
    edited_scores = y_scores[y_true == 1]

    return {
        "n": n,
        "n_pos": pos,
        "n_neg": neg,
        "roc_auc": round(auc_r, 4),
        "pr_auc": round(auc_p, 4),
        "overall_accuracy": round(overall_acc, 4),
        "overall_accuracy_fraction": f"{tp+tn}/{n}",
        "decisive_accuracy": round(decisive_acc, 4),
        "decisive_accuracy_fraction": f"{decisive_correct}/{decisive_cnt}" if decisive_cnt > 0 else "0/0",
        "decisive_coverage": round(coverage, 4),
        "decisive_coverage_fraction": f"{decisive_cnt}/{n}",
        "clean_fpr": round(clean_fpr, 4),
        "clean_fpr_fraction": f"{fp}/{neg}",
        "edited_recall": round(edited_recall, 4),
        "edited_recall_fraction": f"{tp}/{pos}",
        "specificity": round(specificity, 4),
        "specificity_fraction": f"{tn}/{neg}",
        "precision": round(precision, 4),
        "precision_fraction": f"{tp}/{tp+fp}" if (tp + fp) > 0 else "0/0",
        "fnr": round(fnr, 4),
        "fnr_fraction": f"{fn}/{pos}",
        "macro_f1": round(macro_f1, 4),
        "clean_mean": round(float(np.mean(clean_scores)), 4) if len(clean_scores) else 0.0,
        "clean_std": round(float(np.std(clean_scores)), 4) if len(clean_scores) else 0.0,
        "edited_mean": round(float(np.mean(edited_scores)), 4) if len(edited_scores) else 0.0,
        "edited_std": round(float(np.std(edited_scores)), 4) if len(edited_scores) else 0.0,
        "mean_diff": round(float(np.mean(edited_scores) - np.mean(clean_scores)), 4) if len(clean_scores) else 0.0,
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "inconclusive_clean": inconclusive_clean,
            "inconclusive_edited": inconclusive_edited,
            "total_inconclusive": total_inconclusive,
        },
    }


def main():
    cache_path = PROJECT_ROOT / "experiments" / "phase18_semantic_roi_fusion" / "data" / "val_ocr_semantic_cache.json"
    with open(cache_path) as fp:
        cache = json.load(fp)

    p17_records_path = PROJECT_ROOT / "experiments" / "phase17_visual_forensics" / "data" / "audit_sample_records.json"
    with open(p17_records_path) as fp:
        p17_records = json.load(fp)
    p17_map = {r["sample_id"]: r for r in p17_records}

    logger.info("Loaded %d validation records from cache.", len(cache))

    selector_guided = InferenceCandidateSelector(semantic_anomaly_routing=True)
    selector_unguided = InferenceCandidateSelector(semantic_anomaly_routing=False)
    evaluator = InferenceCropEvaluator()
    fusion_engine = Phase18FusionEngine()

    # Pre-evaluate date crops if needed
    date_crop_scores = {}
    for r in cache:
        sid = r["sample_id"]
        img_full_path = PROJECT_ROOT / "Hackathon-VIT" / "CBL04-Dataset" / r["image_path"]
        df = r["detailed_fields"]
        d_box = df.get("date", {}).get("norm_box")
        if d_box is not None:
            w, h = r["image_dimensions"]
            from experiments.phase18_semantic_roi_fusion.candidate_selector import CandidateROI
            cand = CandidateROI(
                candidate_id="date",
                field_name="date",
                is_available=True,
                raw_bbox_norm=d_box,
                raw_bbox_pixel=[int(round(d_box[0]*w)), int(round(d_box[1]*h)), int(round(d_box[2]*w)), int(round(d_box[3]*h))],
                ocr_confidence=df["date"].get("confidence"),
                raw_text=df["date"].get("raw_text"),
                extraction_status=df["date"].get("extraction_status", "extracted"),
                anomaly_flag=False,
                routing_reason="date_crop",
                priority_rank=3,
            )
            with Image.open(img_full_path) as src:
                orig_img = src.convert("RGB")
            res = evaluator.extract_and_evaluate(orig_img, cand)
            date_crop_scores[sid] = res.crop_score
        else:
            date_crop_scores[sid] = 0.50

    eval_rows = []
    y_true_list = []

    for r in cache:
        sid = r["sample_id"]
        pid = r["pair_id"]
        label = r["label"]
        y_val = 1 if label == "edited" else 0
        y_true_list.append(y_val)
        cat = r["edit_category"]
        layout = r["layout_family"]
        p17_rec = p17_map[sid]

        # Available candidate crops for this image
        candidate_crop_scores = {
            "amount": p17_rec["amt_prob"],
            "receiver": p17_rec["rec_prob"],
            "transaction_id": p17_rec["txn_prob"],
            "date": date_crop_scores.get(sid, 0.50),
        }

        # 1. Candidate Selection
        candidates = selector_guided.evaluate_candidates(
            image_dimensions=tuple(r["image_dimensions"]),
            detailed_fields=r["detailed_fields"],
            field_anomalies=r["field_anomalies"],
            reason_codes=r["reason_codes"],
            semantic_manipulation_score=r["semantic_manipulation_score"],
        )

        routed_cands = selector_guided.route_candidates(
            candidates=candidates,
            semantic_manipulation_score=r["semantic_manipulation_score"],
            layout_anomaly=r["field_anomalies"].get("layout_anomaly", False),
        )

        routed_crop_scores = {c.candidate_id: candidate_crop_scores.get(c.candidate_id, 0.5) for c in routed_cands}

        # Strategy A: OCR Only
        res_a = fusion_engine.evaluate_strategy_a_ocr_only(
            sample_id=sid,
            ocr_detailed_fields=r["detailed_fields"],
            ocr_overall_confidence=r["ocr_overall_confidence"],
        )

        # Strategy B: Semantic Only
        res_b = fusion_engine.evaluate_strategy_b_semantic_only(
            sample_id=sid,
            semantic_manipulation_score=r["semantic_manipulation_score"],
            field_anomalies=r["field_anomalies"],
            reason_codes=r["reason_codes"],
            ocr_detailed_fields=r["detailed_fields"],
        )

        # Strategy C: Crop Only (Candidate Mean)
        res_c = fusion_engine.evaluate_strategy_c_crop_only(
            sample_id=sid,
            candidate_crop_scores={k: v for k, v in candidate_crop_scores.items() if k in ["amount", "receiver", "transaction_id"]},
        )

        # Strategy D: Semantic-Guided ROI Fusion
        res_d = fusion_engine.evaluate_strategy_d_semantic_guided_fusion(
            sample_id=sid,
            semantic_manipulation_score=r["semantic_manipulation_score"],
            field_anomalies=r["field_anomalies"],
            reason_codes=r["reason_codes"],
            ocr_detailed_fields=r["detailed_fields"],
            routed_candidates=routed_crop_scores,
            ocr_overall_confidence=r["ocr_overall_confidence"],
        )

        # Strategy E: Ablation Blind Max
        res_e = fusion_engine.evaluate_strategy_e_ablation_blind_max(
            sample_id=sid,
            semantic_manipulation_score=r["semantic_manipulation_score"],
            all_candidate_crop_scores={k: v for k, v in candidate_crop_scores.items() if k in ["amount", "receiver", "transaction_id"]},
        )

        # Strategy F: Ablation No Crop
        res_f = fusion_engine.evaluate_strategy_f_ablation_no_crop(
            sample_id=sid,
            semantic_manipulation_score=r["semantic_manipulation_score"],
            ocr_detailed_fields=r["detailed_fields"],
            reason_codes=r["reason_codes"],
        )

        eval_rows.append({
            "sample_id": sid,
            "pair_id": pid,
            "label": label,
            "y_true": y_val,
            "edit_category": cat,
            "layout_family": layout,
            "routed_candidates": [c.candidate_id for c in routed_cands],
            "routed_crop_scores": routed_crop_scores,
            "strategy_a": res_a,
            "strategy_b": res_b,
            "strategy_c": res_c,
            "strategy_d": res_d,
            "strategy_e": res_e,
            "strategy_f": res_f,
        })

    y_true_arr = np.array(y_true_list)

    # Strategy Results Dictionary
    strategy_results = {}
    strategy_keys = [
        ("strategy_a_ocr_only", "strategy_a"),
        ("strategy_b_semantic_only", "strategy_b"),
        ("strategy_c_crop_only", "strategy_c"),
        ("strategy_d_semantic_guided_fusion", "strategy_d"),
        ("strategy_e_ablation_blind_max", "strategy_e"),
        ("strategy_f_ablation_no_crop", "strategy_f"),
    ]

    for strat_name, row_key in strategy_keys:
        scores = np.array([r[row_key].decision_score for r in eval_rows])
        verdicts = [r[row_key].verdict for r in eval_rows]
        strategy_results[strat_name] = compute_discrete_metrics(y_true_arr, verdicts, scores)

    # Paired Delta Analysis for Strategy D
    pairs = sorted(list(set(r["pair_id"] for r in eval_rows)))
    paired_analysis = []
    d_deltas = []
    for pid in pairs:
        c_row = [r for r in eval_rows if r["pair_id"] == pid and r["y_true"] == 0][0]
        e_row = [r for r in eval_rows if r["pair_id"] == pid and r["y_true"] == 1][0]
        cat = e_row["edit_category"]

        d_c = c_row["strategy_d"].decision_score
        d_e = e_row["strategy_d"].decision_score
        delta_d = d_e - d_c
        d_deltas.append(delta_d)

        paired_analysis.append({
            "pair_id": pid,
            "edit_category": cat,
            "clean_score": d_c,
            "edited_score": d_e,
            "delta": round(delta_d, 4),
            "clean_verdict": c_row["strategy_d"].verdict,
            "edited_verdict": e_row["strategy_d"].verdict,
            "clean_routed": c_row["routed_candidates"],
            "edited_routed": e_row["routed_candidates"],
        })

    d_deltas_arr = np.array(d_deltas)
    paired_summary = {
        "total_pairs": len(pairs),
        "e_gt_c": int(np.sum(d_deltas_arr > 1e-6)),
        "e_eq_c": int(np.sum(np.abs(d_deltas_arr) <= 1e-6)),
        "c_gt_e": int(np.sum(d_deltas_arr < -1e-6)),
        "mean_delta": round(float(np.mean(d_deltas_arr)), 4),
        "std_delta": round(float(np.std(d_deltas_arr)), 4),
        "median_delta": round(float(np.median(d_deltas_arr)), 4),
        "min_delta": round(float(np.min(d_deltas_arr)), 4),
        "max_delta": round(float(np.max(d_deltas_arr)), 4),
        "pairs": paired_analysis,
    }

    # Category breakdown for Strategy D
    category_summary = {}
    for cat in ["amount", "date_time", "other", "recipient", "transaction_id"]:
        sub = [r for r in eval_rows if r["edit_category"] == cat or r["y_true"] == 0]
        sub_y = np.array([r["y_true"] for r in sub])
        sub_scores = np.array([r["strategy_d"].decision_score for r in sub])
        sub_verdicts = [r["strategy_d"].verdict for r in sub]
        category_summary[cat] = compute_discrete_metrics(sub_y, sub_verdicts, sub_scores)

    # Candidate Availability and Localization
    fa_dir = PROJECT_ROOT / "Hackathon-VIT" / "CBL04-Dataset" / "annotations" / "phase4" / "field_annotations"
    avail_counts = {"amount": 0, "receiver": 0, "transaction_id": 0, "date": 0, "status": 0}
    loc_recall_05 = 0
    loc_recall_01 = 0
    gt_contained = 0

    def bbox_iou(b1, b2):
        ix0, iy0 = max(b1[0], b2[0]), max(b1[1], b2[1])
        ix1, iy1 = min(b1[2], b2[2]), min(b1[3], b2[3])
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        iarea = iw * ih
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - iarea
        return iarea / union if union > 0 else 0.0

    for r in cache:
        df = r["detailed_fields"]
        w, h = r["image_dimensions"]
        for f in avail_counts:
            if df.get(f, {}).get("norm_box") is not None:
                avail_counts[f] += 1

        if r["label"] == "edited":
            pid = r["pair_id"]
            meta = json.load(open(fa_dir / f"{pid}.json"))
            gt_box = meta["bounding_boxes_xyxy"][0]
            gt_field = meta["edited_field"]

            ocr_f = {"amount": "amount", "recipient": "receiver", "transaction_id": "transaction_id", "date_time": "date"}.get(gt_field)
            if ocr_f and df.get(ocr_f, {}).get("norm_box"):
                nb = df[ocr_f]["norm_box"]
                ocr_pix = [int(round(nb[0]*w)), int(round(nb[1]*h)), int(round(nb[2]*w)), int(round(nb[3]*h))]
                iou = bbox_iou(ocr_pix, gt_box)
                if iou >= 0.50: loc_recall_05 += 1
                if iou >= 0.10: loc_recall_01 += 1
                # containment
                cx, cy = (gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2
                if ocr_pix[0] <= cx <= ocr_pix[2] and ocr_pix[1] <= cy <= ocr_pix[3]:
                    gt_contained += 1

    candidate_telemetry = {
        "candidate_availability": {
            k: {
                "count": v,
                "percentage": round(v / len(cache) * 100.0, 2),
                "fraction": f"{v}/{len(cache)}",
            }
            for k, v in avail_counts.items()
        },
        "edited_field_localization": {
            "total_edited": 54,
            "iou_ge_05": {"count": loc_recall_05, "fraction": f"{loc_recall_05}/54", "percentage": round(loc_recall_05 / 54 * 100.0, 2)},
            "iou_ge_01": {"count": loc_recall_01, "fraction": f"{loc_recall_01}/54", "percentage": round(loc_recall_01 / 54 * 100.0, 2)},
            "gt_center_contained": {"count": gt_contained, "fraction": f"{gt_contained}/54", "percentage": round(gt_contained / 54 * 100.0, 2)},
        },
    }

    full_report = {
        "evaluation_summary": {
            "total_validation_images": len(cache),
            "clean_images": 54,
            "edited_images": 54,
            "matched_pairs": 54,
            "layout_family": "bank_progress",
        },
        "strategy_benchmarks": strategy_results,
        "paired_analysis_strategy_d": paired_summary,
        "category_breakdown_strategy_d": category_summary,
        "candidate_telemetry": candidate_telemetry,
        "comparison_with_prior_baselines": {
            "phase15_global_cnn": {
                "roc_auc": 0.4938,
                "pr_auc": 0.5028,
                "clean_fpr": 1.0000,
                "clean_mean": 0.7000,
                "edited_mean": 0.6995,
                "mean_diff": -0.0005,
            },
            "phase17_target_field_crop_oracle": {
                "roc_auc": 0.6677,
                "pr_auc": 0.6551,
                "clean_fpr": 0.3519,
                "clean_mean": 0.3759,
                "edited_mean": 0.5697,
                "mean_diff": 0.1938,
            },
            "phase17_blind_max_crop": {
                "roc_auc": 0.5117,
                "pr_auc": 0.5079,
                "clean_fpr": 1.0000,
                "clean_mean": 0.9483,
                "edited_mean": 0.9480,
                "mean_diff": -0.0003,
            },
            "phase18_strategy_d_fusion": strategy_results["strategy_d_semantic_guided_fusion"],
        },
    }

    out_file = PROJECT_ROOT / "experiments" / "phase18_semantic_roi_fusion" / "results" / "phase18_evaluation_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as fp:
        json.dump(full_report, fp, indent=2)

    logger.info("Evaluation complete! Results saved to %s", out_file)
    print(json.dumps(strategy_results, indent=2))
    return full_report


if __name__ == "__main__":
    main()

