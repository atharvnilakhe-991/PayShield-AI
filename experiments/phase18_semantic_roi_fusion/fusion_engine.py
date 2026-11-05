"""
Phase 18: Evidence Fusion Engine.
Implements the 6 required experimental fusion strategies:
  A. OCR-only baseline
  B. Semantic/geometric-only baseline
  C. Candidate crop-model-only baseline (inference-selected crops)
  D. Conservative semantic-guided ROI fusion (proposed system)
  E. Ablation without semantic routing (blind maximum pooling)
  F. Ablation without crop-model scores (semantic + OCR features only)

Supports conservative INCONCLUSIVE outcomes under:
- Candidate region unavailable / OCR unlocalized
- Ambiguous OCR extraction (e.g. amount ambiguity)
- Branch conflict between visual and semantic evidence
- Ambiguity margin [0.35, 0.60]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FusionResult:
    sample_id: str
    strategy_name: str
    decision_score: float         # Continuous score in [0.0, 1.0] for ranking / ROC evaluation
    verdict: str                  # "CLEAN", "EDITED", or "INCONCLUSIVE"
    confidence: float             # Calibrated confidence / certainty in the verdict
    uncertainty: float            # Uncertainty score [0.0, 1.0]
    is_decisive: bool             # True if CLEAN or EDITED; False if INCONCLUSIVE
    routed_candidate_count: int   # Number of candidate crops routed to the visual model
    routed_candidates: List[str]  # List of candidate IDs evaluated (e.g. ["amount"])
    semantic_score: float
    visual_score: float
    ocr_confidence: float
    decision_reasons: List[str]
    warnings: List[str]


class Phase18FusionEngine:
    """
    Implements all 6 evaluation fusion strategies with strict leakage safeguards.
    """

    def __init__(
        self,
        ambiguity_lower: float = 0.35,
        ambiguity_upper: float = 0.60,
    ) -> None:
        self.ambiguity_lower = ambiguity_lower
        self.ambiguity_upper = ambiguity_upper

    def evaluate_strategy_a_ocr_only(
        self,
        sample_id: str,
        ocr_detailed_fields: Dict[str, Any],
        ocr_overall_confidence: float,
    ) -> FusionResult:
        """
        Strategy A: OCR-only baseline.
        Uses extraction status, token confidences, and parsing ambiguity.
        """
        amt_status = ocr_detailed_fields.get("amount", {}).get("extraction_status", "not_found")
        amt_conf = ocr_detailed_fields.get("amount", {}).get("confidence", 0.0) or 0.0

        reasons = []
        warnings = []

        if amt_status == "not_found":
            score = 0.80
            reasons.append("CRITICAL_FIELD_AMOUNT_NOT_FOUND")
        elif amt_status == "ambiguous":
            score = 0.50
            reasons.append("AMOUNT_PARSING_AMBIGUOUS")
            warnings.append("OCR amount parsing yielded multiple conflicting candidates.")
        elif ocr_overall_confidence < 0.60:
            score = 0.65
            reasons.append("OCR_OVERALL_CONFIDENCE_LOW")
        else:
            score = max(0.10, round(1.0 - ocr_overall_confidence, 4))
            reasons.append("OCR_EXTRACTION_CONFIDENT")

        if score >= 0.65:
            verdict = "EDITED"
            conf = score
        elif score < self.ambiguity_lower and amt_status == "extracted":
            verdict = "CLEAN"
            conf = 1.0 - score
        else:
            verdict = "INCONCLUSIVE"
            conf = 1.0 - 2.0 * abs(score - 0.50)

        return FusionResult(
            sample_id=sample_id,
            strategy_name="strategy_a_ocr_only",
            decision_score=round(score, 4),
            verdict=verdict,
            confidence=round(conf, 4),
            uncertainty=round(1.0 - conf if verdict != "INCONCLUSIVE" else conf, 4),
            is_decisive=(verdict != "INCONCLUSIVE"),
            routed_candidate_count=0,
            routed_candidates=[],
            semantic_score=0.0,
            visual_score=0.0,
            ocr_confidence=round(ocr_overall_confidence, 4),
            decision_reasons=reasons,
            warnings=warnings,
        )

    def evaluate_strategy_b_semantic_only(
        self,
        sample_id: str,
        semantic_manipulation_score: float,
        field_anomalies: Dict[str, bool],
        reason_codes: List[str],
        ocr_detailed_fields: Dict[str, Any],
    ) -> FusionResult:
        """
        Strategy B: Semantic/geometric-only baseline.
        Uses deterministic rules and semantic manipulation score.
        """
        amt_status = ocr_detailed_fields.get("amount", {}).get("extraction_status", "not_found")
        score = semantic_manipulation_score
        reasons = list(reason_codes)
        warnings = []

        if amt_status == "ambiguous":
            verdict = "INCONCLUSIVE"
            conf = 0.50
            warnings.append("OCR amount is ambiguous.")
        elif score >= 0.50:
            verdict = "EDITED"
            conf = score
        elif score < 0.20:
            verdict = "CLEAN"
            conf = 1.0 - score
        else:
            verdict = "INCONCLUSIVE"
            conf = 0.50

        return FusionResult(
            sample_id=sample_id,
            strategy_name="strategy_b_semantic_only",
            decision_score=round(score, 4),
            verdict=verdict,
            confidence=round(conf, 4),
            uncertainty=round(1.0 - conf if verdict != "INCONCLUSIVE" else conf, 4),
            is_decisive=(verdict != "INCONCLUSIVE"),
            routed_candidate_count=0,
            routed_candidates=[],
            semantic_score=round(score, 4),
            visual_score=0.0,
            ocr_confidence=0.85,
            decision_reasons=reasons,
            warnings=warnings,
        )

    def evaluate_strategy_c_crop_only(
        self,
        sample_id: str,
        candidate_crop_scores: Dict[str, float],
    ) -> FusionResult:
        """
        Strategy C: Candidate crop-model-only baseline using inference-selected crops.
        Evaluates mean probability across available detected fields (amount, receiver, transaction_id).
        """
        scores = list(candidate_crop_scores.values())
        if not scores:
            return FusionResult(
                sample_id=sample_id,
                strategy_name="strategy_c_crop_only",
                decision_score=0.50,
                verdict="INCONCLUSIVE",
                confidence=0.50,
                uncertainty=1.0,
                is_decisive=False,
                routed_candidate_count=0,
                routed_candidates=[],
                semantic_score=0.0,
                visual_score=0.50,
                ocr_confidence=0.85,
                decision_reasons=["NO_CANDIDATE_CROPS_AVAILABLE"],
                warnings=["Zero candidate ROIs localized by OCR."],
            )

        mean_score = float(np.mean(scores))
        if mean_score >= 0.70:
            verdict = "EDITED"
            conf = mean_score
        elif mean_score < 0.45:
            verdict = "CLEAN"
            conf = 1.0 - mean_score
        else:
            verdict = "INCONCLUSIVE"
            conf = 0.50

        return FusionResult(
            sample_id=sample_id,
            strategy_name="strategy_c_crop_only",
            decision_score=round(mean_score, 4),
            verdict=verdict,
            confidence=round(conf, 4),
            uncertainty=round(1.0 - conf if verdict != "INCONCLUSIVE" else conf, 4),
            is_decisive=(verdict != "INCONCLUSIVE"),
            routed_candidate_count=len(scores),
            routed_candidates=list(candidate_crop_scores.keys()),
            semantic_score=0.0,
            visual_score=round(mean_score, 4),
            ocr_confidence=0.85,
            decision_reasons=["CROP_MODEL_UNWEIGHTED_MEAN"],
            warnings=[],
        )

    def evaluate_strategy_d_semantic_guided_fusion(
        self,
        sample_id: str,
        semantic_manipulation_score: float,
        field_anomalies: Dict[str, bool],
        reason_codes: List[str],
        ocr_detailed_fields: Dict[str, Any],
        routed_candidates: Dict[str, float],  # map of {candidate_id: crop_score}
        ocr_overall_confidence: float = 0.85,
    ) -> FusionResult:
        """
        Strategy D: Conservative Semantic-Guided ROI Fusion (Proposed).
        - If candidate(s) are routed by semantic flags:
            fuses semantic score and routed crop score.
        - If clean semantic profile (0 candidates routed):
            preserves clean visual prior (no false alarms on verified clean receipts).
        - Conservative decision rules enforce INCONCLUSIVE under ambiguity or branch conflict.
        """
        amt_status = ocr_detailed_fields.get("amount", {}).get("extraction_status", "not_found")
        has_field_anomalies = any(field_anomalies.values())
        reasons = list(reason_codes)
        warnings = []

        if len(routed_candidates) > 0:
            # Candidate field(s) were routed
            routed_crop_scores = list(routed_candidates.values())
            visual_score = float(np.max(routed_crop_scores))
            # Balanced 50/50 fusion of semantic indicator and visual patch probability
            decision_score = 0.50 * semantic_manipulation_score + 0.50 * visual_score
            reasons.append(f"SEMANTIC_ROUTED_CANDIDATES: {list(routed_candidates.keys())}")
        else:
            # Clean semantic profile: zero candidates flagged
            visual_score = 0.15  # clean visual baseline prior
            decision_score = 0.10
            reasons.append("ZERO_SEMANTIC_ANOMALIES_ROUTING_BYPASSED")

        # Decision Rule
        if amt_status == "ambiguous":
            verdict = "INCONCLUSIVE"
            conf = 0.50
            warnings.append("Amount parsing was ambiguous; honest abstention enforced.")
        elif len(routed_candidates) > 0:
            # Routed candidate: require concordant evidence for EDITED
            if decision_score >= self.ambiguity_upper and visual_score >= 0.60 and has_field_anomalies:
                verdict = "EDITED"
                conf = decision_score
            elif visual_score < 0.40 and semantic_manipulation_score > 0.30:
                # Evidence branch conflict (semantic flagged, visual clean)
                verdict = "INCONCLUSIVE"
                conf = 0.50
                warnings.append("Evidence branch conflict: semantic flagged anomaly, but visual crop clean.")
            elif decision_score < self.ambiguity_lower and not has_field_anomalies:
                verdict = "CLEAN"
                conf = 1.0 - decision_score
            else:
                verdict = "INCONCLUSIVE"
                conf = 0.50
                warnings.append("Score fell in ambiguity band [0.35, 0.60].")
        else:
            # Zero candidates routed (clean semantic profile)
            if ocr_overall_confidence >= 0.70 and amt_status == "extracted":
                verdict = "CLEAN"
                conf = 0.90
            else:
                verdict = "INCONCLUSIVE"
                conf = 0.50

        return FusionResult(
            sample_id=sample_id,
            strategy_name="strategy_d_semantic_guided_fusion",
            decision_score=round(decision_score, 4),
            verdict=verdict,
            confidence=round(conf, 4),
            uncertainty=round(1.0 - conf if verdict != "INCONCLUSIVE" else conf, 4),
            is_decisive=(verdict != "INCONCLUSIVE"),
            routed_candidate_count=len(routed_candidates),
            routed_candidates=list(routed_candidates.keys()),
            semantic_score=round(semantic_manipulation_score, 4),
            visual_score=round(visual_score, 4),
            ocr_confidence=round(ocr_overall_confidence, 4),
            decision_reasons=reasons,
            warnings=warnings,
        )

    def evaluate_strategy_e_ablation_blind_max(
        self,
        sample_id: str,
        semantic_manipulation_score: float,
        all_candidate_crop_scores: Dict[str, float],
    ) -> FusionResult:
        """
        Strategy E: Ablation without semantic routing (Blind Max Pooling).
        Blindly takes maximum across all available detected fields without semantic gating.
        """
        scores = list(all_candidate_crop_scores.values())
        max_visual = float(np.max(scores)) if scores else 0.50
        decision_score = 0.50 * semantic_manipulation_score + 0.50 * max_visual

        if decision_score >= 0.60:
            verdict = "EDITED"
            conf = decision_score
        elif decision_score < 0.35:
            verdict = "CLEAN"
            conf = 1.0 - decision_score
        else:
            verdict = "INCONCLUSIVE"
            conf = 0.50

        return FusionResult(
            sample_id=sample_id,
            strategy_name="strategy_e_ablation_blind_max",
            decision_score=round(decision_score, 4),
            verdict=verdict,
            confidence=round(conf, 4),
            uncertainty=round(1.0 - conf if verdict != "INCONCLUSIVE" else conf, 4),
            is_decisive=(verdict != "INCONCLUSIVE"),
            routed_candidate_count=len(scores),
            routed_candidates=list(all_candidate_crop_scores.keys()),
            semantic_score=round(semantic_manipulation_score, 4),
            visual_score=round(max_visual, 4),
            ocr_confidence=0.85,
            decision_reasons=["BLIND_MAX_POOLING_ABLATION"],
            warnings=["Ablation negative control: semantic routing disabled."],
        )

    def evaluate_strategy_f_ablation_no_crop(
        self,
        sample_id: str,
        semantic_manipulation_score: float,
        ocr_detailed_fields: Dict[str, Any],
        reason_codes: List[str],
    ) -> FusionResult:
        """
        Strategy F: Ablation without crop-model scores.
        Uses only semantic heuristics and OCR status without any neural visual features.
        """
        amt_status = ocr_detailed_fields.get("amount", {}).get("extraction_status", "not_found")
        score = semantic_manipulation_score

        if amt_status == "ambiguous":
            verdict = "INCONCLUSIVE"
            conf = 0.50
        elif score >= 0.50:
            verdict = "EDITED"
            conf = score
        elif score < 0.20:
            verdict = "CLEAN"
            conf = 1.0 - score
        else:
            verdict = "INCONCLUSIVE"
            conf = 0.50

        return FusionResult(
            sample_id=sample_id,
            strategy_name="strategy_f_ablation_no_crop",
            decision_score=round(score, 4),
            verdict=verdict,
            confidence=round(conf, 4),
            uncertainty=round(1.0 - conf if verdict != "INCONCLUSIVE" else conf, 4),
            is_decisive=(verdict != "INCONCLUSIVE"),
            routed_candidate_count=0,
            routed_candidates=[],
            semantic_score=round(score, 4),
            visual_score=0.0,
            ocr_confidence=0.85,
            decision_reasons=list(reason_codes) + ["NO_CROP_MODEL_ABLATION"],
            warnings=["Ablation control: visual crop classifier removed."],
        )

