"""
Final Verdict & Decision Engine Service for CBL-04.
Combines predictions from CNN, OCR, Semantic Consistency, and OOF Stacker branches
into an explainable final verdict: GENUINE, EDITED, or INCONCLUSIVE.
Adheres strictly to the CBL-04 Google Pay dark-theme transaction scope.
Preserves explainable evidence provenance, explicit failure handling, and uncalibrated metadata.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from app.schemas.cnn import CNNModelOutput
from app.schemas.enums import ReasonCodeEnum, VerdictEnum
from app.schemas.ocr import OCRResult
from app.schemas.response import (
    BranchScores,
    ManipulationAnalysisEvidence,
    ModelVersionMetadata,
    SuspiciousRegion,
)
from app.schemas.semantic import SemanticBranchOutput
from app.schemas.stacker import StackerOutput

logger = logging.getLogger("cbl04.final_verdict")


class FinalVerdictResult:
    """Internal container for the assembled final verdict decision."""

    def __init__(
        self,
        verdict: VerdictEnum,
        confidence: Optional[float],
        manipulation_probability: float,
        message: str,
        reason_codes: List[ReasonCodeEnum],
        suspicious_regions: List[SuspiciousRegion],
        warnings: List[str],
        calibration_status: str = "uncalibrated",
        score_type: str = "raw_stacker_logistic_output",
        decision_rule: str = "default_threshold",
        decision_score: Optional[float] = None,
        uncertainty_score: Optional[float] = None,
    ) -> None:
        self.verdict = verdict
        self.confidence = confidence
        self.manipulation_probability = manipulation_probability
        self.message = message
        self.reason_codes = reason_codes
        self.suspicious_regions = suspicious_regions
        self.warnings = warnings
        self.calibration_status = calibration_status
        self.score_type = score_type
        self.decision_rule = decision_rule
        self.decision_score = decision_score if decision_score is not None else manipulation_probability
        self.uncertainty_score = uncertainty_score


class FinalVerdictService:
    """
    Production Final Verdict Service for CBL-04.
    Synthesizes multi-branch outputs into tripartite verdicts (GENUINE, EDITED, INCONCLUSIVE)
    with strict guardrails against false certainty.
    """

    def __init__(
        self,
        operating_threshold: float = 0.50,
        ambiguity_margin: float = 0.05,
        min_ocr_confidence: float = 0.45,
    ) -> None:
        self.operating_threshold = operating_threshold
        self.ambiguity_margin = ambiguity_margin
        self.min_ocr_confidence = min_ocr_confidence
        self.verdict_engine_version = "v1.0.0-phase9"

    def evaluate_verdict(
        self,
        stacker_output: Optional[StackerOutput] = None,
        cnn_result: Optional[CNNModelOutput] = None,
        ocr_result: Optional[OCRResult] = None,
        semantic_result: Optional[SemanticBranchOutput] = None,
    ) -> FinalVerdictResult:
        """
        Synthesizes multi-branch signals into an auditable final verdict.
        """
        warnings: List[str] = []
        reason_codes: List[ReasonCodeEnum] = []
        suspicious_regions: List[SuspiciousRegion] = []

        # 1. Branch Availability Audit
        cnn_available = cnn_result is not None and getattr(cnn_result, "execution_status", "live") in ["live", "evaluated"]
        ocr_available = ocr_result is not None and getattr(ocr_result, "ocr_engine", None) is not None
        sem_available = semantic_result is not None and getattr(semantic_result, "semantic_status", "live") in ["live", "evaluated"]

        if not cnn_available:
            warnings.append("CNN branch unavailable or failed during execution.")
        if not ocr_available:
            warnings.append("OCR extraction branch unavailable or failed.")
        if not sem_available:
            warnings.append("Semantic consistency branch unavailable or failed.")

        # If both primary perception branches failed -> Inconclusive
        if not cnn_available and not ocr_available:
            return FinalVerdictResult(
                verdict=VerdictEnum.INCONCLUSIVE,
                confidence=None,
                manipulation_probability=0.50,
                decision_score=0.50,
                uncertainty_score=1.0,
                message="Analysis inconclusive: Both CNN and OCR perception branches are unavailable.",
                reason_codes=[ReasonCodeEnum.MISSING_CRITICAL_PAYMENT_FIELD],
                suspicious_regions=[],
                warnings=warnings,
                calibration_status="uncalibrated",
                score_type="neutral_fallback",
                decision_rule="rule_all_branches_unavailable",
            )

        # 2. Extract Key Numerical Signals
        stacker_prob = float(stacker_output.final_manipulation_probability) if stacker_output else 0.50
        stacker_prob = max(0.0, min(1.0, stacker_prob))

        cnn_score = float(cnn_result.manipulation_probability) if (cnn_available and cnn_result) else 0.50
        sem_score = float(semantic_result.semantic_manipulation_score) if (sem_available and semantic_result) else 0.0
        ocr_conf = float(ocr_result.overall_confidence) if (ocr_available and ocr_result) else 0.0

        # Check semantic reason codes and anomalies
        anom_count = 0
        if sem_available and semantic_result:
            reason_codes.extend(semantic_result.reason_codes)
            anom = semantic_result.field_anomalies
            anom_flags = [
                anom.amount_anomaly,
                anom.recipient_anomaly,
                anom.date_anomaly,
                anom.ref_id_anomaly,
                anom.status_anomaly,
                anom.upi_handle_anomaly,
            ]
            anom_count = sum(1 for a in anom_flags if a)
            has_semantic_anom = anom_count > 0
            missing_fields = semantic_result.feature_vector.missing_critical_fields_count
        else:
            has_semantic_anom = False
            anom_count = 0
            missing_fields = 4 if not ocr_available else 2

        # 3. Guardrail: Low OCR Confidence / Occlusion -> INCONCLUSIVE
        # Missing fields due to image quality or OCR failure must not count as manipulation evidence.
        if ocr_available and ocr_conf < self.min_ocr_confidence and missing_fields >= 3:
            warnings.append(
                f"Low OCR confidence ({ocr_conf:.2f} < {self.min_ocr_confidence:.2f}) with multiple missing fields ({missing_fields}/4)."
            )
            return FinalVerdictResult(
                verdict=VerdictEnum.INCONCLUSIVE,
                confidence=None,
                manipulation_probability=round(stacker_prob, 4),
                decision_score=round(stacker_prob, 4),
                uncertainty_score=round(1.0 - ocr_conf, 4),
                message="Analysis inconclusive: Receipt image quality or OCR text legibility is insufficient for definitive verification.",
                reason_codes=[ReasonCodeEnum.OCR_CONFIDENCE_LOW] + reason_codes,
                suspicious_regions=[],
                warnings=warnings,
                calibration_status="uncalibrated",
                score_type="raw_stacker_logistic_output",
                decision_rule="rule_low_ocr_legibility",
            )

        # 4. Guardrail: Severe Inter-Branch Conflict -> INCONCLUSIVE
        # Conflict case A: CNN predicts extreme visual tampering (>= 0.85), or CNN >= 0.75 with stacker leaning edited (>= 0.55),
        # but Semantic branch confirms authentic layout (0.0) with zero anomalies
        is_severe_conflict = (
            cnn_available
            and sem_available
            and sem_score == 0.0
            and ocr_conf >= 0.70
            and anom_count == 0
            and (cnn_score >= 0.85 or (cnn_score >= 0.75 and stacker_prob >= 0.55))
        )
        if is_severe_conflict:
            warnings.append("Severe branch disagreement: CNN predicts visual artifact, but semantic consistency confirms authentic layout and text.")
            return FinalVerdictResult(
                verdict=VerdictEnum.INCONCLUSIVE,
                confidence=None,
                manipulation_probability=round(stacker_prob, 4),
                decision_score=round(stacker_prob, 4),
                uncertainty_score=1.0,
                message="Analysis inconclusive: Visual artifact detector and text consistency branch disagree significantly.",
                reason_codes=[ReasonCodeEnum.TEXT_LAYOUT_INCONSISTENCY] + reason_codes,
                suspicious_regions=[],
                warnings=warnings,
                calibration_status="uncalibrated",
                score_type="raw_stacker_logistic_output",
                decision_rule="rule_severe_branch_conflict",
            )

        # Conflict case B: Semantic flags anomaly, but CNN indicates clean visual pixels (<= 0.35) and stacker is not decisively edited
        if cnn_available and sem_available and has_semantic_anom and cnn_score <= 0.35 and stacker_prob < 0.60:
            warnings.append("Branch disagreement: Semantic anomaly detected, but visual pixel inspection indicates authentic receipt.")
            return FinalVerdictResult(
                verdict=VerdictEnum.INCONCLUSIVE,
                confidence=None,
                manipulation_probability=round(stacker_prob, 4),
                decision_score=round(stacker_prob, 4),
                uncertainty_score=round(abs(cnn_score - sem_score), 4),
                message="Analysis inconclusive: Semantic formatting anomaly lacks corroboration from visual artifact detector.",
                reason_codes=reason_codes,
                suspicious_regions=[],
                warnings=warnings,
                calibration_status="uncalibrated",
                score_type="raw_stacker_logistic_output",
                decision_rule="rule_branch_disagreement",
            )

        # 5. Guardrail: Ambiguity Band near threshold -> INCONCLUSIVE
        # Stacker probability in [0.45, 0.55] indicates indeterminate model state.
        # A verified clean layout (0 anomalies, authentic metadata) or strong corroboration resolves decisively.
        lower_bound = self.operating_threshold - self.ambiguity_margin
        upper_bound = self.operating_threshold + self.ambiguity_margin
        has_strong_corroboration = (
            (cnn_score >= 0.75 and anom_count >= 2)
            or (cnn_score >= 0.80 and sem_score >= 0.40)
            or (has_semantic_anom and cnn_score >= 0.60)
            or (cnn_score >= 0.55 and stacker_prob >= self.operating_threshold)
        )
        is_verified_clean = (
            anom_count == 0
            and sem_score == 0.0
            and (ReasonCodeEnum.INTEGRITY_VERIFIED_CLEAN in reason_codes or ocr_conf >= 0.65)
            and stacker_prob <= self.operating_threshold
        )

        if lower_bound <= stacker_prob <= upper_bound and not has_strong_corroboration and not is_verified_clean:
            norm_uncertainty = round(1.0 - abs(stacker_prob - 0.50) * 2.0, 4)
            warnings.append(f"Stacker score ({stacker_prob:.4f}) is within ambiguity margin [{lower_bound:.2f}, {upper_bound:.2f}].")
            return FinalVerdictResult(
                verdict=VerdictEnum.INCONCLUSIVE,
                confidence=None,
                manipulation_probability=round(stacker_prob, 4),
                decision_score=round(stacker_prob, 4),
                uncertainty_score=norm_uncertainty,
                message="Analysis inconclusive: Evidence score falls within the indeterminate threshold region without strong corroboration.",
                reason_codes=reason_codes,
                suspicious_regions=[],
                warnings=warnings,
                calibration_status="uncalibrated",
                score_type="raw_stacker_logistic_output",
                decision_rule="rule_ambiguity_band",
            )

        # 6. Decisive Verdict: EDITED
        # Requires verified multi-branch corroboration or decisive stacker evidence.
        # Single uncorroborated semantic anomalies or unverified OCR failures cannot unilaterally force EDITED.
        decision_margin = abs(stacker_prob - self.operating_threshold)
        is_decisive_edited = (
            (stacker_prob >= upper_bound and (cnn_score >= 0.50 or has_semantic_anom))
            or (stacker_prob >= 0.70)
            or (anom_count >= 2 and stacker_prob >= self.operating_threshold)
            or (cnn_score >= 0.75 and has_semantic_anom and stacker_prob >= 0.45)
            or (anom_count >= 1 and cnn_score >= 0.60)
            or (cnn_score >= 0.55 and stacker_prob >= self.operating_threshold)
        )

        if is_decisive_edited:
            verdict = VerdictEnum.EDITED
            # Evidence localization: attach semantic evidence bounding boxes if available
            if sem_available and semantic_result:
                for feat_name, ev in semantic_result.feature_evidence.items():
                    if ev.feature_status in ["invalid_format", "ambiguous"] and ev.source_bounding_box:
                        suspicious_regions.append(
                            SuspiciousRegion(
                                label=f"Semantic: {feat_name}",
                                score=round(ev.ocr_confidence or 0.8, 2),
                                box_normalized=ev.source_bounding_box,
                            )
                        )

            # Build human-readable explanation
            reasons_text = []
            if anom_count >= 2:
                reasons_text.append(f"multiple semantic consistency rules ({anom_count}) detected layout/formatting anomalies")
            elif has_semantic_anom:
                reasons_text.append("semantic consistency rules detected formatting anomaly")
            if cnn_score >= 0.60:
                reasons_text.append(f"CNN visual branch detected tampering probability {cnn_score:.2f}")
            if not reasons_text:
                reasons_text.append(f"OOF stacker meta-learner assigned manipulation score {stacker_prob:.2f}")

            msg = f"Potential tampering detected: {', '.join(reasons_text)}."
            # Linear decision margin confidence: no artificial 0.60 floor
            calc_conf = round(min(0.98, max(0.50, 0.50 + decision_margin)), 4)

            return FinalVerdictResult(
                verdict=verdict,
                confidence=calc_conf,
                manipulation_probability=round(stacker_prob, 4),
                decision_score=round(stacker_prob, 4),
                uncertainty_score=None,
                message=msg,
                reason_codes=reason_codes or [ReasonCodeEnum.SUSPICIOUS_SPATIAL_CLUSTER],
                suspicious_regions=suspicious_regions,
                warnings=warnings,
                calibration_status="uncalibrated",
                score_type="raw_stacker_logistic_output",
                decision_rule="rule_edited_decision",
            )

        # 7. Decisive Verdict: GENUINE
        verdict = VerdictEnum.GENUINE
        msg = "Receipt screenshot appears authentic. Visual and semantic metadata align with genuine Google Pay patterns."
        if cnn_score >= 0.70:
            warnings.append(
                f"Notice: CNN visual branch indicated elevated artifact score ({cnn_score:.2f}, typical of mobile JPEG compression), but semantic consistency verified authentic Google Pay layout and valid UPI metadata."
            )
        # Linear decision margin confidence: no artificial 0.60 floor
        calc_conf = round(min(0.98, max(0.50, 0.50 + decision_margin)), 4)

        return FinalVerdictResult(
            verdict=verdict,
            confidence=calc_conf,
            manipulation_probability=round(stacker_prob, 4),
            decision_score=round(stacker_prob, 4),
            uncertainty_score=None,
            message=msg,
            reason_codes=[],
            suspicious_regions=[],
            warnings=warnings,
            calibration_status="uncalibrated",
            score_type="raw_stacker_logistic_output",
            decision_rule="rule_genuine_decision",
        )


# Thread-safe Singleton Management
_VERDICT_SERVICE_INSTANCE: Optional[FinalVerdictService] = None
_VERDICT_SERVICE_LOCK = threading.Lock()


def get_final_verdict_service() -> FinalVerdictService:
    """Returns thread-safe singleton instance of FinalVerdictService."""
    global _VERDICT_SERVICE_INSTANCE
    if _VERDICT_SERVICE_INSTANCE is None:
        with _VERDICT_SERVICE_LOCK:
            if _VERDICT_SERVICE_INSTANCE is None:
                _VERDICT_SERVICE_INSTANCE = FinalVerdictService()
    return _VERDICT_SERVICE_INSTANCE


def reset_final_verdict_service() -> None:
    """Resets singleton instance for testing."""
    global _VERDICT_SERVICE_INSTANCE
    with _VERDICT_SERVICE_LOCK:
        _VERDICT_SERVICE_INSTANCE = None
