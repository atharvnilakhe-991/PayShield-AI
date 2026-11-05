"""
Phase 13: Confidence Semantics and Inconclusive UI Correction Tests.
Validates:
1. INCONCLUSIVE verdicts return confidence=None (never 99.9% or 100% fake certainty).
2. Boundary ambiguity is accurately reflected in uncertainty_score [0.0 to 1.0].
3. Uncalibrated decision scores are cleanly exposed as decision_score without being mislabeled as calibrated probabilities.
4. Schema validation and serialization handle null confidence and optional scores safely.
5. Frontend contract compatibility and score presentation guarantees.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io

from app.main import app
from app.schemas.enums import ReasonCodeEnum, VerdictEnum
from app.schemas.response import (
    AnalyzeResponse,
    BranchScores,
    ManipulationAnalysisEvidence,
)
from app.schemas.cnn import CNNModelOutput
from app.schemas.common import ModelVersionInfo, NormalizedBoundingBox
from app.schemas.enums import ReasonCodeEnum, VerdictEnum
from app.schemas.ocr import ExtractedTransactionData, OCRResult
from app.schemas.semantic import (
    FieldAnomalyIndicators,
    OCRConfidenceInfo,
    SemanticBranchOutput,
    SemanticFeatureEvidence,
    SemanticFeatureVector,
)
from app.schemas.stacker import StackerOutput
from app.services.final_verdict_service import (
    FinalVerdictService,
    get_final_verdict_service,
    reset_final_verdict_service,
)


def _mock_stacker(prob: float = 0.50) -> StackerOutput:
    return StackerOutput(
        final_manipulation_probability=prob,
        calibrated_confidence=abs(prob - 0.5) * 2.0,
        predicted_class="EDITED" if prob >= 0.5 else "GENUINE",
        stacker_model_version="v1.0.0-phase8",
        feature_importance_ranking=["cnn", "semantic"],
    )


def _mock_cnn(score: float = 0.50) -> CNNModelOutput:
    return CNNModelOutput(
        manipulation_probability=score,
        predicted_class="EDITED" if score >= 0.5 else "GENUINE",
        model_info=ModelVersionInfo(
            model_name="resnet18_multitask",
            model_version="v1.0.0-phase7",
        ),
        execution_status="live",
    )


def _mock_ocr(conf: float = 0.90) -> OCRResult:
    return OCRResult(
        fields=ExtractedTransactionData(
            amount="₹1,250",
            transaction_id="UPI/2026/001",
            date="12 Oct 2026",
            sender="Alice",
            receiver="Bob",
        ),
        overall_confidence=conf,
        ocr_engine="easyocr",
        ocr_version="1.7.2",
    )


def _mock_semantic(score: float = 0.0, has_anom: bool = False, missing: int = 0) -> SemanticBranchOutput:
    anom = FieldAnomalyIndicators(
        amount_anomaly=has_anom,
        recipient_anomaly=False,
        status_anomaly=False,
        date_anomaly=False,
        ref_id_anomaly=False,
        layout_anomaly=False,
        upi_handle_anomaly=False,
    )
    ev = {}
    if has_anom:
        ev["amount_valid_format"] = SemanticFeatureEvidence(
            feature_name="amount_valid_format",
            feature_value=False,
            feature_status="invalid_format",
            source_ocr_text="{500",
            source_bounding_box=NormalizedBoundingBox(xmin=0.1, ymin=0.2, xmax=0.4, ymax=0.3),
            rule_id="R_AMT_001",
            explanation="Invalid rupee prefix symbol",
            ocr_confidence=0.92,
        )

    fv = SemanticFeatureVector(
        amount_valid_format=not has_anom,
        amount_currency_symbol_valid=True,
        amount_numeric_value_reasonable=True,
        ref_id_length_valid=True,
        ref_id_numeric_only=True,
        status_text_authentic=True,
        upi_handle_valid_format=True,
        date_chronology_valid=True,
        time_format_valid=True,
        sender_receiver_distinct=True,
        gpay_keywords_present=True,
        missing_critical_fields_count=missing,
        total_fields_detected=4 - missing,
    )
    return SemanticBranchOutput(
        semantic_manipulation_score=score,
        field_anomalies=anom,
        feature_vector=fv,
        feature_evidence=ev,
        ocr_confidence_info=OCRConfidenceInfo(
            average_confidence=0.90,
            min_field_confidence=0.85,
            critical_fields_confidence=0.90,
            total_detected_tokens=25,
        ),
        reason_codes=[ReasonCodeEnum.AMOUNT_FORMAT_ANOMALY] if has_anom else [],
        model_info=ModelVersionInfo(model_name="rule_based_semantic_engine", model_version="v1.0.0-phase6"),
        semantic_status="live",
        processing_time_ms=12.0,
    )


class TestConfidenceSemantics:
    """Verifies that confidence semantics are correct across all decision branches."""

    def setup_method(self):
        reset_final_verdict_service()
        self.service = get_final_verdict_service()

    def teardown_method(self):
        reset_final_verdict_service()

    def test_inconclusive_ambiguity_band_has_null_confidence_and_high_uncertainty(self):
        """
        When stacker score is near threshold (0.4997), verdict is INCONCLUSIVE.
        Confidence MUST be None (not 0.9994 or 99.9%).
        Uncertainty score MUST capture the high boundary ambiguity (> 0.99).
        Decision score MUST be the uncalibrated score (0.4997).
        """
        stacker = _mock_stacker(prob=0.4997)
        cnn = _mock_cnn(score=0.51)
        ocr = _mock_ocr(conf=0.90)
        sem = _mock_semantic(score=0.0, has_anom=False)

        res = self.service.evaluate_verdict(stacker, cnn, ocr, sem)
        assert res.verdict == VerdictEnum.INCONCLUSIVE
        assert res.decision_rule == "rule_ambiguity_band"
        assert res.confidence is None
        assert res.uncertainty_score is not None
        assert res.uncertainty_score >= 0.99
        assert res.decision_score == 0.4997
        assert res.manipulation_probability == 0.4997

    def test_inconclusive_branch_conflict_has_null_confidence(self):
        """Severe inter-branch disagreement must yield INCONCLUSIVE with null confidence."""
        stacker = _mock_stacker(prob=0.60)
        cnn = _mock_cnn(score=0.85)
        ocr = _mock_ocr(conf=0.92)
        sem = _mock_semantic(score=0.0, has_anom=False)

        res = self.service.evaluate_verdict(stacker, cnn, ocr, sem)
        assert res.verdict == VerdictEnum.INCONCLUSIVE
        assert res.decision_rule == "rule_severe_branch_conflict"
        assert res.confidence is None
        assert res.uncertainty_score == 1.0
        assert res.decision_score == 0.60

    def test_inconclusive_low_ocr_legibility_has_null_confidence(self):
        """Low OCR confidence with missing critical fields must yield INCONCLUSIVE with null confidence."""
        stacker = _mock_stacker(prob=0.68)
        cnn = _mock_cnn(score=0.65)
        ocr = _mock_ocr(conf=0.25)
        sem = _mock_semantic(score=0.0, has_anom=False, missing=3)

        res = self.service.evaluate_verdict(stacker, cnn, ocr, sem)
        assert res.verdict == VerdictEnum.INCONCLUSIVE
        assert res.decision_rule == "rule_low_ocr_legibility"
        assert res.confidence is None
        assert res.uncertainty_score == round(1.0 - 0.25, 4)
        assert res.decision_score == 0.68

    def test_inconclusive_all_branches_failed_has_null_confidence(self):
        """Total branch failure must yield INCONCLUSIVE with null confidence and maximum uncertainty."""
        res = self.service.evaluate_verdict(None, None, None, None)
        assert res.verdict == VerdictEnum.INCONCLUSIVE
        assert res.decision_rule == "rule_all_branches_unavailable"
        assert res.confidence is None
        assert res.uncertainty_score == 1.0
        assert res.decision_score == 0.50

    def test_genuine_decisive_has_uncalibrated_decision_score_and_null_uncertainty(self):
        """Decisive GENUINE provides uncalibrated decision_score and null uncertainty_score."""
        stacker = _mock_stacker(prob=0.15)
        cnn = _mock_cnn(score=0.18)
        ocr = _mock_ocr(conf=0.92)
        sem = _mock_semantic(score=0.0, has_anom=False)

        res = self.service.evaluate_verdict(stacker, cnn, ocr, sem)
        assert res.verdict == VerdictEnum.GENUINE
        assert res.decision_rule == "rule_genuine_decision"
        assert res.decision_score == 0.15
        assert res.uncertainty_score is None
        assert res.confidence is not None
        assert 0.50 <= res.confidence <= 0.98

    def test_edited_decisive_has_uncalibrated_decision_score_and_null_uncertainty(self):
        """Decisive EDITED provides uncalibrated decision_score and null uncertainty_score."""
        stacker = _mock_stacker(prob=0.88)
        cnn = _mock_cnn(score=0.85)
        ocr = _mock_ocr(conf=0.91)
        sem = _mock_semantic(score=0.75, has_anom=True)

        res = self.service.evaluate_verdict(stacker, cnn, ocr, sem)
        assert res.verdict == VerdictEnum.EDITED
        assert res.decision_rule == "rule_edited_decision"
        assert res.decision_score == 0.88
        assert res.uncertainty_score is None
        assert res.confidence is not None
        assert 0.50 <= res.confidence <= 0.98


class TestAnalyzeResponseSchemaValidation:
    """Verifies that AnalyzeResponse schema handles null confidence and optional scores safely."""

    def test_response_allows_null_confidence_for_inconclusive(self):
        resp = AnalyzeResponse(
            request_id="test-null-conf",
            verdict=VerdictEnum.INCONCLUSIVE,
            confidence=None,
            decision_score=0.5002,
            uncertainty_score=0.9996,
            manipulation_probability=0.5002,
            message="Evidence is ambiguous.",
            manipulation_analysis=ManipulationAnalysisEvidence(detected=False),
        )
        assert resp.confidence is None
        assert resp.decision_score == 0.5002
        assert resp.uncertainty_score == 0.9996

        dump = resp.model_dump()
        assert dump["confidence"] is None
        assert dump["decision_score"] == 0.5002
        assert dump["uncertainty_score"] == 0.9996

        json_str = resp.model_dump_json()
        data = json.loads(json_str)
        assert data["confidence"] is None

    def test_response_auto_syncs_decision_score(self):
        """If decision_score is not passed explicitly, it syncs with stacker_score."""
        resp = AnalyzeResponse(
            request_id="test-sync",
            verdict=VerdictEnum.GENUINE,
            confidence=0.85,
            manipulation_probability=0.15,
            stacker_score=0.15,
            message="Clean receipt.",
            manipulation_analysis=ManipulationAnalysisEvidence(detected=False),
        )
        assert resp.decision_score == 0.15

    def test_response_handles_missing_optional_scores_gracefully(self):
        """AnalyzeResponse can be instantiated without decision_score or uncertainty_score."""
        resp = AnalyzeResponse(
            request_id="test-minimal",
            verdict=VerdictEnum.GENUINE,
            confidence=0.80,
            manipulation_probability=0.20,
            message="Minimal valid response.",
            manipulation_analysis=ManipulationAnalysisEvidence(detected=False),
        )
        assert resp.decision_score == 0.20
        assert resp.uncertainty_score is None


class TestLiveAPIConfidenceSemantics:
    """Tests the live FastAPI analyze endpoint for confidence and decision score contract."""

    def setup_method(self):
        self.client = TestClient(app)
        reset_final_verdict_service()

    def test_analyze_endpoint_blank_image_returns_inconclusive_with_null_confidence(self):
        """A blank dark image triggers low OCR legibility -> INCONCLUSIVE with null confidence."""
        img = Image.new("RGB", (350, 700), color=(18, 18, 18))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        response = self.client.post(
            "/api/v1/analyze",
            files={"image": ("blank_receipt.png", buf.getvalue(), "image/png")},
            data={"request_id": "req-blank-001"},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["verdict"] == "INCONCLUSIVE"
        assert data["confidence"] is None
        assert data["uncertainty_score"] is not None
        assert data["decision_score"] is not None
        assert 0.0 <= data["decision_score"] <= 1.0
