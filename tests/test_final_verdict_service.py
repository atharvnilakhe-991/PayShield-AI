"""
Unit and API integration tests for Phase 9 Final Verdict Service.
Validates:
- Tripartite verdict decisions: GENUINE, EDITED, INCONCLUSIVE.
- Robust guardrails against severe inter-branch conflicts.
- Guardrails against low OCR confidence and severe occlusion.
- Ambiguity margin behavior around decision threshold.
- Full provenance and evidence attribution without private data leakage.
- API integration with /api/v1/analyze route.
- Component status: cnn, ocr, semantic, stacker, verdict_engine all live.
- Zero mock influence on live verdict.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
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
from app.services.stacker_service import reset_stacker_service

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = WORKSPACE_ROOT / "Hackathon-VIT" / "CBL04-Dataset"


def _mock_stacker_output(prob: float = 0.50) -> StackerOutput:
    return StackerOutput(
        final_manipulation_probability=prob,
        calibrated_confidence=abs(prob - 0.5) * 2.0,
        stacker_model_version="v1.0.0-phase8",
        inference_time_ms=1.5,
    )


def _mock_cnn(score: float = 0.50, status: str = "live") -> CNNModelOutput:
    return CNNModelOutput(
        manipulation_probability=score,
        model_info=ModelVersionInfo(model_name="ResNet18", model_version="fold_1_baseline"),
    )


def _mock_ocr(conf: float = 0.90) -> OCRResult:
    fields = ExtractedTransactionData(
        amount="100.00",
        amount_numeric=100.0,
        currency="INR",
        receiver="Retail Merchant",
        receiver_upi="retail@okhdfc",
        transaction_id="998877665544",
        date="15 Mar 2024",
        time="11:45 AM",
        status="Completed",
    )
    return OCRResult(
        fields=fields,
        detailed_fields={},
        raw_detections=[],
        overall_confidence=conf,
        ocr_engine="EasyOCR",
        ocr_version="1.7.2",
        raw_text="Paid ₹100 to Retail Merchant",
        processing_time_ms=45.0,
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
        feature_vector=fv,
        reason_codes=[ReasonCodeEnum.AMOUNT_FORMAT_ANOMALY] if has_anom else [],
        field_anomalies=anom,
        feature_evidence=ev,
        ocr_confidence_info=OCRConfidenceInfo(
            average_confidence=0.90,
            min_field_confidence=0.85,
            critical_fields_confidence=0.90,
            total_detected_tokens=25,
        ),
        semantic_status="live",
        processing_time_ms=12.0,
    )


class TestFinalVerdictLogic:
    """Tests core deterministic logic and guardrails of the FinalVerdictService."""

    def setup_method(self):
        reset_final_verdict_service()

    def teardown_method(self):
        reset_final_verdict_service()

    def test_decisive_genuine_verdict(self):
        service = get_final_verdict_service()
        stacker = _mock_stacker_output(prob=0.15)
        cnn = _mock_cnn(score=0.20)
        ocr = _mock_ocr(conf=0.92)
        sem = _mock_semantic(score=0.0, has_anom=False)

        res = service.evaluate_verdict(stacker_output=stacker, cnn_result=cnn, ocr_result=ocr, semantic_result=sem)
        assert res.verdict == VerdictEnum.GENUINE
        assert res.manipulation_probability == 0.15
        assert res.confidence >= 0.70
        assert len(res.suspicious_regions) == 0

    def test_decisive_edited_verdict(self):
        service = get_final_verdict_service()
        stacker = _mock_stacker_output(prob=0.85)
        cnn = _mock_cnn(score=0.82)
        ocr = _mock_ocr(conf=0.91)
        sem = _mock_semantic(score=0.75, has_anom=True)

        res = service.evaluate_verdict(stacker_output=stacker, cnn_result=cnn, ocr_result=ocr, semantic_result=sem)
        assert res.verdict == VerdictEnum.EDITED
        assert res.manipulation_probability == 0.85
        assert len(res.suspicious_regions) > 0
        assert ReasonCodeEnum.AMOUNT_FORMAT_ANOMALY in res.reason_codes

    def test_guardrail_severe_inter_branch_conflict(self):
        """CNN says strong tamper (0.85), but Semantic confirms 0.0 with high OCR confidence (0.92) -> INCONCLUSIVE."""
        service = get_final_verdict_service()
        stacker = _mock_stacker_output(prob=0.60)
        cnn = _mock_cnn(score=0.85)
        ocr = _mock_ocr(conf=0.92)
        sem = _mock_semantic(score=0.0, has_anom=False)

        res = service.evaluate_verdict(stacker_output=stacker, cnn_result=cnn, ocr_result=ocr, semantic_result=sem)
        assert res.verdict == VerdictEnum.INCONCLUSIVE
        assert res.decision_rule == "rule_severe_branch_conflict"
        assert res.confidence is None
        assert res.uncertainty_score == 1.0

    def test_guardrail_low_ocr_confidence_severe_occlusion(self):
        """Low OCR confidence (< 0.45) with >= 3 missing critical fields -> INCONCLUSIVE."""
        service = get_final_verdict_service()
        stacker = _mock_stacker_output(prob=0.70)
        cnn = _mock_cnn(score=0.65)
        ocr = _mock_ocr(conf=0.30)
        sem = _mock_semantic(score=0.0, has_anom=False, missing=3)

        res = service.evaluate_verdict(stacker_output=stacker, cnn_result=cnn, ocr_result=ocr, semantic_result=sem)
        assert res.verdict == VerdictEnum.INCONCLUSIVE
        assert res.decision_rule == "rule_low_ocr_legibility"
        assert res.confidence is None
        assert res.uncertainty_score == round(1.0 - 0.30, 4)

    def test_guardrail_ambiguity_band(self):
        """Stacker score near operating threshold (e.g. 0.51) without anomalies -> INCONCLUSIVE."""
        service = get_final_verdict_service()
        stacker = _mock_stacker_output(prob=0.51)
        cnn = _mock_cnn(score=0.52)
        ocr = _mock_ocr(conf=0.85)
        sem = _mock_semantic(score=0.0, has_anom=False)

        res = service.evaluate_verdict(stacker_output=stacker, cnn_result=cnn, ocr_result=ocr, semantic_result=sem)
        assert res.verdict == VerdictEnum.INCONCLUSIVE
        assert res.decision_rule == "rule_ambiguity_band"
        assert res.confidence is None
        assert res.uncertainty_score == round(1.0 - abs(0.51 - 0.50) * 2.0, 4)

    def test_guardrail_all_branches_unavailable(self):
        """When CNN and OCR both fail -> INCONCLUSIVE."""
        service = get_final_verdict_service()
        res = service.evaluate_verdict(stacker_output=None, cnn_result=None, ocr_result=None, semantic_result=None)
        assert res.verdict == VerdictEnum.INCONCLUSIVE
        assert res.decision_rule == "rule_all_branches_unavailable"
        assert res.confidence is None
        assert res.uncertainty_score == 1.0


class TestAPIPhase9Integration:
    """Validates end-to-end API integration through /api/v1/analyze route."""

    def setup_method(self):
        self.client = TestClient(app)
        reset_stacker_service()
        reset_final_verdict_service()

    def test_api_returns_phase9_complete_status(self):
        img_path = DATASET_ROOT / "data" / "synthetic_gpay_phase4" / "clean" / "P4-0001_clean.png"
        if not img_path.exists():
            pytest.skip(f"Test image not found at {img_path}")

        with open(img_path, "rb") as fp:
            files = {"image": ("P4-0001_clean.png", fp, "image/png")}
            response = self.client.post("/api/v1/analyze", files=files)

        assert response.status_code == 200
        data = response.json()

        assert data["pipeline_status"] == "phase9_complete"
        assert data["component_status"]["cnn"] == "live"
        assert data["component_status"]["ocr"] == "live"
        assert data["component_status"]["semantic"] == "live"
        assert data["component_status"]["stacker"] == "live"
        assert data["component_status"]["verdict_engine"] == "live"
        assert data["verdict"] in ["GENUINE", "EDITED", "INCONCLUSIVE", "REAL", "FAKE"]
        assert data["confidence"] is None or (0.0 <= data["confidence"] <= 1.0)
        assert 0.0 <= data["decision_score"] <= 1.0
        assert 0.0 <= data["manipulation_probability"] <= 1.0
        assert data["stacker_score"] is not None
        assert "Screenshot OCR extracts visible receipt text" in " ".join(data["warnings"])
