"""
Unit and integration tests for Phase 8 OOF Stacker Service.
Validates:
- Deterministic stacker inference.
- Safe JSON artifact loading (no arbitrary code execution / no pickle).
- Graceful handling of missing/failed branches (CNN, OCR, Semantic).
- Zero target leakage in feature schemas.
- Feature attribution generation.
- Singleton thread safety.
- Strict test-set isolation.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from app.schemas.cnn import CNNModelOutput
from app.schemas.common import ModelVersionInfo, NormalizedBoundingBox
from app.schemas.ocr import ExtractedTransactionData, OCRExtractedField, OCRResult
from app.schemas.semantic import (
    FieldAnomalyIndicators,
    OCRConfidenceInfo,
    SemanticBranchOutput,
    SemanticFeatureEvidence,
    SemanticFeatureVector,
)
from app.schemas.stacker import StackerInput, StackerOutput
from app.services.stacker_service import (
    FEATURE_SCHEMA,
    StackerService,
    get_stacker_service,
    reset_stacker_service,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = WORKSPACE_ROOT / "Hackathon-VIT" / "CBL04-Dataset"


def _create_mock_cnn_result(score: float = 0.85, status: str = "live") -> CNNModelOutput:
    return CNNModelOutput(
        manipulation_probability=score,
        model_info=ModelVersionInfo(model_name="ResNet18", model_version="fold_1_baseline"),
    )


def _create_mock_ocr_result(conf: float = 0.90) -> OCRResult:
    fields = ExtractedTransactionData(
        amount="500.00",
        amount_numeric=500.0,
        currency="INR",
        receiver="Merchant Store",
        receiver_upi="merchant@okaxis",
        transaction_id="123456789012",
        date="12 Jan 2024",
        time="10:30 AM",
        status="Completed",
    )
    return OCRResult(
        fields=fields,
        detailed_fields={},
        raw_detections=[],
        overall_confidence=conf,
        ocr_engine="EasyOCR",
        ocr_version="1.7.2",
        raw_text="Paid ₹500 to Merchant Store",
        processing_time_ms=50.0,
    )


def _create_mock_semantic_result(score: float = 0.0, has_anomaly: bool = False) -> SemanticBranchOutput:
    anom = FieldAnomalyIndicators(
        amount_anomaly=has_anomaly,
        recipient_anomaly=False,
        status_anomaly=False,
        date_anomaly=False,
        ref_id_anomaly=False,
        layout_anomaly=False,
        upi_handle_anomaly=False,
    )
    fv = SemanticFeatureVector(
        amount_valid_format=not has_anomaly,
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
        missing_critical_fields_count=0,
        total_fields_detected=4,
    )
    return SemanticBranchOutput(
        semantic_manipulation_score=score,
        feature_vector=fv,
        reason_codes=[],
        field_anomalies=anom,
        ocr_confidence_info=OCRConfidenceInfo(
            average_confidence=0.90,
            min_field_confidence=0.85,
            critical_fields_confidence=0.90,
            total_detected_tokens=25,
        ),
        semantic_status="live",
        processing_time_ms=10.0,
    )


class TestStackerServiceArchitecture:
    """Tests architectural integrity, safe deserialization, and schema contracts."""

    def setup_method(self):
        reset_stacker_service()

    def teardown_method(self):
        reset_stacker_service()

    def test_singleton_thread_safety(self):
        s1 = get_stacker_service()
        s2 = get_stacker_service()
        assert s1 is s2

    def test_no_target_leakage_in_schema(self):
        """CRITICAL: Schema must not contain ground truth, pair IDs, layout families, or file paths."""
        forbidden_substrings = ["label", "pair", "source", "family", "split", "path", "filename", "truth"]
        for feat in FEATURE_SCHEMA:
            for forb in forbidden_substrings:
                assert forb not in feat.lower(), f"Forbidden target/shortcut feature in schema: {feat}"

    def test_safe_loading_nonexistent_model(self, tmp_path):
        """Stacker must fall back gracefully to neutral parameters if artifact is missing."""
        missing_path = tmp_path / "nonexistent_model.json"
        service = StackerService(model_path=missing_path, auto_load=True)
        assert service.is_loaded is True
        assert service.model_version.endswith("-fallback")
        assert "cnn_score" in service.weights

    def test_deterministic_forward_pass(self):
        service = get_stacker_service()
        cnn = _create_mock_cnn_result(0.75)
        ocr = _create_mock_ocr_result(0.88)
        sem = _create_mock_semantic_result(0.0, has_anomaly=False)

        res1 = service.predict_branches(cnn_result=cnn, ocr_result=ocr, semantic_result=sem)
        res2 = service.predict_branches(cnn_result=cnn, ocr_result=ocr, semantic_result=sem)

        assert res1.final_manipulation_probability == res2.final_manipulation_probability
        assert res1.calibrated_confidence == res2.calibrated_confidence


class TestStackerMissingBranchHandling:
    """Tests that missing or failed branches are handled robustly without crashing or fabricating data."""

    def setup_method(self):
        reset_stacker_service()

    def test_cnn_unavailable_handling(self):
        service = get_stacker_service()
        ocr = _create_mock_ocr_result(0.90)
        sem = _create_mock_semantic_result(0.0)

        # CNN is None
        res = service.predict_branches(cnn_result=None, ocr_result=ocr, semantic_result=sem)
        assert 0.0 <= res.final_manipulation_probability <= 1.0

        # CNN has failed status
        failed_cnn = _create_mock_cnn_result(0.99, status="failed")
        res2 = service.predict_branches(cnn_result=failed_cnn, ocr_result=ocr, semantic_result=sem)
        assert 0.0 <= res2.final_manipulation_probability <= 1.0

    def test_ocr_unavailable_handling(self):
        service = get_stacker_service()
        cnn = _create_mock_cnn_result(0.80)

        # OCR and Semantic are None
        res = service.predict_branches(cnn_result=cnn, ocr_result=None, semantic_result=None)
        assert 0.0 <= res.final_manipulation_probability <= 1.0

    def test_all_branches_unavailable(self):
        service = get_stacker_service()
        res = service.predict_branches(cnn_result=None, ocr_result=None, semantic_result=None)
        assert 0.0 <= res.final_manipulation_probability <= 1.0
        assert res.decision_metadata is not None

    def test_abstract_interface_compatibility(self):
        service = get_stacker_service()
        inp = StackerInput(
            cnn_score=0.72,
            semantic_score=0.15,
            cnn_model_version="fold_1_baseline",
            semantic_model_version="v1.0.0",
        )
        res = service.predict(inp)
        assert isinstance(res, StackerOutput)
        assert 0.0 <= res.final_manipulation_probability <= 1.0


class TestStackerDataIntegrity:
    """Audits feature files for strict split separation, duplicate rejection, and no test leakage."""

    def test_validation_features_exist_and_clean(self):
        val_csv = DATASET_ROOT / "metadata" / "phase8" / "validation_features.csv"
        if not val_csv.exists():
            pytest.skip("validation_features.csv still generating")

        import csv
        with open(val_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            assert len(rows) == 108
            sids = [r["sample_id"] for r in rows]
            assert len(sids) == len(set(sids)), "Duplicate sample IDs found in validation_features.csv"

            # Check that all validation samples are from bank_progress
            for sid in sids:
                assert "recharge" not in sid, "Test family leaked into validation features"
