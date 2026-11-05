"""
Unit and Integration Tests for Phase 6 Semantic Feature Engine.
Tests all deterministic heuristic rules, evidence provenance, missing-field isolation,
anomaly indicators, reason codes, truthful API status, and CNN verdict decoupling.
"""

import io
import pytest
from PIL import Image
from starlette.testclient import TestClient

from app.core.config import get_settings
from app.main import create_application
from app.schemas.common import NormalizedBoundingBox, PixelBoundingBox
from app.schemas.enums import ReasonCodeEnum, VerdictEnum
from app.schemas.ocr import ExtractedTransactionData, OCRExtractedField, OCRResult
from app.schemas.response import AnalyzeResponse
from app.services.semantic_feature_service import (
    SemanticFeatureService,
    get_semantic_service,
    reset_semantic_service,
)


def _build_mock_ocr_result(
    amount: str = "₹1,500.00",
    amount_numeric: float = 1500.0,
    receiver: str = "Cafe Delight",
    receiver_upi: str = "cafedelight@okhdfcbank",
    sender: str = "John Doe",
    sender_upi: str = "johndoe@oksbi",
    date: str = "14 Oct 2024",
    time: str = "12:45 PM",
    transaction_id: str = "DEMO-UPI-24036454",
    google_txn_id: str = "CICAgIDV7-6OEQ",
    status: str = "Completed",
) -> OCRResult:
    """Helper creating a fully populated synthetic OCRResult."""
    norm_box = NormalizedBoundingBox(xmin=0.1, ymin=0.2, xmax=0.9, ymax=0.3)
    pix_box = PixelBoundingBox(x=100.0, y=200.0, width=800.0, height=50.0)

    detailed = {}
    tokens = []

    def make_field(name: str, raw: str, norm_val: any, height: float = 30.0):
        pb = PixelBoundingBox(x=100.0, y=200.0, width=300.0, height=height)
        f = OCRExtractedField(
            field_name=name,
            raw_text=raw,
            confidence=0.92,
            bounding_box=norm_box,
            pixel_box=pb,
            normalized_value=norm_val,
            is_detected=raw is not None,
            extraction_status="extracted" if raw is not None else "not_found",
        )
        detailed[name] = f
        if raw is not None:
            tokens.append(f)
        return f

    make_field("amount", amount, amount_numeric, height=60.0)
    make_field("receiver", receiver, receiver, height=30.0)
    make_field("receiver_upi", receiver_upi, receiver_upi, height=20.0)
    make_field("sender", sender, sender, height=25.0)
    make_field("sender_upi", sender_upi, sender_upi, height=20.0)
    make_field("date", date, date, height=20.0)
    make_field("time", time, time, height=20.0)
    make_field("transaction_id", transaction_id, transaction_id, height=20.0)
    make_field("google_txn_id", google_txn_id, google_txn_id, height=20.0)
    make_field("status", status, status, height=25.0)

    fields = ExtractedTransactionData(
        amount=amount,
        amount_numeric=amount_numeric,
        currency="INR",
        receiver=receiver,
        receiver_upi=receiver_upi,
        sender=sender,
        sender_upi=sender_upi,
        date=date,
        time=time,
        transaction_id=transaction_id,
        google_txn_id=google_txn_id,
        status=status,
    )

    return OCRResult(
        fields=fields,
        detailed_fields=detailed,
        raw_detections=tokens,
        overall_confidence=0.92,
        ocr_engine="EasyOCR",
        ocr_version="1.7.2",
        raw_text=f"Paid {amount} to {receiver} {receiver_upi} {date} {time} UPI ref: {transaction_id}",
        processing_time_ms=120.0,
    )


class TestSemanticFeatureServiceUnit:
    """Unit tests for semantic consistency feature extraction rules."""

    def setup_method(self):
        reset_semantic_service()

    def test_singleton_lifecycle(self):
        s1 = get_semantic_service()
        s2 = get_semantic_service()
        assert s1 is s2
        reset_semantic_service()
        s3 = get_semantic_service()
        assert s3 is not s1

    def test_complete_valid_ocr_output(self):
        ocr = _build_mock_ocr_result()
        service = get_semantic_service()
        res = service.extract_features(ocr)

        assert res.semantic_status == "live"
        assert res.semantic_manipulation_score == 0.0
        assert res.field_anomalies.amount_anomaly is False
        assert res.field_anomalies.recipient_anomaly is False
        assert res.field_anomalies.date_anomaly is False
        assert res.field_anomalies.ref_id_anomaly is False
        assert res.field_anomalies.status_anomaly is False
        assert ReasonCodeEnum.INTEGRITY_VERIFIED_CLEAN in res.reason_codes

        # Check evidence provenance
        assert "amount_valid_format" in res.feature_evidence
        assert res.feature_evidence["amount_valid_format"].rule_id == "R_AMT_003"
        assert res.feature_evidence["amount_valid_format"].feature_status == "available"
        assert res.feature_evidence["amount_valid_format"].is_missing is False

    def test_missing_amount_does_not_flag_tampering(self):
        """Missing amount sets is_missing=True, status=not_available, but manipulation score remains 0."""
        ocr = _build_mock_ocr_result(amount=None, amount_numeric=None)
        service = get_semantic_service()
        res = service.extract_features(ocr)

        ev = res.feature_evidence["amount_valid_format"]
        assert ev.is_missing is True
        assert ev.feature_status == "not_available"
        assert res.field_anomalies.amount_anomaly is False
        assert res.semantic_manipulation_score == 0.0

    def test_invalid_amount_format_flags_anomaly(self):
        """Amount with invalid grouping like '1,50,00,00.00.00' triggers amount anomaly."""
        ocr = _build_mock_ocr_result(amount="1,500.00.99", amount_numeric=None)
        service = get_semantic_service()
        res = service.extract_features(ocr)

        assert res.field_anomalies.amount_anomaly is True
        assert ReasonCodeEnum.AMOUNT_FORMAT_ANOMALY in res.reason_codes
        assert res.feature_evidence["amount_valid_format"].feature_status == "invalid_format"
        assert res.semantic_manipulation_score > 0.0

    def test_missing_currency_symbol_flags_reason_code(self):
        """Amount without ₹ or currency indicator flags AMOUNT_CURRENCY_SYMBOL_MISMATCH."""
        ocr = _build_mock_ocr_result(amount="1500.00", amount_numeric=1500.0)
        service = get_semantic_service()
        res = service.extract_features(ocr)

        assert res.field_anomalies.amount_anomaly is True
        assert ReasonCodeEnum.AMOUNT_CURRENCY_SYMBOL_MISMATCH in res.reason_codes
        assert res.feature_evidence["amount_currency_symbol_valid"].feature_value is False

    def test_invalid_upi_vpa_syntax_flags_anomaly(self):
        """VPA with illegal characters or double @@ flags UPI_ID_SYNTAX_INVALID."""
        ocr = _build_mock_ocr_result(receiver_upi="bad user@@invalid bank")
        service = get_semantic_service()
        res = service.extract_features(ocr)

        assert res.field_anomalies.upi_handle_anomaly is True
        assert ReasonCodeEnum.UPI_ID_SYNTAX_INVALID in res.reason_codes
        assert res.feature_evidence["upi_handle_valid_format"].feature_status == "invalid_format"
        assert res.semantic_manipulation_score > 0.0

    def test_missing_recipient_does_not_flag_tampering(self):
        """Missing recipient sets is_missing=True without incrementing manipulation score."""
        ocr = _build_mock_ocr_result(receiver=None, receiver_upi=None)
        service = get_semantic_service()
        res = service.extract_features(ocr)

        ev = res.feature_evidence["upi_handle_valid_format"]
        assert ev.is_missing is True
        assert ev.feature_status == "not_available"
        assert res.field_anomalies.recipient_anomaly is False
        assert res.semantic_manipulation_score == 0.0

    def test_sender_receiver_identical_flags_anomaly(self):
        """Sender and receiver having identical name/UPI flags SENDER_RECEIVER_IDENTICAL."""
        ocr = _build_mock_ocr_result(
            receiver="Same Name", receiver_upi="same@okaxis",
            sender="Same Name", sender_upi="same@okaxis"
        )
        service = get_semantic_service()
        res = service.extract_features(ocr)

        assert res.field_anomalies.recipient_anomaly is True
        assert ReasonCodeEnum.SENDER_RECEIVER_IDENTICAL in res.reason_codes
        assert res.feature_evidence["sender_receiver_distinct"].feature_status == "invalid_format"

    def test_invalid_calendar_date_flags_anomaly(self):
        """Impossible calendar date like '31 Feb 2024' or '32 Oct 2024' flags TIMESTAMP_CHRONOLOGY_ANOMALY."""
        ocr = _build_mock_ocr_result(date="31 Feb 2024")
        service = get_semantic_service()
        res = service.extract_features(ocr)

        assert res.field_anomalies.date_anomaly is True
        assert ReasonCodeEnum.TIMESTAMP_CHRONOLOGY_ANOMALY in res.reason_codes
        assert res.feature_evidence["date_chronology_valid"].feature_status == "invalid_format"
        assert res.semantic_manipulation_score > 0.0

    def test_invalid_time_format_flags_anomaly(self):
        """Time with invalid hour like '25:70 PM' flags TIME_FORMAT_INVALID."""
        ocr = _build_mock_ocr_result(time="25:70 PM")
        service = get_semantic_service()
        res = service.extract_features(ocr)

        assert res.field_anomalies.date_anomaly is True
        assert ReasonCodeEnum.TIME_FORMAT_INVALID in res.reason_codes
        assert res.feature_evidence["time_format_valid"].feature_status == "invalid_format"
        assert res.semantic_manipulation_score > 0.0

    def test_invalid_ref_id_length_flags_anomaly(self):
        """Real UPI reference number that is not 12 digits (e.g. 7 digits) flags REF_ID_FORMAT_ANOMALY."""
        ocr = _build_mock_ocr_result(transaction_id="1234567")
        service = get_semantic_service()
        res = service.extract_features(ocr)

        assert res.field_anomalies.ref_id_anomaly is True
        assert ReasonCodeEnum.REF_ID_FORMAT_ANOMALY in res.reason_codes
        assert res.feature_evidence["ref_id_length_valid"].feature_status == "invalid_format"
        assert res.semantic_manipulation_score > 0.0

    def test_synthetic_ref_id_is_valid(self):
        """Synthetic dataset format 'DEMO-UPI-24036454' is recognized as valid."""
        ocr = _build_mock_ocr_result(transaction_id="DEMO-UPI-24036454")
        service = get_semantic_service()
        res = service.extract_features(ocr)

        assert res.field_anomalies.ref_id_anomaly is False
        assert res.feature_evidence["ref_id_length_valid"].feature_value is True
        assert res.feature_evidence["ref_id_length_valid"].feature_status == "available"

    def test_non_authentic_status_text_flags_anomaly(self):
        """Unrecognized status like 'Bypassed' or 'Altered' flags STATUS_TEXT_ANOMALY."""
        ocr = _build_mock_ocr_result(status="Bypassed Verification")
        service = get_semantic_service()
        res = service.extract_features(ocr)

        assert res.field_anomalies.status_anomaly is True
        assert ReasonCodeEnum.STATUS_TEXT_ANOMALY in res.reason_codes
        assert res.feature_evidence["status_text_authentic"].feature_status == "invalid_format"
        assert res.semantic_manipulation_score > 0.0

    def test_no_fabricated_values_on_blank_ocr(self):
        """An empty OCRResult produces missing evidence states and zero hallucinated text."""
        blank_ocr = OCRResult(
            fields=ExtractedTransactionData(),
            detailed_fields={},
            raw_detections=[],
            overall_confidence=0.0,
            ocr_engine="EasyOCR",
            ocr_version="1.7.2",
            raw_text="",
        )
        service = get_semantic_service()
        res = service.extract_features(blank_ocr)

        assert res.semantic_manipulation_score == 0.0
        assert res.feature_evidence["amount_valid_format"].is_missing is True
        assert res.feature_evidence["amount_valid_format"].source_ocr_text is None
        assert res.feature_evidence["ref_id_length_valid"].is_missing is True


class TestSemanticAPIIntegration:
    """Integration tests verifying semantic execution in FastAPI pipeline."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        app = create_application()
        self.client = TestClient(app)

    def _create_synthetic_image(self) -> bytes:
        img = Image.new("RGB", (600, 1000), color=(18, 18, 18))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_api_returns_live_semantic_component_status(self):
        img_bytes = self._create_synthetic_image()
        resp = self.client.post(
            "/api/v1/analyze",
            files={"image": ("test_gpay.png", img_bytes, "image/png")},
            data={"request_id": "test-semantic-api-001"},
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["pipeline_status"] in ["phase6_semantic_integrated", "phase9_complete"]
        assert data["component_status"]["cnn"] == "live"
        assert data["component_status"]["ocr"] == "live"
        assert data["component_status"]["semantic"] == "live"
        assert data["component_status"]["stacker"] in ["mock", "live"]

        # Semantic details presence
        assert "semantic_details" in data
        sem_det = data["semantic_details"]
        assert sem_det is not None
        assert sem_det["semantic_status"] == "live"
        assert "feature_evidence" in sem_det
        assert "amount_valid_format" in sem_det["feature_evidence"]

    def test_cnn_verdict_remains_independent_of_semantic_features(self):
        """Verifies that CNN prediction and verdict are NOT altered by semantic feature score."""
        img_bytes = self._create_synthetic_image()
        resp = self.client.post(
            "/api/v1/analyze",
            files={"image": ("test_gpay.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()

        cnn_score = data["cnn_score"]
        verdict = data["verdict"]

        settings = get_settings()
        expected_verdict = (
            VerdictEnum.EDITED if cnn_score >= settings.CNN_DECISION_THRESHOLD else VerdictEnum.GENUINE
        )
        assert verdict in [expected_verdict, VerdictEnum.INCONCLUSIVE]

