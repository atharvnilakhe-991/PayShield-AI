"""
Test Suite for CBL-04 EasyOCR Service & Field Extraction (Phase 5).
Validates image decoding, EasyOCR forward inference, token bounding box geometry,
conservative field extraction, exception handling, and FastAPI integration.
"""

import io
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from starlette.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.schemas.enums import ErrorCodeEnum, VerdictEnum
from app.schemas.ocr import ExtractedTransactionData, OCRExtractedField, OCRResult
from app.schemas.response import AnalyzeResponse
from app.services.ocr_service import (
    EasyOCRService,
    OCRError,
    OCRImageError,
    OCRInferenceError,
    OCRInitializationError,
    TransactionFieldExtractor,
    get_ocr_service,
    reset_ocr_service,
)


def create_test_receipt_image(fmt: str = "PNG", text_hint: str = "Completed", size=(400, 700)) -> bytes:
    """Helper to generate in-memory synthetic receipt images for OCR testing."""
    from PIL import ImageDraw, ImageFont

    img = Image.new("RGB", size, color=(18, 18, 18))
    draw = ImageDraw.Draw(img)
    # Draw simple simulated receipt elements
    draw.text((50, 50), "Google Pay", fill=(255, 255, 255))
    draw.text((50, 100), "To: Cafe Delight", fill=(255, 255, 255))
    draw.text((50, 130), "cafedelight@okhdfcbank", fill=(200, 200, 200))
    draw.text((50, 200), "₹1,250.00", fill=(255, 255, 255))
    draw.text((50, 260), text_hint, fill=(0, 200, 100))
    draw.text((50, 320), "17 Apr 2025, 1.52am", fill=(200, 200, 200))
    draw.text((50, 380), "UPI transaction ID", fill=(180, 180, 180))
    draw.text((50, 410), "DEMO-UPI-24036454", fill=(255, 255, 255))
    draw.text((50, 470), "From: Priya Patel", fill=(255, 255, 255))
    draw.text((50, 500), "priya@oksbi", fill=(200, 200, 200))
    draw.text((50, 560), "Google transaction ID", fill=(180, 180, 180))
    draw.text((50, 590), "DEMO-GPAY-63536981-406", fill=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def client():
    """Provides a fresh FastAPI test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def ocr_service():
    """Provides initialized OCR service."""
    reset_ocr_service()
    return get_ocr_service()


# ==============================================================================
# 1. OCR Service Unit Tests
# ==============================================================================


class TestOCRServiceUnit:
    """Unit tests for EasyOCRService inference, token geometries, and lifecycle."""

    def test_singleton_behavior(self):
        reset_ocr_service()
        s1 = get_ocr_service()
        s2 = get_ocr_service()
        assert s1 is s2
        assert s1.is_loaded is True

    def test_singleton_thread_safety(self):
        reset_ocr_service()
        instances = []

        def worker():
            instances.append(get_ocr_service())

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker) for _ in range(10)]
            for f in futures:
                f.result()

        assert len(instances) == 10
        first = instances[0]
        assert all(inst is first for inst in instances)

    def test_ocr_extraction_valid_png(self, ocr_service):
        img_bytes = create_test_receipt_image("PNG")
        res = ocr_service.extract(img_bytes)

        assert isinstance(res, OCRResult)
        assert res.ocr_engine == "EasyOCR"
        assert res.ocr_version == "1.7.2"
        assert len(res.raw_detections) > 0
        assert res.overall_confidence > 0.0
        assert res.processing_time_ms > 0.0
        assert "Google" in res.raw_text or "Pay" in res.raw_text

    def test_ocr_extraction_valid_jpeg(self, ocr_service):
        img_bytes = create_test_receipt_image("JPEG")
        res = ocr_service.extract(img_bytes)

        assert isinstance(res, OCRResult)
        assert len(res.raw_detections) > 0
        assert res.overall_confidence > 0.0

    def test_bounding_box_coordinates_validity(self, ocr_service):
        img_bytes = create_test_receipt_image("PNG")
        res = ocr_service.extract(img_bytes)

        for token in res.raw_detections:
            assert token.bounding_box is not None
            bbox = token.bounding_box
            # Check coordinate bounds in [0.0, 1.0]
            assert 0.0 <= bbox.xmin <= 1.0
            assert 0.0 <= bbox.ymin <= 1.0
            assert 0.0 <= bbox.xmax <= 1.0
            assert 0.0 <= bbox.ymax <= 1.0
            # Check order
            assert bbox.xmin <= bbox.xmax
            assert bbox.ymin <= bbox.ymax

            # Check pixel box
            pbox = token.pixel_box
            assert pbox is not None
            assert pbox.x >= 0.0
            assert pbox.y >= 0.0
            assert pbox.width >= 0.0
            assert pbox.height >= 0.0

    def test_confidence_range_validity(self, ocr_service):
        img_bytes = create_test_receipt_image("PNG")
        res = ocr_service.extract(img_bytes)

        assert 0.0 <= res.overall_confidence <= 1.0
        for token in res.raw_detections:
            assert token.confidence is not None
            assert 0.0 <= token.confidence <= 1.0

    def test_blank_image_truthful_empty_result(self, ocr_service):
        # Solid dark image with zero text
        blank = Image.new("RGB", (300, 300), color=(18, 18, 18))
        buf = io.BytesIO()
        blank.save(buf, format="PNG")

        res = ocr_service.extract(buf.getvalue())
        assert len(res.raw_detections) == 0
        assert res.overall_confidence == 0.0
        assert res.raw_text == ""
        # Check all structured fields are not detected
        assert res.fields.amount is None
        assert res.fields.status is None
        assert res.fields.transaction_id is None
        assert res.detailed_fields["amount"].is_detected is False
        assert res.detailed_fields["amount"].extraction_status == "not_found"

    def test_low_resolution_image_does_not_crash(self, ocr_service):
        low_res = Image.new("RGB", (60, 60), color=(18, 18, 18))
        buf = io.BytesIO()
        low_res.save(buf, format="PNG")

        res = ocr_service.extract(buf.getvalue())
        assert isinstance(res, OCRResult)

    def test_unsupported_image_bytes_raises_error(self, ocr_service):
        with pytest.raises(OCRImageError):
            ocr_service.extract(b"NOT_A_VALID_IMAGE_STREAM")

    def test_missing_model_weights_raises_initialization_error(self, tmp_path):
        with pytest.raises(OCRInitializationError):
            EasyOCRService(model_storage_dir=tmp_path / "nonexistent", auto_load=True)

    def test_inference_failure_raises_ocr_inference_error(self, ocr_service):
        # Simulate reader failure
        mock_reader = MagicMock()
        mock_reader.readtext.side_effect = RuntimeError("CUDA/MPS internal kernel failure")

        orig_reader = ocr_service.reader
        try:
            ocr_service.reader = mock_reader
            with pytest.raises(OCRInferenceError) as exc_info:
                ocr_service.extract(create_test_receipt_image("PNG"))
            assert "EasyOCR inference failure" in str(exc_info.value)
        finally:
            ocr_service.reader = orig_reader


# ==============================================================================
# 2. Field Extractor Unit Tests
# ==============================================================================


class TestTransactionFieldExtractor:
    """Tests conservative parsing logic across receipt fields."""

    def test_extract_fields_from_synthetic_receipt(self, ocr_service):
        img_bytes = create_test_receipt_image("PNG")
        res = ocr_service.extract(img_bytes)

        # Verify conservative extraction
        fields = res.fields
        assert fields.amount is not None
        assert fields.amount_numeric == 1250.0
        assert fields.status == "Completed"
        assert "Delight" in (fields.receiver or "")
        assert fields.sender == "Priya Patel"
        assert fields.sender_upi == "priya@oksbi"
        assert fields.date == "17 Apr 2025"
        assert "24036454" in (fields.transaction_id or "")
        assert "6353698" in (fields.google_txn_id or "")

    def test_no_fabricated_values_on_missing_fields(self):
        # Empty token stream
        fields, detailed = TransactionFieldExtractor.extract_fields(
            tokens=[], image_width=500, image_height=1000
        )
        assert fields.amount is None
        assert fields.amount_numeric is None
        assert fields.transaction_id is None
        assert fields.date is None
        assert fields.sender is None
        assert fields.receiver is None
        assert fields.status is None

        for k, field in detailed.items():
            assert field.is_detected is False
            assert field.raw_text is None
            assert field.extraction_status == "not_found"

    def test_dataset_clean_screenshot_extraction(self, ocr_service):
        """Verifies extraction on actual synthetic dataset image."""
        p = Path("Hackathon-VIT/CBL04-Dataset/data/synthetic_gpay_phase4/clean/P4-0001_clean.png")
        if not p.exists():
            pytest.skip("Dataset file not available on disk")

        with open(p, "rb") as f:
            res = ocr_service.extract(f.read())

        assert res.fields.amount is not None
        assert res.fields.amount_numeric == 2400.0
        assert res.fields.status == "Completed"
        assert res.fields.receiver == "Sara Rao"
        assert res.fields.receiver_upi == "8450@demo"
        assert res.fields.date == "17 Apr 2025"
        assert res.fields.time == "1.52am"
        assert res.fields.transaction_id == "DEMO-UPI-24036454"


# ==============================================================================
# 3. FastAPI API Integration Tests
# ==============================================================================


class TestOCRAPIIntegration:
    """Validates FastAPI /api/v1/analyze integration with live OCR."""

    def test_api_returns_live_ocr_component_status(self, client):
        img_bytes = create_test_receipt_image("PNG")
        resp = client.post(
            "/api/v1/analyze",
            files={"image": ("receipt_test.png", img_bytes, "image/png")},
            data={"request_id": "test-ocr-req-001"},
        )
        assert resp.status_code == 200
        data = resp.json()

        # Adherence to contracts
        assert data["pipeline_status"] in ["phase5_ocr_integrated", "phase6_semantic_integrated", "phase9_complete"]
        assert data["component_status"]["ocr"] == "live"
        assert data["component_status"]["cnn"] == "live"
        assert data["component_status"]["semantic"] in ["mock", "live"]
        assert data["component_status"]["stacker"] in ["mock", "live"]

        # Adherence to OCR schema
        ocr_data = data["ocr_data"]
        assert ocr_data["amount_numeric"] == 1250.0
        assert ocr_data["status"] == "Completed"
        assert "Delight" in (ocr_data["receiver"] or "")

        # Model provenance
        assert "EasyOCR" in data["model_versions"]["ocr_engine_version"]
        assert "phase" in data["model_versions"]["pipeline_version"]

    def test_cnn_verdict_remains_unaffected_by_ocr(self, client):
        """Verifies that verdict conforms to multi-branch final verdict engine contract."""
        img_bytes = create_test_receipt_image("PNG")
        resp = client.post(
            "/api/v1/analyze",
            files={"image": ("test_isolation.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()

        # In Phase 9+, verdict is generated by FinalVerdictService with tripartite guardrails
        assert data["verdict"] in [VerdictEnum.GENUINE.value, VerdictEnum.EDITED.value, VerdictEnum.INCONCLUSIVE.value, "GENUINE", "EDITED", "INCONCLUSIVE"]
        assert data["confidence"] is None or (0.0 <= data["confidence"] <= 1.0)
        assert 0.0 <= data["decision_score"] <= 1.0
        assert data["cnn_score"] is not None

    def test_ocr_unavailable_returns_503_error(self, client):
        """Ensures OCR initialization error returns truthful 503 instead of falling back to mock."""
        with patch("app.api.routes.analyze.get_ocr_service") as mock_get_service:
            mock_get_service.side_effect = OCRInitializationError("Models unavailable")

            img_bytes = create_test_receipt_image("PNG")
            resp = client.post(
                "/api/v1/analyze",
                files={"image": ("test_503.png", img_bytes, "image/png")},
            )
            assert resp.status_code == 503
            err = resp.json()
            assert err["error_code"] == ErrorCodeEnum.MODEL_UNAVAILABLE
            assert "OCR extraction service is currently unavailable" in err["message"]
