"""
Comprehensive API integration test suite for CBL-04 FastAPI Foundation.
Tests health checks, image upload validation, mock pipeline inference,
error handling, and security sanitization using synthetic test images.
"""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import app
from app.schemas.enums import ErrorCodeEnum, VerdictEnum
from app.schemas.response import AnalyzeResponse


def create_synthetic_image(
    format_name: str = "PNG",
    size: tuple[int, int] = (300, 500),
    color: str = "blue",
) -> bytes:
    """Helper creating an in-memory synthetic image of specified format and dimensions."""
    buf = io.BytesIO()
    # JPEG does not support RGBA or alpha, use RGB
    mode = "RGB" if format_name.upper() in ["JPEG", "JPG"] else "RGBA"
    img = Image.new(mode, size, color)
    img.save(buf, format=format_name.upper())
    return buf.getvalue()


@pytest.fixture
def client():
    """Returns FastAPI TestClient instance."""
    return TestClient(app)


class TestHealthEndpoints:
    """Validates service health endpoints."""

    def test_health_endpoint_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "cbl04-gpay-detection-api"
        assert "version" in data

    def test_health_v1_alias(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestAnalyzeEndpointSuccess:
    """Validates successful image uploads across supported formats."""

    def test_analyze_accepts_valid_png(self, client):
        img_bytes = create_synthetic_image("PNG", (400, 600))
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("receipt.png", img_bytes, "image/png")},
            data={"request_id": "test-req-png-001"},
        )
        assert response.status_code == 200
        data = response.json()

        # Validate that response conforms to AnalyzeResponse
        parsed = AnalyzeResponse.model_validate(data)
        assert parsed.request_id == "test-req-png-001"
        assert parsed.verdict in [VerdictEnum.GENUINE, VerdictEnum.EDITED, VerdictEnum.SUSPICIOUS, VerdictEnum.INCONCLUSIVE]
        assert parsed.confidence is None or (0.0 <= parsed.confidence <= 1.0)
        assert 0.0 <= parsed.decision_score <= 1.0
        assert 0.0 <= parsed.manipulation_probability <= 1.0

    def test_analyze_accepts_valid_jpeg(self, client):
        img_bytes = create_synthetic_image("JPEG", (350, 700))
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("gpay_receipt.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        parsed = AnalyzeResponse.model_validate(response.json())
        assert parsed.ocr_data is not None

    def test_analyze_accepts_valid_webp(self, client):
        img_bytes = create_synthetic_image("WEBP", (400, 600))
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("gpay_receipt.webp", img_bytes, "image/webp")},
        )
        assert response.status_code == 200
        parsed = AnalyzeResponse.model_validate(response.json())
        assert parsed.manipulation_analysis is not None

    def test_analyze_without_prefix_alias(self, client):
        img_bytes = create_synthetic_image("PNG", (300, 500))
        response = client.post(
            "/analyze",
            files={"image": ("receipt.png", img_bytes, "image/png")},
        )
        assert response.status_code == 200

    def test_mock_pipeline_structures_populated(self, client):
        img_bytes = create_synthetic_image("PNG", (400, 800))
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("gpay_sample.png", img_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()

        # Check warning indicator
        assert any("Notice:" in w or "uncalibrated" in w or "MOCK PIPELINE" in w for w in data["warnings"])

        # Check OCR data
        assert "amount" in data["ocr_data"]
        assert "transaction_id" in data["ocr_data"]

        # Check branch scores
        assert data["branch_scores"]["cnn_score"] is not None
        assert data["branch_scores"]["semantic_score"] is not None
        assert data["branch_scores"]["stacker_score"] is not None

        # Check model versions metadata
        assert any(token in data["model_versions"]["pipeline_version"] for token in ["Mock", "phase", "v0.", "v1."])

    def test_frontend_verdict_style_override(self, client):
        img_bytes = create_synthetic_image("PNG", (400, 600))
        # Request with verdict_style=frontend
        response = client.post(
            "/api/v1/analyze?verdict_style=frontend",
            files={"image": ("receipt_real_sample.png", img_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["verdict"] in ["REAL", "FAKE", "SUSPICIOUS", "INCONCLUSIVE"]


class TestAnalyzeEndpointErrors:
    """Validates rejection of malformed, corrupt, or out-of-spec image uploads."""

    def test_missing_image_returns_400(self, client):
        # Sending form data without the required 'image' file
        response = client.post(
            "/api/v1/analyze",
            data={"request_id": "missing-file-test"},
        )
        assert response.status_code == 400
        data = response.json()
        assert data["error_code"] == ErrorCodeEnum.MISSING_REQUIRED_INPUT
        assert "image" in data["message"].lower()

    def test_empty_file_returns_400(self, client):
        empty_bytes = b""
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("empty.png", empty_bytes, "image/png")},
        )
        assert response.status_code == 400
        data = response.json()
        assert data["error_code"] == ErrorCodeEnum.CORRUPT_IMAGE
        assert "empty" in data["message"].lower()

    def test_corrupt_image_returns_400(self, client):
        corrupt_bytes = b"\x89PNG\r\n\x1a\nCorruptedImageDataNotDecodeable"
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("corrupted.png", corrupt_bytes, "image/png")},
        )
        assert response.status_code == 400
        data = response.json()
        assert data["error_code"] in [ErrorCodeEnum.CORRUPT_IMAGE, ErrorCodeEnum.IMAGE_UNREADABLE]

    def test_unsupported_format_extension_returns_415(self, client):
        pdf_bytes = b"%PDF-1.4 Fake PDF document bytes"
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("receipt.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 415
        data = response.json()
        assert data["error_code"] == ErrorCodeEnum.INVALID_IMAGE_FORMAT

    def test_unsupported_image_type_gif_returns_415(self, client):
        # Create a real GIF image (valid image file, but disallowed by GPay contract)
        gif_buf = io.BytesIO()
        img = Image.new("RGB", (200, 200), "red")
        img.save(gif_buf, format="GIF")
        gif_bytes = gif_buf.getvalue()

        response = client.post(
            "/api/v1/analyze",
            files={"image": ("receipt.gif", gif_bytes, "image/gif")},
        )
        assert response.status_code == 415
        data = response.json()
        assert data["error_code"] == ErrorCodeEnum.INVALID_IMAGE_FORMAT

    def test_dimensions_too_small_returns_422(self, client):
        # 50x50 is below min 100x100
        tiny_bytes = create_synthetic_image("PNG", (50, 50))
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("tiny.png", tiny_bytes, "image/png")},
        )
        assert response.status_code == 422
        data = response.json()
        assert data["error_code"] == ErrorCodeEnum.IMAGE_DIMENSIONS_OUT_OF_RANGE
        assert "small" in data["message"].lower()

    def test_dimensions_too_large_returns_422(self, client):
        # 4200x500 is above max 4096px
        huge_bytes = create_synthetic_image("PNG", (4200, 500))
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("huge.png", huge_bytes, "image/png")},
        )
        assert response.status_code == 422
        data = response.json()
        assert data["error_code"] == ErrorCodeEnum.IMAGE_DIMENSIONS_OUT_OF_RANGE
        assert "exceed" in data["message"].lower()

    def test_oversized_upload_returns_413(self, client, monkeypatch):
        # Set max upload size to 500 bytes for testing limit enforcement
        settings = get_settings()
        monkeypatch.setattr(settings, "MAX_FILE_SIZE_BYTES", 500)

        # Create image larger than 500 bytes
        oversized_bytes = create_synthetic_image("PNG", (300, 300))
        assert len(oversized_bytes) > 500

        response = client.post(
            "/api/v1/analyze",
            files={"image": ("large.png", oversized_bytes, "image/png")},
        )
        assert response.status_code == 413
        data = response.json()
        assert data["error_code"] == ErrorCodeEnum.FILE_TOO_LARGE

    def test_errors_do_not_expose_stack_traces_or_paths(self, client):
        corrupt_bytes = b"Not an image at all"
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("bad.png", corrupt_bytes, "image/png")},
        )
        data = response.json()
        json_str = str(data)

        # Check for path leakage
        assert "C:\\" not in json_str
        assert "/home/" not in json_str
        assert "Traceback" not in json_str
        assert "File " not in json_str
