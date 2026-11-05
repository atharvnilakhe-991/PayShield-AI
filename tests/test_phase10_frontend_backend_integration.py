"""
Phase 10: Live Frontend-Backend Integration & Contract Tests.
Validates end-to-end integration contracts between the React/Vite frontend and FastAPI backend,
including endpoint routes, multipart field names, canonical verdicts, truthful localization suppression,
OCR metadata extraction, multi-branch score breakdowns, and error responses.
"""

import io
import os
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.schemas.enums import ErrorCodeEnum, VerdictEnum
from app.schemas.response import AnalyzeResponse


@pytest.fixture
def client():
    """TestClient instance for FastAPI application."""
    return TestClient(app)


@pytest.fixture
def clean_screenshot_path():
    """Path to an actual synthetic clean GPay dark-theme transaction screenshot."""
    path = "Hackathon-VIT/CBL04-Dataset/data/synthetic_gpay_phase4/clean/P4-0001_clean.png"
    if not os.path.exists(path):
        pytest.skip(f"Test screenshot not found: {path}")
    return path


@pytest.fixture
def edited_screenshot_path():
    """Path to an actual synthetic edited GPay dark-theme transaction screenshot."""
    path = "Hackathon-VIT/CBL04-Dataset/data/synthetic_gpay_phase4/edited/P4-0001_edited.png"
    if not os.path.exists(path):
        pytest.skip(f"Test screenshot not found: {path}")
    return path


class TestPhase10HealthCheck:
    """Verifies backend liveness and version contracts used by frontend App.jsx connection tester."""

    def test_health_v1_endpoint(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "version" in data

    def test_health_root_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestPhase10RealAnalyzeEndpoint:
    """Verifies live analyze endpoint behavior against frontend component expectations."""

    def test_clean_screenshot_live_execution(self, client, clean_screenshot_path):
        with open(clean_screenshot_path, "rb") as f:
            response = client.post(
                "/api/v1/analyze",
                files={"image": ("clean_sample.png", f, "image/png")},
            )

        assert response.status_code == 200
        data = response.json()
        parsed = AnalyzeResponse.model_validate(data)

        # 1. Canonical Verdict Validation
        assert parsed.verdict in [
            VerdictEnum.GENUINE,
            VerdictEnum.EDITED,
            VerdictEnum.INCONCLUSIVE,
        ]

        # 2. Confidence & Uncalibrated Scores
        assert parsed.confidence is None or (0.0 <= parsed.confidence <= 1.0)
        assert 0.0 <= parsed.decision_score <= 1.0
        assert 0.0 <= parsed.manipulation_probability <= 1.0

        # 3. OCR Fields Expected by OCRDataCard.jsx
        ocr = parsed.ocr_data
        assert hasattr(ocr, "transaction_id")
        assert hasattr(ocr, "amount")
        assert hasattr(ocr, "date")
        assert hasattr(ocr, "sender")
        assert hasattr(ocr, "receiver")

        # 4. Truthful Localization Suppression (HeatmapView.jsx)
        # Heatmap URL must be None (no fabricated or hallucinated heatmaps)
        assert parsed.manipulation_analysis.heatmap_url is None

        # 5. Multi-Branch Evidence (ResultsDashboard.jsx)
        assert parsed.branch_scores.cnn_score is not None
        assert parsed.branch_scores.semantic_score is not None
        assert parsed.branch_scores.stacker_score is not None

        # 6. Pipeline & Component Provenance
        assert parsed.pipeline_status == "phase9_complete"
        assert parsed.component_status["cnn"] == "live"
        assert parsed.component_status["ocr"] == "live"
        assert parsed.component_status["semantic"] == "live"
        assert parsed.component_status["stacker"] == "live"
        assert parsed.component_status["verdict_engine"] == "live"

        # 7. Settlement Disclaimer
        assert "bank-side settlement" in parsed.disclaimer.lower()

    def test_edited_screenshot_live_execution(self, client, edited_screenshot_path):
        with open(edited_screenshot_path, "rb") as f:
            response = client.post(
                "/api/v1/analyze",
                files={"image": ("edited_sample.png", f, "image/png")},
            )

        assert response.status_code == 200
        data = response.json()
        parsed = AnalyzeResponse.model_validate(data)
        assert parsed.verdict in [
            VerdictEnum.GENUINE,
            VerdictEnum.EDITED,
            VerdictEnum.INCONCLUSIVE,
        ]
        assert parsed.manipulation_analysis.heatmap_url is None


class TestPhase10ErrorHandlingAndBoundaries:
    """Verifies that API error responses conform to frontend formatApiError expectations."""

    def test_missing_image_parameter_returns_400(self, client):
        response = client.post("/api/v1/analyze")
        assert response.status_code == 400
        data = response.json()
        assert data["error_code"] == ErrorCodeEnum.MISSING_REQUIRED_INPUT.value
        assert "image" in data["message"].lower()

    def test_unsupported_media_type_returns_415(self, client):
        text_content = b"This is not a receipt image."
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("test.txt", io.BytesIO(text_content), "text/plain")},
        )
        assert response.status_code == 415
        data = response.json()
        assert data["error_code"] == ErrorCodeEnum.INVALID_IMAGE_FORMAT.value

    def test_empty_file_returns_400(self, client):
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("empty.png", io.BytesIO(b""), "image/png")},
        )
        assert response.status_code == 400
        data = response.json()
        assert data["error_code"] == ErrorCodeEnum.CORRUPT_IMAGE.value

