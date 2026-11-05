"""
Unit and integration tests for Phase 3 CNN Inference Service and API Integration.
Tests checkpoint loading, preprocessing, tensor shapes, forward inference,
error handling, singleton management, and end-to-end FastAPI endpoint integration.
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.schemas.cnn import CNNModelInput, CNNModelOutput
from app.schemas.enums import ErrorCodeEnum, VerdictEnum
from app.schemas.response import AnalyzeResponse
from app.services.cnn_inference_service import (
    CNNInferenceService,
    CNNModelError,
    CNNModelUnavailableError,
    get_cnn_service,
    reset_cnn_service,
)


def create_synthetic_image(
    format_name: str = "PNG",
    size: tuple[int, int] = (300, 500),
    color: str = "blue",
) -> bytes:
    """Helper generating synthetic image bytes for testing."""
    buf = io.BytesIO()
    mode = "RGB" if format_name.upper() in ["JPEG", "JPG"] else "RGBA"
    img = Image.new(mode, size, color)
    img.save(buf, format=format_name.upper())
    return buf.getvalue()


@pytest.fixture(autouse=True)
def clean_cnn_singleton():
    """Ensure singleton state is reset after each test."""
    yield
    reset_cnn_service()


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


class TestCNNInferenceServiceUnit:
    """Unit tests for CNNInferenceService class and preprocessing."""

    def test_cnn_service_singleton_behavior(self):
        """Verifies singleton pattern and reset functionality."""
        reset_cnn_service()
        svc1 = get_cnn_service()
        svc2 = get_cnn_service()
        assert svc1 is svc2

        reset_cnn_service()
        svc3 = get_cnn_service()
        assert svc1 is not svc3

    def test_checkpoint_loading_fold1(self):
        """Verifies default fold_1 checkpoint loads without errors."""
        service = CNNInferenceService(
            checkpoint_path="Hackathon-VIT/CBL04-Dataset/models/phase7/fold_1_resnet18_multitask.pt"
        )
        assert service.is_loaded is True
        assert service.model is not None
        assert service.weights_hash is not None
        assert len(service.weights_hash) == 16
        assert service.device.type in ["cpu", "mps"]

    def test_checkpoint_loading_weights_only_security(self):
        """Verifies checkpoint loads securely with weights_only=True."""
        service = CNNInferenceService(
            checkpoint_path="Hackathon-VIT/CBL04-Dataset/models/phase7/fold_1_resnet18_multitask.pt"
        )
        assert service.is_loaded is True
        assert service.full_weights_hash == "44b654aa846fddfde7be6dab44ceada6687d8dfd35e2c83edbcb60fed909f62a"

    def test_singleton_thread_safety(self):
        """Verifies concurrent threads obtain the same singleton instance without race conditions."""
        import concurrent.futures
        reset_cnn_service()
        instances = []

        def worker():
            return get_cnn_service()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker) for _ in range(10)]
            for f in concurrent.futures.as_completed(futures):
                instances.append(f.result())

        assert len(instances) == 10
        first = instances[0]
        assert all(inst is first for inst in instances)

    def test_checkpoint_loading_fold2(self):
        """Verifies fold_2 checkpoint loads without errors."""
        service = CNNInferenceService(
            checkpoint_path="Hackathon-VIT/CBL04-Dataset/models/phase7/fold_2_resnet18_multitask.pt"
        )
        assert service.is_loaded is True
        assert service.model is not None
        assert service.weights_hash is not None

    def test_checkpoint_loading_phase4_improved_candidate(self):
        """Verifies Phase 4 improved candidate checkpoint loads securely and runs inference."""
        improved_path = Path("Hackathon-VIT/CBL04-Dataset/models/phase9_improved/EXP1_MACRO_F1_fold_1_best.pt")
        if not improved_path.exists():
            pytest.skip("Phase 4 improved checkpoint not present")

        service = CNNInferenceService(checkpoint_path=improved_path)
        assert service.is_loaded is True
        assert service.model is not None

        img_bytes = create_synthetic_image("PNG", (320, 640))
        output = service.predict(CNNModelInput(image_bytes=img_bytes))
        assert 0.0 <= output.manipulation_probability <= 1.0
        assert output.score_type == "uncalibrated_model_score"
        assert output.localization_available is False

    def test_preprocessing_tensor_shape_and_normalization(self):
        """Verifies aspect-preserving resize produces [1, 3, 320, 160] tensor."""
        service = get_cnn_service()
        test_img = Image.new("RGB", (720, 1280), (100, 150, 200))
        tensor = service.preprocess(test_img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 320, 160)
        assert tensor.dtype == torch.float32
        assert tensor.device == service.device

    def test_predict_success_with_bytes(self):
        """Verifies successful CPU/MPS forward pass producing valid CNNModelOutput."""
        service = get_cnn_service()
        img_bytes = create_synthetic_image("PNG", (360, 640))
        input_data = CNNModelInput(image_bytes=img_bytes)

        output = service.predict(input_data)
        assert isinstance(output, CNNModelOutput)
        assert 0.0 <= output.manipulation_probability <= 1.0
        assert len(output.raw_logits) == 2
        assert output.score_type == "uncalibrated_model_score"
        assert output.localization_available is False
        assert output.suspicious_regions == []
        assert output.heatmap_data_uri is None
        assert output.inference_time_ms > 0.0
        assert "ResNet" in output.model_info.architecture

    def test_predict_with_unsupported_bytes_raises_error(self):
        """Verifies corrupted bytes raise CNNModelError."""
        service = get_cnn_service()
        bad_input = CNNModelInput(image_bytes=b"not an image stream")
        with pytest.raises(CNNModelError) as exc_info:
            service.predict(bad_input)
        assert "Could not decode" in str(exc_info.value)

    def test_predict_without_image_source_raises_error(self):
        """Verifies calling predict with no bytes or path raises CNNModelError."""
        service = get_cnn_service()
        empty_input = CNNModelInput()
        with pytest.raises(CNNModelError) as exc_info:
            service.predict(empty_input)
        assert "must supply either image_bytes or image_path" in str(exc_info.value)

    def test_missing_checkpoint_raises_unavailable_error(self):
        """Verifies pointing to nonexistent checkpoint raises CNNModelUnavailableError."""
        with pytest.raises(CNNModelUnavailableError) as exc_info:
            CNNInferenceService(checkpoint_path="nonexistent/path/model.pt")
        assert "checkpoint not found" in str(exc_info.value).lower()

    def test_corrupt_checkpoint_raises_error(self):
        """Verifies corrupt checkpoint file raises CNNModelUnavailableError."""
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            tmp.write(b"corrupted binary state dict header")
            tmp.flush()
            with pytest.raises(CNNModelUnavailableError):
                CNNInferenceService(checkpoint_path=tmp.name)


class TestCNNAPIIntegration:
    """Integration tests for FastAPI /api/v1/analyze with live CNN inference."""

    def test_analyze_endpoint_returns_live_cnn_score(self, client):
        """Verifies /api/v1/analyze executes live CNN inference and populates metadata."""
        img_bytes = create_synthetic_image("PNG", (400, 700))
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("test_gpay_receipt.png", img_bytes, "image/png")},
            data={"request_id": "test-live-cnn-001"},
        )
        assert response.status_code == 200
        data = response.json()

        # Validate structured schema adherence
        parsed = AnalyzeResponse.model_validate(data)
        assert parsed.request_id == "test-live-cnn-001"
        assert parsed.pipeline_status in ["phase3_cnn_integrated", "phase5_ocr_integrated", "phase6_semantic_integrated", "phase9_complete"]
        assert parsed.component_status["cnn"] == "live"
        assert parsed.component_status["ocr"] in ["mock", "live"]
        assert parsed.component_status["semantic"] in ["mock", "live"]
        assert parsed.component_status["stacker"] in ["mock", "live"]

        # Check live CNN score populated
        assert parsed.cnn_score is not None
        assert 0.0 <= parsed.cnn_score <= 1.0
        assert parsed.branch_scores.cnn_score == parsed.cnn_score

        # Check model version provenance
        assert "ResNet" in parsed.model_versions.cnn_model_version or "fold_" in parsed.model_versions.cnn_model_version

        # Check transparent warnings
        assert any("uncalibrated experimental model" in w or "Notice:" in w for w in parsed.warnings)
        assert any("IoU (0.059)" in w for w in parsed.warnings)

    def test_analyze_endpoint_verdict_styles(self, client):
        """Verifies canonical and frontend verdict formatting."""
        img_bytes = create_synthetic_image("PNG", (350, 650))

        # Canonical request
        resp_canonical = client.post(
            "/api/v1/analyze?verdict_style=canonical",
            files={"image": ("receipt.png", img_bytes, "image/png")},
        )
        assert resp_canonical.status_code == 200
        assert resp_canonical.json()["verdict"] in [
            VerdictEnum.GENUINE,
            VerdictEnum.EDITED,
            VerdictEnum.SUSPICIOUS,
            VerdictEnum.INCONCLUSIVE,
        ]

        # Frontend request
        resp_frontend = client.post(
            "/api/v1/analyze?verdict_style=frontend",
            files={"image": ("receipt.png", img_bytes, "image/png")},
        )
        assert resp_frontend.status_code == 200
        assert resp_frontend.json()["verdict"] in [
            VerdictEnum.REAL,
            VerdictEnum.FAKE,
            VerdictEnum.SUSPICIOUS,
            VerdictEnum.INCONCLUSIVE,
        ]

    def test_analyze_mock_mode_fallback(self, client, monkeypatch):
        """Verifies system cleanly falls back to pure mock pipeline when MOCK_MODE is enabled."""
        settings = get_settings()
        monkeypatch.setattr(settings, "MOCK_MODE", True)

        img_bytes = create_synthetic_image("PNG", (300, 500))
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("receipt.png", img_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["pipeline_status"] == "phase2_mock"
        assert data["component_status"]["cnn"] == "mock"

    def test_analyze_cnn_unavailable_returns_503(self, client, monkeypatch):
        """Verifies 503 error is returned when CNN model cannot be loaded."""
        def mock_get_service():
            raise CNNModelUnavailableError("Checkpoint file is missing from disk")

        monkeypatch.setattr("app.api.routes.analyze.get_cnn_service", mock_get_service)

        img_bytes = create_synthetic_image("PNG", (300, 500))
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("receipt.png", img_bytes, "image/png")},
        )
        assert response.status_code == 503
        data = response.json()
        assert data["error_code"] == ErrorCodeEnum.MODEL_UNAVAILABLE
        assert "unavailable" in data["message"].lower()
        # Verify no traceback leakage
        assert "Traceback" not in str(data)
        assert "File " not in str(data)

    def test_analyze_real_dataset_receipt_screenshot(self, client):
        """Verifies end-to-end API inference using an actual synthetic GPay receipt from the dataset."""
        receipt_path = Path("Hackathon-VIT/CBL04-Dataset/data/synthetic_gpay_phase4/edited/P4-0049_edited.png")
        if not receipt_path.exists():
            pytest.skip("Dataset sample image not present on disk")

        with open(receipt_path, "rb") as f:
            real_bytes = f.read()

        response = client.post(
            "/api/v1/analyze",
            files={"image": ("P4-0049_edited.png", real_bytes, "image/png")},
            data={"request_id": "test-real-dataset-receipt-001"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["pipeline_status"] in ["phase3_cnn_integrated", "phase5_ocr_integrated", "phase6_semantic_integrated", "phase9_complete"]
        assert data["component_status"]["cnn"] == "live"
        assert data["component_status"]["ocr"] in ["mock", "live"]
        assert data["cnn_score"] is not None
        assert 0.0 <= data["cnn_score"] <= 1.0
        assert data["verdict"] in [VerdictEnum.GENUINE, VerdictEnum.EDITED, VerdictEnum.INCONCLUSIVE]
        assert data["manipulation_analysis"]["detected"] is False
        assert data["manipulation_analysis"]["suspicious_regions"] == []

