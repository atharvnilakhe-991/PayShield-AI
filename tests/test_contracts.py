"""
Contract validation test suite for CBL-04 Google Pay Screenshot Manipulation Detection.
Validates Pydantic schema constraints, coordinate transformations, error sanitization,
branch interfaces, and compatibility with Member 4 frontend mock fixtures.
"""

import json
import pytest
from pydantic import ValidationError

from app.schemas import (
    VerdictEnum,
    ReasonCodeEnum,
    ErrorCodeEnum,
    REASON_CODE_DESCRIPTIONS,
    NormalizedBoundingBox,
    PixelBoundingBox,
    ModelVersionInfo,
    AnalyzeRequest,
    ImageUploadConfig,
    ClientMetadata,
    validate_image_metadata,
    OCRExtractedField,
    ExtractedTransactionData,
    OCRResult,
    CNNSuspiciousRegion,
    CNNModelInput,
    CNNModelOutput,
    CNNInferenceInterface,
    SemanticFeatureVector,
    FieldAnomalyIndicators,
    OCRConfidenceInfo,
    SemanticBranchOutput,
    CalibrationMetadata,
    StackerInput,
    StackerOutput,
    StackerInferenceInterface,
    SuspiciousRegion,
    ManipulationAnalysisEvidence,
    BranchScores,
    ModelVersionMetadata,
    AnalyzeResponse,
    APIErrorResponse,
    create_error_response,
    sanitize_error_detail,
)


class TestEnums:
    """Validates verdicts, reason codes, error codes, and canonical mapping."""

    def test_verdict_values(self):
        assert VerdictEnum.GENUINE == "GENUINE"
        assert VerdictEnum.EDITED == "EDITED"
        assert VerdictEnum.SUSPICIOUS == "SUSPICIOUS"
        assert VerdictEnum.INCONCLUSIVE == "INCONCLUSIVE"
        assert VerdictEnum.REAL == "REAL"
        assert VerdictEnum.FAKE == "FAKE"

    def test_verdict_canonical_mapping(self):
        assert VerdictEnum.to_canonical(VerdictEnum.REAL) == VerdictEnum.GENUINE
        assert VerdictEnum.to_canonical("REAL") == VerdictEnum.GENUINE
        assert VerdictEnum.to_canonical(VerdictEnum.FAKE) == VerdictEnum.EDITED
        assert VerdictEnum.to_canonical("fake") == VerdictEnum.EDITED
        assert VerdictEnum.to_canonical(VerdictEnum.GENUINE) == VerdictEnum.GENUINE
        assert VerdictEnum.to_canonical(VerdictEnum.EDITED) == VerdictEnum.EDITED
        assert VerdictEnum.to_canonical(VerdictEnum.SUSPICIOUS) == VerdictEnum.SUSPICIOUS
        assert VerdictEnum.to_canonical("unknown_str") == VerdictEnum.INCONCLUSIVE

    def test_reason_codes_documented(self):
        for code in ReasonCodeEnum:
            assert code in REASON_CODE_DESCRIPTIONS
            desc = REASON_CODE_DESCRIPTIONS[code]
            assert isinstance(desc, str) and len(desc) > 10

    def test_error_codes_exist(self):
        assert ErrorCodeEnum.INVALID_IMAGE_FORMAT == "INVALID_IMAGE_FORMAT"
        assert ErrorCodeEnum.FILE_TOO_LARGE == "FILE_TOO_LARGE"
        assert ErrorCodeEnum.CORRUPT_IMAGE == "CORRUPT_IMAGE"
        assert ErrorCodeEnum.MODEL_UNAVAILABLE == "MODEL_UNAVAILABLE"


class TestBoundingBoxes:
    """Validates normalized coordinates, pixel boxes, and geometric conversions."""

    def test_valid_normalized_box(self):
        box = NormalizedBoundingBox(xmin=0.1, ymin=0.2, xmax=0.8, ymax=0.9)
        assert box.width == pytest.approx(0.7)
        assert box.height == pytest.approx(0.7)

    def test_invalid_normalized_box_out_of_bounds(self):
        with pytest.raises(ValidationError):
            NormalizedBoundingBox(xmin=-0.1, ymin=0.0, xmax=0.5, ymax=0.5)

        with pytest.raises(ValidationError):
            NormalizedBoundingBox(xmin=0.0, ymin=0.0, xmax=1.1, ymax=0.5)

    def test_invalid_normalized_box_inverted_coordinates(self):
        with pytest.raises(ValidationError):
            NormalizedBoundingBox(xmin=0.8, ymin=0.2, xmax=0.2, ymax=0.9)

        with pytest.raises(ValidationError):
            NormalizedBoundingBox(xmin=0.1, ymin=0.9, xmax=0.8, ymax=0.2)

    def test_normalized_to_pixel_conversion(self):
        box = NormalizedBoundingBox(xmin=0.1, ymin=0.2, xmax=0.5, ymax=0.6)
        pixel_box = box.to_pixel(image_width=1000, image_height=2000)

        assert pixel_box.x == 100.0
        assert pixel_box.y == 400.0
        assert pixel_box.width == 400.0
        assert pixel_box.height == 800.0
        assert pixel_box.area == 320000.0

    def test_pixel_to_normalized_conversion(self):
        pixel_box = PixelBoundingBox(x=100, y=200, width=400, height=800)
        norm_box = pixel_box.to_normalized(image_width=1000, image_height=2000)

        assert norm_box.xmin == pytest.approx(0.1)
        assert norm_box.ymin == pytest.approx(0.1)
        assert norm_box.xmax == pytest.approx(0.5)
        assert norm_box.ymax == pytest.approx(0.5)


class TestRequestValidation:
    """Validates analyze request contracts and image upload constraints."""

    def test_analyze_request_sanitizes_id(self):
        req = AnalyzeRequest(request_id="   trace-abc-123   ", return_heatmap=True)
        assert req.request_id == "trace-abc-123"

        req_empty = AnalyzeRequest(request_id="   ")
        assert req_empty.request_id is None

    def test_image_upload_config_defaults(self):
        cfg = ImageUploadConfig()
        assert "image/png" in cfg.supported_mime_types
        assert "image/jpeg" in cfg.supported_mime_types
        assert "image/webp" in cfg.supported_mime_types
        assert cfg.max_file_size_bytes == 10 * 1024 * 1024

    def test_validate_image_metadata_success(self):
        is_valid, err_code, msg = validate_image_metadata(
            filename="gpay_receipt.png",
            content_type="image/png",
            file_size_bytes=500_000,
        )
        assert is_valid is True
        assert err_code is None
        assert msg is None

    def test_validate_image_metadata_empty_file(self):
        is_valid, err_code, msg = validate_image_metadata(
            filename="gpay_receipt.jpg",
            content_type="image/jpeg",
            file_size_bytes=0,
        )
        assert is_valid is False
        assert err_code == ErrorCodeEnum.CORRUPT_IMAGE
        assert "empty" in msg.lower()

    def test_validate_image_metadata_too_large(self):
        is_valid, err_code, msg = validate_image_metadata(
            filename="large_receipt.png",
            content_type="image/png",
            file_size_bytes=15 * 1024 * 1024,
        )
        assert is_valid is False
        assert err_code == ErrorCodeEnum.FILE_TOO_LARGE

    def test_validate_image_metadata_unsupported_type(self):
        is_valid, err_code, msg = validate_image_metadata(
            filename="document.pdf",
            content_type="application/pdf",
            file_size_bytes=100_000,
        )
        assert is_valid is False
        assert err_code == ErrorCodeEnum.INVALID_IMAGE_FORMAT


class TestOCRSchema:
    """Validates OCR extraction schema, missing field tolerance, and normalization."""

    def test_extracted_transaction_data_all_null(self):
        data = ExtractedTransactionData()
        assert data.amount is None
        assert data.transaction_id is None
        assert data.sender is None
        assert data.receiver is None
        assert data.currency == "INR"

    def test_extracted_field_with_normalization(self):
        field = OCRExtractedField(
            field_name="amount",
            raw_text="₹ 1,500.00",
            confidence=0.965,
            bounding_box=NormalizedBoundingBox(xmin=0.2, ymin=0.3, xmax=0.8, ymax=0.4),
            normalized_value=1500.0,
            is_detected=True,
        )
        assert field.confidence == 0.965
        assert field.normalized_value == 1500.0
        assert field.is_detected is True

    def test_ocr_result_serialization(self):
        res = OCRResult(
            fields=ExtractedTransactionData(
                transaction_id="430829103842",
                amount="₹1,500.00",
                amount_numeric=1500.0,
                date="2026-11-03",
                receiver="Merchant Mart",
                status="Paid",
            ),
            overall_confidence=0.94,
            ocr_engine="EasyOCR",
            ocr_version="1.7.2",
            raw_text="Paid ₹1,500.00 to Merchant Mart UPI Ref: 430829103842",
        )
        dump = res.model_dump()
        assert dump["fields"]["amount_numeric"] == 1500.0
        assert dump["overall_confidence"] == 0.94


class TestCNNInterface:
    """Validates CNN interface schemas and abstract class contract."""

    def test_cnn_model_output_validation(self):
        output = CNNModelOutput(
            manipulation_probability=0.88,
            heatmap_relative_path="static/heatmaps/demo-123.png",
            suspicious_regions=[
                CNNSuspiciousRegion(
                    label="Amount tampering",
                    score=0.92,
                    bbox_normalized=NormalizedBoundingBox(xmin=0.1, ymin=0.2, xmax=0.4, ymax=0.3),
                    anomaly_type="splicing",
                )
            ],
            model_info=ModelVersionInfo(
                model_name="EfficientNet-B4",
                model_version="v1.0.0",
                architecture="efficientnet_b4_ela",
            ),
            inference_time_ms=45.2,
        )
        assert output.manipulation_probability == 0.88
        assert len(output.suspicious_regions) == 1
        assert output.suspicious_regions[0].anomaly_type == "splicing"

    def test_cnn_abstract_interface_enforcement(self):
        class DummyCNN(CNNInferenceInterface):
            def predict(self, input_data: CNNModelInput) -> CNNModelOutput:
                return CNNModelOutput(
                    manipulation_probability=0.1,
                    model_info=ModelVersionInfo(model_name="Dummy", model_version="v0.1"),
                )

            def get_model_info(self) -> ModelVersionInfo:
                return ModelVersionInfo(model_name="Dummy", model_version="v0.1")

        dummy = DummyCNN()
        out = dummy.predict(CNNModelInput(image_dimensions=(1080, 1920)))
        assert out.manipulation_probability == 0.1


class TestSemanticContract:
    """Validates semantic engine outputs, reason codes, and anomaly indicators."""

    def test_semantic_branch_output(self):
        sem = SemanticBranchOutput(
            semantic_manipulation_score=0.85,
            feature_vector=SemanticFeatureVector(
                amount_valid_format=False,
                ref_id_length_valid=False,
                font_size_ratio_amount_to_payee=0.8,
            ),
            reason_codes=[
                ReasonCodeEnum.AMOUNT_FORMAT_ANOMALY,
                ReasonCodeEnum.REF_ID_FORMAT_ANOMALY,
            ],
            field_anomalies=FieldAnomalyIndicators(amount_anomaly=True, ref_id_anomaly=True),
            ocr_confidence_info=OCRConfidenceInfo(
                average_confidence=0.91,
                min_field_confidence=0.68,
                critical_fields_confidence=0.88,
                total_detected_tokens=24,
            ),
        )
        assert sem.semantic_manipulation_score == 0.85
        assert ReasonCodeEnum.AMOUNT_FORMAT_ANOMALY in sem.reason_codes
        assert sem.field_anomalies.amount_anomaly is True


class TestStackerContract:
    """Validates OOF stacker input, output, and calibration metadata."""

    def test_stacker_payloads(self):
        inp = StackerInput(
            cnn_score=0.82,
            semantic_score=0.91,
            cnn_model_version="cnn_v1.0",
            semantic_model_version="semantic_v1.0",
        )
        assert inp.cnn_score == 0.82

        out = StackerOutput(
            final_manipulation_probability=0.89,
            calibrated_confidence=0.93,
            stacker_model_version="lgbm_oof_v1.0",
            decision_metadata=CalibrationMetadata(
                calibration_method="isotonic_regression",
                cnn_weight=0.45,
                semantic_weight=0.55,
                uncertainty_score=0.08,
                decision_threshold=0.65,
            ),
        )
        assert out.final_manipulation_probability == 0.89
        assert out.decision_metadata.calibration_method == "isotonic_regression"


class TestResponseContract:
    """Validates final API response serialization, disclaimer, and frontend compatibility."""

    def test_genuine_response_fixture(self):
        resp = AnalyzeResponse(
            request_id="demo-real-001",
            verdict=VerdictEnum.GENUINE,
            confidence=0.98,
            manipulation_probability=0.02,
            message="Analysis completed. Receipt integrity verified; no manipulation detected.",
            ocr_data=ExtractedTransactionData(
                transaction_id="CIC492049203",
                amount="₹350",
                date="2026-11-03",
                sender="Priya Patel",
                receiver="Cafe Delight",
            ),
            manipulation_analysis=ManipulationAnalysisEvidence(
                detected=False,
                suspicious_regions=[],
                heatmap_url=None,
            ),
            branch_scores=BranchScores(
                cnn_score=0.03,
                semantic_score=0.01,
                stacker_score=0.02,
            ),
            reason_codes=[ReasonCodeEnum.INTEGRITY_VERIFIED_CLEAN],
        )
        # Check automatic top-level branch score sync
        assert resp.cnn_score == 0.03
        assert resp.semantic_score == 0.01
        assert resp.stacker_score == 0.02
        assert resp.verdict == VerdictEnum.GENUINE
        assert "not verify bank-side settlement" in resp.disclaimer

        # Ensure valid JSON serialization
        json_str = resp.model_dump_json()
        data = json.loads(json_str)
        assert data["ocr_data"]["amount"] == "₹350"
        assert data["manipulation_analysis"]["detected"] is False

    def test_edited_response_fixture(self):
        resp = AnalyzeResponse(
            request_id="demo-fake-001",
            verdict=VerdictEnum.EDITED,
            confidence=0.95,
            manipulation_probability=0.94,
            message="Analysis completed. Image exhibits localized tampering around amount and font inconsistencies.",
            ocr_data=ExtractedTransactionData(
                transaction_id="UPI/234809283402",
                amount="₹5,000",
                date="2026-11-03",
                sender="Rahul Sharma",
                receiver="Merchant Store",
            ),
            manipulation_analysis=ManipulationAnalysisEvidence(
                detected=True,
                suspicious_regions=[
                    SuspiciousRegion(
                        label="Amount field",
                        x=100.0,
                        y=200.0,
                        width=140.0,
                        height=48.0,
                        score=0.94,
                        box_normalized=NormalizedBoundingBox(xmin=0.1, ymin=0.2, xmax=0.24, ymax=0.25),
                    )
                ],
                heatmap_url=None,
            ),
            branch_scores=BranchScores(
                cnn_score=0.95,
                semantic_score=0.92,
                stacker_score=0.94,
            ),
            reason_codes=[
                ReasonCodeEnum.AMOUNT_FORMAT_ANOMALY,
                ReasonCodeEnum.TEXT_LAYOUT_INCONSISTENCY,
            ],
        )
        assert resp.verdict == VerdictEnum.EDITED
        assert len(resp.manipulation_analysis.suspicious_regions) == 1
        assert resp.manipulation_analysis.suspicious_regions[0].x == 100.0

    def test_response_supports_frontend_real_fake_alias(self):
        # Frontend might receive REAL / FAKE
        resp = AnalyzeResponse(
            request_id="demo-alias",
            verdict=VerdictEnum.REAL,
            confidence=0.9,
            manipulation_probability=0.1,
            message="OK",
            manipulation_analysis=ManipulationAnalysisEvidence(detected=False),
        )
        assert resp.verdict == VerdictEnum.REAL
        assert VerdictEnum.to_canonical(resp.verdict) == VerdictEnum.GENUINE


class TestErrorContract:
    """Validates API error responses, status codes, and security sanitization."""

    def test_error_response_creation(self):
        err = create_error_response(
            error_code=ErrorCodeEnum.FILE_TOO_LARGE,
            message="Image file size exceeds maximum limit of 10 MB.",
            status_code=413,
            request_id="req-999",
        )
        assert err.error_code == ErrorCodeEnum.FILE_TOO_LARGE
        assert err.status_code == 413
        assert err.request_id == "req-999"

    def test_error_sanitization_removes_internal_paths(self):
        leaked_windows_path = r"Error reading file C:\Users\athar\app\models\weights.bin: internal bug"
        leaked_unix_path = r"Failure in /home/ubuntu/app/server.py line 45"
        trace_str = 'Traceback: File "app/server.py", line 42 in run'

        sanitized_win = sanitize_error_detail(leaked_windows_path)
        assert r"C:\Users\athar" not in sanitized_win
        assert "[REDACTED_PATH]" in sanitized_win

        sanitized_unix = sanitize_error_detail(leaked_unix_path)
        assert "/home/ubuntu" not in sanitized_unix

        sanitized_trace = sanitize_error_detail(trace_str)
        assert 'line 42' not in sanitized_trace

