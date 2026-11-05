"""
Test Suite for CBL-04 Phase 14: OCR Accuracy Remediation, Ambiguity Guardrails & Downstream Integrity.
Validates:
1. Currency symbol prefix handling (₹, Rs, INR, {, [, ?, etc.).
2. The "₹1 vs ₹21" safeguard: never silently strip digits without explicit structural evidence.
3. Phone number, transaction ID, and bank account suffix rejection.
4. Ambiguity reporting and candidate telemetry attachment.
5. Downstream semantic feature engine behavior: ambiguous extractions do not trigger false positive anomalies.
6. Downstream final verdict engine: ambiguous OCR does not flip authentic screenshots to EDITED.
7. Backward compatibility of OCR response schemas.
"""

import io
import pytest
from PIL import Image, ImageDraw

from app.schemas.common import NormalizedBoundingBox, PixelBoundingBox
from app.schemas.enums import ReasonCodeEnum, VerdictEnum
from app.schemas.ocr import ExtractedTransactionData, OCRExtractedField, OCRResult
from app.schemas.semantic import SemanticBranchOutput
from app.services.final_verdict_service import FinalVerdictService
from app.services.ocr_service import (
    EasyOCRService,
    TransactionFieldExtractor,
    get_ocr_service,
    reset_ocr_service,
)
from app.services.semantic_feature_service import SemanticFeatureService


@pytest.fixture
def ocr_service():
    """Provides singleton OCR service."""
    reset_ocr_service()
    return get_ocr_service()


@pytest.fixture
def semantic_service():
    """Provides fresh semantic feature engine."""
    return SemanticFeatureService()


@pytest.fixture
def verdict_service():
    """Provides fresh final verdict engine."""
    return FinalVerdictService()


def _make_token(
    text: str,
    ymin: float = 0.22,
    ymax: float = 0.28,
    xmin: float = 0.35,
    xmax: float = 0.65,
    height_px: float = 75.0,
    conf: float = 0.95,
) -> OCRExtractedField:
    """Helper to construct synthetic OCR token with exact coordinates."""
    actual_ymax = max(ymax, ymin + 0.04)
    return OCRExtractedField(
        field_name="token_test",
        raw_text=text,
        confidence=conf,
        bounding_box=NormalizedBoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=actual_ymax),
        pixel_box=PixelBoundingBox(x=100.0, y=100.0, width=200.0, height=height_px),
        normalized_value=text,
        is_detected=True,
        extraction_status="extracted",
    )


class TestPhase14AmountExtraction:
    """Tests for targeted amount candidate extraction, parsing, and ranking."""

    def test_currency_symbol_prefix_variations(self):
        """Standard and misrecognized currency prefixes must extract correctly with confirmed status."""
        test_cases = [
            ("₹1,250.00", 1250.0),
            ("{999", 999.0),
            ("[450", 450.0),
            ("?511", 511.0),
            ("Rs. 2,400", 2400.0),
            ("INR 500", 500.0),
            ("§750", 750.0),
            ("Ft 300", 300.0),
        ]
        for raw, expected_val in test_cases:
            tok = _make_token(raw, ymin=0.22, height_px=75.0)
            field, val = TransactionFieldExtractor._extract_amount([tok], image_height=1280)
            assert field.is_detected is True, f"Failed detection on {raw}"
            assert field.extraction_status == "extracted", f"Expected 'extracted' on {raw}, got {field.extraction_status}"
            assert val == pytest.approx(expected_val), f"Expected {expected_val} on {raw}, got {val}"
            assert field.raw_text == raw

    def test_rupee_1_vs_21_safeguard_never_strip_digits(self):
        """
        Rule 2 compliance: Pure digit strings without explicit currency symbols
        must NEVER have digits silently stripped, preserving literal value with ambiguous status.
        """
        # Scenario A: Genuine transaction of ₹21 read by OCR as '21' without symbol
        tok_21 = _make_token("21", ymin=0.22, height_px=75.0)
        field_21, val_21 = TransactionFieldExtractor._extract_amount([tok_21], image_height=1280)
        assert field_21.is_detected is True
        assert field_21.extraction_status == "ambiguous"
        assert val_21 == 21.0, f"Expected literal 21.0, but digits were stripped to {val_21}"
        assert field_21.raw_text == "21"

        # Scenario B: Genuine transaction of ₹3,450 read as '3450' without symbol
        tok_3450 = _make_token("3450", ymin=0.22, height_px=75.0)
        field_3450, val_3450 = TransactionFieldExtractor._extract_amount([tok_3450], image_height=1280)
        assert field_3450.is_detected is True
        assert field_3450.extraction_status == "ambiguous"
        assert val_3450 == 3450.0, f"Expected literal 3450.0, but digits were stripped to {val_3450}"
        assert field_3450.raw_text == "3450"

        # Scenario C: Candidate telemetry records the hypothesized alternative without fabricating primary value
        assert field_3450.candidates is not None
        hypothesized = [c for c in field_3450.candidates if "hypothesized_amount_without_prefix_digit" in c]
        assert len(hypothesized) > 0
        assert hypothesized[0]["hypothesized_amount_without_prefix_digit"] == 450.0
        assert hypothesized[0]["hypothesized_prefix_glyph"] == "3"

    def test_phone_number_exclusion(self):
        """Tokens representing recipient phone numbers or fragments must be excluded."""
        tokens = [
            _make_token("+91 9876543210", ymin=0.15, height_px=22.0),
            _make_token("'+91'", ymin=0.15, height_px=22.0),
            _make_token("'89", ymin=0.16, height_px=20.0),
            _make_token("9876543210", ymin=0.16, height_px=22.0),
            _make_token("{1,500.00", ymin=0.22, height_px=78.0),
        ]
        field, val = TransactionFieldExtractor._extract_amount(tokens, image_height=1280)
        assert field.is_detected is True
        assert field.extraction_status == "extracted"
        assert val == 1500.0
        assert field.raw_text == "{1,500.00}" or field.raw_text == "{1,500.00"

    def test_transaction_id_and_account_suffix_exclusion(self):
        """12-digit RRNs, transaction ID strings, and small bank account numbers must be excluded."""
        tokens = [
            _make_token("DEMO-UPI-13649734", ymin=0.55, height_px=22.0),
            _make_token("123456789012", ymin=0.56, height_px=20.0),
            _make_token("6429", ymin=0.45, height_px=24.0),  # Bank account ending
            _make_token("₹2,400.00", ymin=0.22, height_px=76.0),
        ]
        field, val = TransactionFieldExtractor._extract_amount(tokens, image_height=1280)
        assert field.is_detected is True
        assert val == 2400.0
        assert field.raw_text == "₹2,400.00"

    def test_adjacent_currency_token_linking(self):
        """When the currency symbol is localized as a separate token immediately to the left, link it."""
        sym_tok = _make_token("₹", ymin=0.22, ymax=0.28, xmin=0.30, xmax=0.34, height_px=70.0)
        num_tok = _make_token("2,500", ymin=0.22, ymax=0.28, xmin=0.35, xmax=0.60, height_px=75.0)

        field, val = TransactionFieldExtractor._extract_amount([sym_tok, num_tok], image_height=1280)
        assert field.is_detected is True
        assert field.extraction_status == "extracted"
        assert val == 2500.0

    def test_empty_tokens_returns_not_found(self):
        """When no amount tokens exist, returns truthful not_found state without fabrication."""
        field, val = TransactionFieldExtractor._extract_amount([], image_height=1280)
        assert field.is_detected is False
        assert field.extraction_status == "not_found"
        assert val is None
        assert field.raw_text is None


class TestPhase14DownstreamIntegration:
    """Tests downstream interaction between ambiguous OCR extractions and Semantic / Verdict engines."""

    def test_semantic_service_no_false_positive_on_ambiguous_amount(self, semantic_service):
        """
        When amount extraction status is 'ambiguous', semantic engine must record
        feature_status='ambiguous' without asserting amount_anomaly=True.
        """
        amt_field = OCRExtractedField(
            field_name="amount",
            raw_text="3385",
            confidence=0.65,
            bounding_box=NormalizedBoundingBox(xmin=0.3, ymin=0.2, xmax=0.7, ymax=0.28),
            pixel_box=PixelBoundingBox(x=150, y=250, width=300, height=75),
            normalized_value=3385.0,
            is_detected=True,
            extraction_status="ambiguous",
        )
        rx_field = OCRExtractedField(
            field_name="receiver",
            raw_text="To: Dev Malhotra",
            confidence=0.95,
            normalized_value="Dev Malhotra",
            is_detected=True,
            extraction_status="extracted",
        )
        status_field = OCRExtractedField(
            field_name="status",
            raw_text="Completed",
            confidence=0.98,
            normalized_value="Completed",
            is_detected=True,
            extraction_status="extracted",
        )

        ocr_res = OCRResult(
            fields=ExtractedTransactionData(amount="3385", amount_numeric=3385.0, receiver="Dev Malhotra", status="Completed"),
            detailed_fields={"amount": amt_field, "receiver": rx_field, "status": status_field},
            raw_detections=[amt_field, rx_field, status_field],
            overall_confidence=0.85,
        )

        sem_out = semantic_service.extract_features(ocr_res)
        assert isinstance(sem_out, SemanticBranchOutput)

        # Critical safeguard assertions:
        assert sem_out.field_anomalies.amount_anomaly is False, "Ambiguous amount must not assert amount_anomaly=True"
        assert ReasonCodeEnum.AMOUNT_CURRENCY_SYMBOL_MISMATCH not in sem_out.reason_codes
        evidence_sym = sem_out.feature_evidence.get("amount_currency_symbol_valid")
        assert evidence_sym is not None
        assert evidence_sym.feature_status == "ambiguous"
        assert evidence_sym.feature_value is True

    def test_final_verdict_clean_receipt_not_flipped_by_ambiguous_amount(self, semantic_service, verdict_service):
        """
        An authentic screenshot with ambiguous OCR amount extraction and clean CNN score
        must NOT trigger an EDITED or INCONCLUSIVE verdict.
        """
        from app.schemas.cnn import CNNModelOutput
        from app.schemas.stacker import StackerOutput

        # Authentic CNN score
        cnn_out = CNNModelOutput(
            predicted_class="clean",
            predicted_label="GENUINE",
            clean_probability=0.92,
            manipulation_probability=0.08,
            class_probabilities={"clean": 0.92, "edited": 0.08},
            model_version="resnet18-multitask-v1",
            execution_status="evaluated",
        )

        amt_field = OCRExtractedField(
            field_name="amount",
            raw_text="3385",
            confidence=0.65,
            bounding_box=NormalizedBoundingBox(xmin=0.3, ymin=0.2, xmax=0.7, ymax=0.28),
            pixel_box=PixelBoundingBox(x=150, y=250, width=300, height=75),
            normalized_value=3385.0,
            is_detected=True,
            extraction_status="ambiguous",
        )
        rx_field = OCRExtractedField(
            field_name="receiver",
            raw_text="To: Dev Malhotra",
            confidence=0.95,
            normalized_value="Dev Malhotra",
            is_detected=True,
            extraction_status="extracted",
        )
        status_field = OCRExtractedField(
            field_name="status",
            raw_text="Completed",
            confidence=0.98,
            normalized_value="Completed",
            is_detected=True,
            extraction_status="extracted",
        )

        ocr_res = OCRResult(
            fields=ExtractedTransactionData(amount="3385", amount_numeric=3385.0, receiver="Dev Malhotra", status="Completed"),
            detailed_fields={"amount": amt_field, "receiver": rx_field, "status": status_field},
            raw_detections=[amt_field, rx_field, status_field],
            overall_confidence=0.86,
        )

        sem_out = semantic_service.extract_features(ocr_res)
        assert sem_out.semantic_manipulation_score == 0.0

        stacker_out = StackerOutput(
            final_manipulation_probability=0.07,
            calibrated_confidence=0.93,
            stacker_model_version="stacker_v1_oof",
        )

        verdict_res = verdict_service.evaluate_verdict(
            stacker_output=stacker_out,
            cnn_result=cnn_out,
            ocr_result=ocr_res,
            semantic_result=sem_out,
        )

        assert verdict_res.verdict == VerdictEnum.GENUINE, f"Expected GENUINE, got {verdict_res.verdict}"
        assert verdict_res.manipulation_probability < 0.20


class TestPhase14SchemaCompatibility:
    """Verifies schema backward compatibility and serialization."""

    def test_candidates_field_serialization(self):
        """OCRExtractedField must serialize candidates cleanly and default to None."""
        field = OCRExtractedField(
            field_name="amount",
            raw_text="₹1,000",
            confidence=0.99,
            normalized_value=1000.0,
            is_detected=True,
            extraction_status="extracted",
            candidates=[{"candidate_text": "₹1,000", "numeric_value": 1000.0, "score": 12.5}],
        )
        d = field.model_dump()
        assert "candidates" in d
        assert len(d["candidates"]) == 1
        assert d["candidates"][0]["numeric_value"] == 1000.0

        # Without candidates (backward compatibility check)
        field_legacy = OCRExtractedField(
            field_name="status",
            raw_text="Completed",
            confidence=0.95,
            is_detected=True,
            extraction_status="extracted",
        )
        d_legacy = field_legacy.model_dump()
        assert d_legacy["candidates"] is None
