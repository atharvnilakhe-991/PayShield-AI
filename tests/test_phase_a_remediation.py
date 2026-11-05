"""
Phase A Remediation Verification Test Suite for CBL-04.
Validates:
1. Final Verdict Engine corrections:
   - Removal of artificial 60% confidence floor.
   - Preservation of uncalibrated score labeling (no misleading calibrated probabilities).
   - Single uncorroborated semantic anomalies do not force EDITED.
   - Legitimate corroborated evidence produces EDITED.
   - Inter-branch evidence disagreement produces INCONCLUSIVE.
   - Missing OCR fields do not establish manipulation.
2. OCR Amount Extraction corrections:
   - ₹1 does not become ₹21; legitimate ₹21 remains ₹21.
   - ₹450 is extracted correctly.
   - Valid 4-digit amounts (e.g. ₹3,450, ₹1,250) are not discarded.
   - Masked phone number suffixes (e.g. "'17", "'39") are rejected.
   - Commas and decimal amounts are parsed correctly.
   - Uncertain OCR tokens preserve ambiguity status without fabricating corrections.
"""

import pytest
from app.schemas.enums import ReasonCodeEnum, VerdictEnum
from app.schemas.common import ModelVersionInfo, NormalizedBoundingBox, PixelBoundingBox
from app.schemas.cnn import CNNModelOutput
from app.schemas.ocr import ExtractedTransactionData, OCRExtractedField, OCRResult
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
from app.services.ocr_service import TransactionFieldExtractor


# ------------------------------------------------------------------------------
# Test Fixtures & Helpers
# ------------------------------------------------------------------------------

def _make_ocr_token(
    text: str,
    confidence: float = 0.90,
    ymin: float = 0.20,
    height: float = 70.0,
    xmin: float = 0.30,
    width: float = 120.0,
) -> OCRExtractedField:
    return OCRExtractedField(
        field_name="token",
        raw_text=text,
        confidence=confidence,
        bounding_box=NormalizedBoundingBox(
            xmin=xmin,
            ymin=ymin,
            xmax=xmin + width / 400.0,
            ymax=ymin + height / 800.0,
        ),
        pixel_box=PixelBoundingBox(x=xmin * 400.0, y=ymin * 800.0, width=width, height=height),
        is_detected=True,
        extraction_status="extracted",
    )


def _mock_stacker(prob: float = 0.50) -> StackerOutput:
    return StackerOutput(
        final_manipulation_probability=prob,
        calibrated_confidence=abs(prob - 0.5) * 2.0,
        stacker_model_version="v1.0.0-phase8",
        inference_time_ms=1.2,
    )


def _mock_cnn(score: float = 0.50) -> CNNModelOutput:
    return CNNModelOutput(
        manipulation_probability=score,
        model_info=ModelVersionInfo(model_name="ResNet18", model_version="fold_1_baseline"),
    )


def _mock_ocr(conf: float = 0.90) -> OCRResult:
    return OCRResult(
        fields=ExtractedTransactionData(
            amount="1,250.00",
            amount_numeric=1250.0,
            currency="INR",
            receiver="Demo Merchant",
            transaction_id="123456789012",
            date="15 Mar 2026",
            status="Completed",
        ),
        detailed_fields={},
        raw_detections=[],
        overall_confidence=conf,
        ocr_engine="EasyOCR",
        ocr_version="1.7.2",
        raw_text="Paid ₹1,250.00 to Demo Merchant",
        processing_time_ms=30.0,
    )


def _mock_semantic(
    score: float = 0.0,
    amount_anom: bool = False,
    recipient_anom: bool = False,
    date_anom: bool = False,
    missing: int = 0,
) -> SemanticBranchOutput:
    anom = FieldAnomalyIndicators(
        amount_anomaly=amount_anom,
        recipient_anomaly=recipient_anom,
        status_anomaly=False,
        date_anomaly=date_anom,
        ref_id_anomaly=False,
        layout_anomaly=False,
        upi_handle_anomaly=False,
    )
    codes = []
    if amount_anom:
        codes.append(ReasonCodeEnum.AMOUNT_CURRENCY_SYMBOL_MISMATCH)
    if recipient_anom:
        codes.append(ReasonCodeEnum.SENDER_RECEIVER_IDENTICAL)
    if date_anom:
        codes.append(ReasonCodeEnum.TIMESTAMP_CHRONOLOGY_ANOMALY)

    fv = SemanticFeatureVector(
        amount_valid_format=not amount_anom,
        amount_currency_symbol_valid=not amount_anom,
        amount_numeric_value_reasonable=True,
        ref_id_length_valid=True,
        ref_id_numeric_only=True,
        status_text_authentic=True,
        upi_handle_valid_format=True,
        date_chronology_valid=not date_anom,
        time_format_valid=True,
        sender_receiver_distinct=not recipient_anom,
        gpay_keywords_present=True,
        missing_critical_fields_count=missing,
        total_fields_detected=4 - missing,
    )
    return SemanticBranchOutput(
        semantic_manipulation_score=score,
        feature_vector=fv,
        reason_codes=codes,
        field_anomalies=anom,
        feature_evidence={},
        ocr_confidence_info=OCRConfidenceInfo(
            average_confidence=0.90,
            min_field_confidence=0.85,
            critical_fields_confidence=0.90,
            total_detected_tokens=20,
        ),
        semantic_status="live",
        processing_time_ms=10.0,
    )


# ==============================================================================
# Part 1: Final Verdict Engine Remediation Tests
# ==============================================================================

class TestFinalVerdictRemediation:
    """Verifies elimination of the 60% floor, uncalibrated score labeling, and ambiguity protection."""

    def setup_method(self):
        reset_final_verdict_service()
        self.service = get_final_verdict_service()

    def teardown_method(self):
        reset_final_verdict_service()

    def test_stacker_near_threshold_does_not_produce_sixty_percent_floor(self):
        """
        CRITICAL BUG FIX: When stacker probability is 0.5008 (essentially 50-50 uncertainty),
        confidence must reflect true decision margin (~0.50), NOT an artificial 0.60 floor.
        """
        stacker = _mock_stacker(prob=0.5008)
        cnn = _mock_cnn(score=0.52)
        ocr = _mock_ocr(conf=0.90)
        sem = _mock_semantic(score=0.0, amount_anom=False)

        res = self.service.evaluate_verdict(stacker, cnn, ocr, sem)
        # Inside [0.45, 0.55], it must trigger ambiguity band -> INCONCLUSIVE
        assert res.verdict == VerdictEnum.INCONCLUSIVE
        # Confidence must be None for INCONCLUSIVE, with uncertainty score capturing boundary ambiguity
        assert res.confidence is None
        assert res.uncertainty_score > 0.99
        assert res.decision_score == 0.5008

    def test_uncalibrated_score_metadata(self):
        """Ensures score_type and calibration_status remain explicitly uncalibrated."""
        stacker = _mock_stacker(prob=0.88)
        cnn = _mock_cnn(score=0.85)
        ocr = _mock_ocr(conf=0.92)
        sem = _mock_semantic(score=0.70, amount_anom=True, recipient_anom=True)

        res = self.service.evaluate_verdict(stacker, cnn, ocr, sem)
        assert res.verdict == VerdictEnum.EDITED
        assert res.calibration_status == "uncalibrated"
        assert res.score_type == "raw_stacker_logistic_output"

    def test_single_uncertain_semantic_anomaly_does_not_force_edited(self):
        """
        A single uncorroborated anomaly (e.g. currency symbol mismatch due to OCR)
        must NOT override the ambiguity margin or force an EDITED verdict when CNN is non-corroborating.
        """
        stacker = _mock_stacker(prob=0.5015)
        cnn = _mock_cnn(score=0.52)
        ocr = _mock_ocr(conf=0.88)
        # Only 1 anomaly (amount_anom=True)
        sem = _mock_semantic(score=0.20, amount_anom=True, recipient_anom=False)

        res = self.service.evaluate_verdict(stacker, cnn, ocr, sem)
        # Must produce INCONCLUSIVE under rule_ambiguity_band, NOT EDITED
        assert res.verdict == VerdictEnum.INCONCLUSIVE
        assert res.decision_rule == "rule_ambiguity_band"
        assert ReasonCodeEnum.AMOUNT_CURRENCY_SYMBOL_MISMATCH in res.reason_codes

    def test_strong_corroborated_evidence_produces_edited(self):
        """Independent corroboration (stacker + CNN + multiple semantic anomalies) produces EDITED."""
        stacker = _mock_stacker(prob=0.82)
        cnn = _mock_cnn(score=0.79)
        ocr = _mock_ocr(conf=0.91)
        sem = _mock_semantic(score=0.65, amount_anom=True, recipient_anom=True)

        res = self.service.evaluate_verdict(stacker, cnn, ocr, sem)
        assert res.verdict == VerdictEnum.EDITED
        assert res.decision_rule == "rule_edited_decision"
        assert res.confidence >= 0.80

    def test_conflicting_evidence_produces_inconclusive(self):
        """Severe branch conflict (e.g. CNN says 0.80 tampering, but Semantic says 0.0 clean) produces INCONCLUSIVE."""
        stacker = _mock_stacker(prob=0.58)
        cnn = _mock_cnn(score=0.80)
        ocr = _mock_ocr(conf=0.92)
        sem = _mock_semantic(score=0.0, amount_anom=False)

        res = self.service.evaluate_verdict(stacker, cnn, ocr, sem)
        assert res.verdict == VerdictEnum.INCONCLUSIVE
        assert res.decision_rule == "rule_severe_branch_conflict"

    def test_branch_disagreement_semantic_tamper_cnn_clean(self):
        """When semantic branch flags anomaly but CNN indicates clean pixels (<= 0.35), return INCONCLUSIVE."""
        stacker = _mock_stacker(prob=0.48)
        cnn = _mock_cnn(score=0.25)
        ocr = _mock_ocr(conf=0.88)
        sem = _mock_semantic(score=0.40, amount_anom=True)

        res = self.service.evaluate_verdict(stacker, cnn, ocr, sem)
        assert res.verdict == VerdictEnum.INCONCLUSIVE
        assert res.decision_rule in ["rule_branch_disagreement", "rule_ambiguity_band"]

    def test_missing_ocr_fields_do_not_independently_establish_manipulation(self):
        """Missing OCR fields due to poor scan quality produce INCONCLUSIVE, not EDITED."""
        stacker = _mock_stacker(prob=0.49)
        cnn = _mock_cnn(score=0.45)
        ocr = _mock_ocr(conf=0.35)
        sem = _mock_semantic(score=0.0, amount_anom=False, missing=3)

        res = self.service.evaluate_verdict(stacker, cnn, ocr, sem)
        assert res.verdict == VerdictEnum.INCONCLUSIVE
        assert res.decision_rule == "rule_low_ocr_legibility"


# ==============================================================================
# Part 2: OCR Amount Extraction Remediation Tests
# ==============================================================================

class TestOCRAmountRemediation:
    """Verifies correct amount extraction, 4-digit preservation, phone suffix rejection, and ₹1 vs ₹21 integrity."""

    def test_rupee_one_does_not_become_twenty_one(self):
        """₹1 must be extracted with numeric value 1.0, NOT 21.0."""
        tok = _make_ocr_token("₹1", ymin=0.22, height=72.0)
        field, val = TransactionFieldExtractor._extract_amount([tok], image_height=800)
        assert field.is_detected is True
        assert val == 1.0
        assert field.raw_text == "₹1"

    def test_legitimate_rupee_twenty_one_remains_twenty_one(self):
        """Legitimate ₹21 must remain 21.0, NOT rewritten to 1.0."""
        tok = _make_ocr_token("₹21", ymin=0.22, height=72.0)
        field, val = TransactionFieldExtractor._extract_amount([tok], image_height=800)
        assert field.is_detected is True
        assert val == 21.0
        assert field.raw_text == "₹21"

    def test_four_digit_amount_not_discarded(self):
        """
        CRITICAL BUG FIX: Four-digit amounts (e.g. 3450, 1250, 2400) without currency
        symbol must NOT be discarded by the extractor.
        """
        tok = _make_ocr_token("3450", ymin=0.20, height=76.0)
        field, val = TransactionFieldExtractor._extract_amount([tok], image_height=800)
        assert field.is_detected is True
        assert val == 3450.0
        assert field.raw_text == "3450"

    def test_rupee_four_fifty_extracted_correctly(self):
        """Amount {450 with curly brace OCR prefix is recognized as 450.0."""
        tok = _make_ocr_token("{450", ymin=0.20, height=70.0)
        field, val = TransactionFieldExtractor._extract_amount([tok], image_height=800)
        assert field.is_detected is True
        assert val == 450.0
        assert field.raw_text == "{450"

    def test_phone_number_suffix_rejected_as_amount(self):
        """
        CRITICAL BUG FIX: Subtitle masked phone suffixes like "'17" or "'39"
        at y < 0.18 must NOT be selected as transaction amounts.
        """
        phone_tok = _make_ocr_token("'+91 98xxx xx'17", ymin=0.15, height=22.0)
        phone_suffix = _make_ocr_token("'17", ymin=0.152, height=18.0)
        amount_tok = _make_ocr_token("3450", ymin=0.20, height=76.0)

        field, val = TransactionFieldExtractor._extract_amount(
            [phone_tok, phone_suffix, amount_tok], image_height=800
        )
        assert field.is_detected is True
        assert val == 3450.0
        assert field.raw_text == "3450"

    def test_amounts_with_commas_and_decimals(self):
        """Commas and decimal points in amounts are parsed accurately."""
        tok1 = _make_ocr_token("₹2,400.00", ymin=0.22, height=75.0)
        _, val1 = TransactionFieldExtractor._extract_amount([tok1], image_height=800)
        assert val1 == 2400.0

        tok2 = _make_ocr_token("{1,25,000.50", ymin=0.22, height=75.0)
        _, val2 = TransactionFieldExtractor._extract_amount([tok2], image_height=800)
        assert val2 == 125000.5

    def test_uncertain_ocr_tokens_preserve_ambiguity(self):
        """Amounts without an explicit currency symbol preserve ambiguity status for downstream auditing."""
        tok = _make_ocr_token("450", ymin=0.20, height=70.0)
        field, val = TransactionFieldExtractor._extract_amount([tok], image_height=800)
        assert field.is_detected is True
        assert val == 450.0
        # When currency indicator is missing, field should preserve ambiguous status
        assert field.extraction_status == "ambiguous"

