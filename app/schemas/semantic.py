"""
Semantic Feature Engine Contract for Member 3.
Defines output schemas for OCR-driven semantic heuristic analysis, rule-based feature vectors,
field-level anomaly flags, and machine-readable reason codes.
"""

from typing import Any, Dict, List, Optional
from pydantic import Field
from app.schemas.common import BaseSchema, ModelVersionInfo, NormalizedBoundingBox
from app.schemas.enums import ReasonCodeEnum


class SemanticFeatureEvidence(BaseSchema):
    """
    Evidence provenance for an individual semantic feature.
    Links the rule evaluation back to the raw OCR text, bounding box, confidence, and rule explanation.
    """
    feature_name: str = Field(..., description="Name of the evaluated semantic feature")
    feature_value: Any = Field(..., description="Evaluated feature value (bool, int, float, str, or None)")
    feature_status: str = Field(
        default="available",
        description="Feature status: 'available', 'not_available', 'ambiguous', 'invalid_format', 'not_evaluated'"
    )
    source_ocr_text: Optional[str] = Field(default=None, description="Raw OCR text token(s) used as evidence")
    source_bounding_box: Optional[NormalizedBoundingBox] = Field(
        default=None,
        description="Bounding box of the primary OCR token providing evidence"
    )
    rule_id: str = Field(..., description="Identifier of the rule responsible for evaluation (e.g. 'R_AMT_001')")
    explanation: str = Field(..., description="Human-readable explanation of the rule evaluation")
    ocr_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="OCR confidence of the source evidence")
    is_missing: bool = Field(default=False, description="Flag indicating whether feature evidence was missing from OCR")


class SemanticFeatureVector(BaseSchema):
    """
    Structured numeric and boolean features extracted from OCR text and receipt layout geometry.
    This feature representation feeds both rule-based heuristics and downstream stacking classifiers.
    """
    amount_valid_format: bool = Field(
        default=True,
        description="Whether the amount string adheres to standard Indian currency formatting (e.g. ₹X,XXX.XX)"
    )
    amount_currency_symbol_valid: bool = Field(
        default=True,
        description="Whether the Indian Rupee symbol (₹) is present and correctly positioned"
    )
    amount_numeric_value_reasonable: bool = Field(
        default=True,
        description="Whether the numeric amount falls within standard UPI limits (<= ₹1,00,000 for standard P2P)"
    )
    ref_id_length_valid: bool = Field(
        default=True,
        description="Whether the UPI reference number (RRN) is exactly 12 digits"
    )
    ref_id_numeric_only: bool = Field(
        default=True,
        description="Whether the UPI reference number contains only numeric digits"
    )
    status_text_authentic: bool = Field(
        default=True,
        description="Whether the transaction status matches authentic GPay labels ('Paid', 'Payment to', 'Completed')"
    )
    upi_handle_valid_format: bool = Field(
        default=True,
        description="Whether visible UPI IDs follow valid RFC-style syntax (user@bankhandle)"
    )
    date_chronology_valid: bool = Field(
        default=True,
        description="Whether the transaction date is in the past/present and represents a valid calendar day"
    )
    time_format_valid: bool = Field(
        default=True,
        description="Whether the time string conforms to 12-hour (e.g. 2:30 PM) or 24-hour GPay standard"
    )
    sender_receiver_distinct: bool = Field(
        default=True,
        description="Whether payer and payee identifiers are distinct"
    )
    gpay_keywords_present: bool = Field(
        default=True,
        description="Whether key Google Pay interface anchor keywords are detected in receipt layout"
    )
    font_size_ratio_amount_to_payee: float = Field(
        default=1.0,
        ge=0.0,
        description="Ratio of amount text bounding height to recipient name height (expected > 1.2 in genuine GPay)"
    )
    text_vertical_alignment_variance: float = Field(
        default=0.0,
        ge=0.0,
        description="Variance in vertical centerlines for horizontally aligned metadata pairs"
    )
    missing_critical_fields_count: int = Field(
        default=0,
        ge=0,
        description="Count of mandatory receipt fields missing from OCR detections"
    )
    total_fields_detected: int = Field(
        default=0,
        ge=0,
        description="Total number of transaction fields successfully extracted"
    )
    custom_numeric_features: Dict[str, float] = Field(
        default_factory=dict,
        description="Dynamic dictionary for additional engineered continuous features"
    )


class FieldAnomalyIndicators(BaseSchema):
    """
    Field-level anomaly flags isolating specific transaction attributes that triggered suspicion.
    Provides fine-grained explainability without declaring legal fraud.
    """
    amount_anomaly: bool = Field(default=False, description="Flagged format, currency, or decimal anomaly in amount")
    recipient_anomaly: bool = Field(default=False, description="Flagged anomaly in payee name or UPI address")
    status_anomaly: bool = Field(default=False, description="Flagged anomaly in payment status text or casing")
    date_anomaly: bool = Field(default=False, description="Flagged anomaly in date or time chronology")
    ref_id_anomaly: bool = Field(default=False, description="Flagged anomaly in UPI 12-digit RRN pattern")
    layout_anomaly: bool = Field(default=False, description="Flagged geometric misalignment or unnatural spacing")
    upi_handle_anomaly: bool = Field(default=False, description="Flagged invalid bank handle suffix (e.g. missing @ok...)")


class OCRConfidenceInfo(BaseSchema):
    """Statistical summary of OCR token recognition confidence."""
    average_confidence: float = Field(..., ge=0.0, le=1.0, description="Mean OCR token confidence score")
    min_field_confidence: float = Field(..., ge=0.0, le=1.0, description="Minimum confidence observed across key fields")
    critical_fields_confidence: float = Field(..., ge=0.0, le=1.0, description="Mean confidence for amount, status, and ref_id")
    total_detected_tokens: int = Field(default=0, ge=0, description="Total word/token bounding boxes extracted")


class SemanticBranchOutput(BaseSchema):
    """
    Standardized output schema produced by the Semantic Feature Engine (Phase 4).
    Supplies the semantic manipulation score, feature vector, reason codes, and anomaly indicators.
    """
    semantic_manipulation_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of manipulation evaluated by semantic and layout heuristics [0.0 = Authentic, 1.0 = Manipulated]"
    )
    feature_vector: SemanticFeatureVector = Field(
        ...,
        description="Structured machine-readable semantic feature vector"
    )
    reason_codes: List[ReasonCodeEnum] = Field(
        default_factory=list,
        description="Machine-readable structured reason codes identifying detected anomalies"
    )
    field_anomalies: FieldAnomalyIndicators = Field(
        default_factory=FieldAnomalyIndicators,
        description="Binary flags indicating which individual fields exhibited anomalies"
    )
    ocr_confidence_info: OCRConfidenceInfo = Field(
        ...,
        description="Statistical summary of OCR recognition confidence"
    )
    model_info: ModelVersionInfo = Field(
        default_factory=lambda: ModelVersionInfo(
            model_name="GPay_Semantic_Engine",
            model_version="v1.0.0",
            architecture="RuleEngine+XGBoost_Baseline",
        ),
        description="Version and architecture descriptor of the semantic engine"
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Semantic analysis wall-clock time in milliseconds"
    )
    feature_evidence: Dict[str, SemanticFeatureEvidence] = Field(
        default_factory=dict,
        description="Traceable rule-level evidence and provenance for extracted semantic features"
    )
    semantic_status: str = Field(
        default="live",
        description="Status of semantic branch: 'live', 'mock', 'not_run', or 'ocr_failed'"
    )

