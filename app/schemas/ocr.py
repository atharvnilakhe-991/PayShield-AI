"""
OCR Extraction output contract for CBL-04 Google Pay receipt analysis.
Defines field-level detection schemas, normalized transaction metadata, and bounding box attachments.
"""

from typing import Any, Dict, List, Optional
from pydantic import Field, field_validator
from app.schemas.common import BaseSchema, NormalizedBoundingBox, PixelBoundingBox


class OCRExtractedField(BaseSchema):
    """
    Standardized schema for an individual field extracted via OCR.
    All fields are nullable to accommodate occluded, missing, or cropped transaction items.
    """
    field_name: str = Field(..., description="Canonical field identifier (e.g. 'amount', 'transaction_id')")
    raw_text: Optional[str] = Field(default=None, description="Exact OCR extracted string without modification")
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for this specific field detection [0.0 to 1.0]"
    )
    bounding_box: Optional[NormalizedBoundingBox] = Field(
        default=None,
        description="Normalized bounding box [0.0, 1.0] relative to image dimensions"
    )
    pixel_box: Optional[PixelBoundingBox] = Field(
        default=None,
        description="Absolute pixel coordinates (x, y, width, height) relative to original image"
    )
    normalized_value: Optional[Any] = Field(
        default=None,
        description="Parsed/canonical value (e.g., float for amount, ISO string for date/time)"
    )
    is_detected: bool = Field(
        default=False,
        description="Flag indicating whether this field was successfully localized and read"
    )
    extraction_status: Optional[str] = Field(
        default="not_found",
        description="Field extraction state: 'extracted', 'ambiguous', or 'not_found'"
    )
    candidates: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="All evaluated candidate amount tokens and bounding boxes"
    )

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            return round(max(0.0, min(1.0, float(v))), 4)
        return v


class ExtractedTransactionData(BaseSchema):
    """
    Aggregated and normalized transaction metadata extracted from a Google Pay screenshot.
    Designed for seamless serialization to both frontend cards and semantic feature extractors.
    All fields are optional and support null values when unreadable.
    """
    transaction_id: Optional[str] = Field(
        default=None,
        description="UPI Reference ID (12-digit RRN) or Google transaction ID string"
    )
    amount: Optional[str] = Field(
        default=None,
        description="Formatted transaction amount including currency symbol (e.g. '₹1,500.00')"
    )
    amount_numeric: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Clean parsed numeric amount (e.g. 1500.00)"
    )
    currency: Optional[str] = Field(
        default="INR",
        description="ISO 4217 currency code (default: INR)"
    )
    date: Optional[str] = Field(
        default=None,
        description="Parsed or raw transaction date string (e.g. '2026-11-03' or 'Nov 3, 2026')"
    )
    time: Optional[str] = Field(
        default=None,
        description="Parsed or raw transaction time string (e.g. '14:30:25' or '2:30 PM')"
    )
    sender: Optional[str] = Field(
        default=None,
        description="Sender display name or banking account holder"
    )
    sender_upi: Optional[str] = Field(
        default=None,
        description="Sender Virtual Payment Address (e.g. 'user@okaxis')"
    )
    receiver: Optional[str] = Field(
        default=None,
        description="Payee, merchant, or recipient display name"
    )
    receiver_upi: Optional[str] = Field(
        default=None,
        description="Payee Virtual Payment Address (e.g. 'store@okhdfcbank')"
    )
    status: Optional[str] = Field(
        default=None,
        description="Transaction status text (e.g. 'Completed', 'Payment to', 'Paid')"
    )
    google_txn_id: Optional[str] = Field(
        default=None,
        description="Google-specific transaction identifier if distinct from UPI ref"
    )


class OCRResult(BaseSchema):
    """
    Comprehensive output schema of the OCR extraction module (Phase 3).
    Feeds downstream semantic feature analysis (Phase 4) and API responses.
    """
    fields: ExtractedTransactionData = Field(
        default_factory=ExtractedTransactionData,
        description="Normalized key-value transaction metadata"
    )
    detailed_fields: Dict[str, OCRExtractedField] = Field(
        default_factory=dict,
        description="Dictionary mapping field names to their full bounding box and confidence telemetry"
    )
    raw_detections: List[OCRExtractedField] = Field(
        default_factory=list,
        description="List of all detected text tokens and bounding regions prior to field association"
    )
    overall_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mean confidence score across all successfully detected receipt fields"
    )
    ocr_engine: str = Field(
        default="EasyOCR",
        description="Name of the underlying OCR engine"
    )
    ocr_version: str = Field(
        default="1.7.2",
        description="Version string of the OCR engine"
    )
    raw_text: str = Field(
        default="",
        description="Unstructured raw text stream concatenation of all extracted tokens"
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken to perform OCR extraction in milliseconds"
    )

