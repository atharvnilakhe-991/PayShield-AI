"""
Enums and constant identifiers for CBL-04 Google Pay Screenshot Manipulation Detection.
Defines canonical verdicts, machine-readable reason codes, error codes, and coordinate conventions.
"""

from enum import Enum
from typing import Dict


class VerdictEnum(str, Enum):
    """
    Standard verdicts for screenshot authenticity analysis.
    Supports canonical verdicts and backwards-compatible aliases for frontend integration.
    """
    GENUINE = "GENUINE"
    SUSPICIOUS = "SUSPICIOUS"
    EDITED = "EDITED"
    INCONCLUSIVE = "INCONCLUSIVE"

    # Compatibility aliases (mapped from frontend Member 4 conventions)
    REAL = "REAL"
    FAKE = "FAKE"

    @classmethod
    def to_canonical(cls, value: "VerdictEnum | str") -> "VerdictEnum":
        """Maps compatibility aliases to canonical verdicts."""
        val_str = value.value if isinstance(value, VerdictEnum) else str(value).upper()
        mapping = {
            "REAL": cls.GENUINE,
            "FAKE": cls.EDITED,
            "GENUINE": cls.GENUINE,
            "EDITED": cls.EDITED,
            "SUSPICIOUS": cls.SUSPICIOUS,
            "INCONCLUSIVE": cls.INCONCLUSIVE,
        }
        return mapping.get(val_str, cls.INCONCLUSIVE)


class ReasonCodeEnum(str, Enum):
    """
    Structured, machine-readable reason codes for semantic and visual anomalies.
    Codes describe verifiable inconsistencies and digital artifacts without asserting legal fraud.
    """
    # Amount & Currency checks
    AMOUNT_FORMAT_ANOMALY = "AMOUNT_FORMAT_ANOMALY"
    AMOUNT_CURRENCY_SYMBOL_MISMATCH = "AMOUNT_CURRENCY_SYMBOL_MISMATCH"
    AMOUNT_DECIMAL_ANOMALY = "AMOUNT_DECIMAL_ANOMALY"

    # Layout & Alignment checks
    TEXT_LAYOUT_INCONSISTENCY = "TEXT_LAYOUT_INCONSISTENCY"
    FONT_PROPORTION_ANOMALY = "FONT_PROPORTION_ANOMALY"
    SPACING_OR_PADDING_IRREGULAR = "SPACING_OR_PADDING_IRREGULAR"

    # Status & Header checks
    STATUS_TEXT_ANOMALY = "STATUS_TEXT_ANOMALY"
    HEADER_STRUCTURE_ANOMALY = "HEADER_STRUCTURE_ANOMALY"

    # Reference & Transaction Identifiers
    REF_ID_FORMAT_ANOMALY = "REF_ID_FORMAT_ANOMALY"
    UPI_ID_SYNTAX_INVALID = "UPI_ID_SYNTAX_INVALID"
    TRANSACTION_FIELD_INCONSISTENCY = "TRANSACTION_FIELD_INCONSISTENCY"

    # Temporal / Timestamp checks
    TIMESTAMP_CHRONOLOGY_ANOMALY = "TIMESTAMP_CHRONOLOGY_ANOMALY"
    TIME_FORMAT_INVALID = "TIME_FORMAT_INVALID"

    # Parties involved
    SENDER_RECEIVER_IDENTICAL = "SENDER_RECEIVER_IDENTICAL"
    MISSING_CRITICAL_PAYMENT_FIELD = "MISSING_CRITICAL_PAYMENT_FIELD"

    # Spatial / Visual Artifacts
    COMPRESSION_MISMATCH_DETECTED = "COMPRESSION_MISMATCH_DETECTED"
    SUSPICIOUS_SPATIAL_CLUSTER = "SUSPICIOUS_SPATIAL_CLUSTER"

    # OCR extraction quality
    OCR_CONFIDENCE_LOW = "OCR_CONFIDENCE_LOW"
    OCR_TEXT_CORRUPTED = "OCR_TEXT_CORRUPTED"

    # Clean verification
    INTEGRITY_VERIFIED_CLEAN = "INTEGRITY_VERIFIED_CLEAN"


class ErrorCodeEnum(str, Enum):
    """
    Standardized machine-readable error codes for API error responses.
    """
    INVALID_IMAGE_FORMAT = "INVALID_IMAGE_FORMAT"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    CORRUPT_IMAGE = "CORRUPT_IMAGE"
    IMAGE_UNREADABLE = "IMAGE_UNREADABLE"
    IMAGE_DIMENSIONS_OUT_OF_RANGE = "IMAGE_DIMENSIONS_OUT_OF_RANGE"
    MISSING_REQUIRED_INPUT = "MISSING_REQUIRED_INPUT"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"
    INTERNAL_PROCESSING_FAILURE = "INTERNAL_PROCESSING_FAILURE"


class CoordinateSystemEnum(str, Enum):
    """
    Defines coordinate systems used for bounding box localization.
    Normalized coordinates [0.0, 1.0] are preferred for cross-resolution invariance.
    """
    NORMALIZED = "NORMALIZED"  # Floating point [0.0, 1.0] relative to (width, height)
    PIXEL = "PIXEL"            # Absolute integer/float pixels relative to original natural resolution


# Human-readable reason code documentation
REASON_CODE_DESCRIPTIONS: Dict[ReasonCodeEnum, str] = {
    ReasonCodeEnum.AMOUNT_FORMAT_ANOMALY: "Transaction amount formatting, delimiter, or digits deviate from standard GPay format.",
    ReasonCodeEnum.AMOUNT_CURRENCY_SYMBOL_MISMATCH: "Currency symbol font glyph, size, or placement does not match the official Indian Rupee font.",
    ReasonCodeEnum.AMOUNT_DECIMAL_ANOMALY: "Irregular decimal formatting detected in transaction amount.",
    ReasonCodeEnum.TEXT_LAYOUT_INCONSISTENCY: "Extracted text bounding boxes exhibit misalignment or irregular vertical baseline shifts.",
    ReasonCodeEnum.FONT_PROPORTION_ANOMALY: "Font stroke width, letter-spacing, or glyph ratio differs from standard Roboto/Google Sans.",
    ReasonCodeEnum.SPACING_OR_PADDING_IRREGULAR: "Spacing between transaction header, amount block, and metadata deviates from official template.",
    ReasonCodeEnum.STATUS_TEXT_ANOMALY: "Payment status text (e.g. 'Completed', 'Paid') contains unexpected wording, casing, or styling.",
    ReasonCodeEnum.HEADER_STRUCTURE_ANOMALY: "Top application bar or receipt header differs structurally from genuine Google Pay UI.",
    ReasonCodeEnum.REF_ID_FORMAT_ANOMALY: "UPI reference number or Google transaction ID length/character pattern does not match 12-digit standard.",
    ReasonCodeEnum.UPI_ID_SYNTAX_INVALID: "UPI Virtual Payment Address (VPA) fails standard syntax validation (username@bankhandle).",
    ReasonCodeEnum.TRANSACTION_FIELD_INCONSISTENCY: "Inconsistency between complementary transaction metadata fields.",
    ReasonCodeEnum.TIMESTAMP_CHRONOLOGY_ANOMALY: "Transaction timestamp contains an irrational date, future date, or impossible calendar value.",
    ReasonCodeEnum.TIME_FORMAT_INVALID: "Time string fails 12-hour/24-hour GPay formatting standard.",
    ReasonCodeEnum.SENDER_RECEIVER_IDENTICAL: "Payer and payee names or UPI IDs are identical.",
    ReasonCodeEnum.MISSING_CRITICAL_PAYMENT_FIELD: "One or more core payment fields (amount, status, or date) were not detected.",
    ReasonCodeEnum.COMPRESSION_MISMATCH_DETECTED: "Localized difference in JPEG compression block artifacts or error level analysis (ELA).",
    ReasonCodeEnum.SUSPICIOUS_SPATIAL_CLUSTER: "CNN Grad-CAM activation or localized detector flags clustered manipulation artifacts.",
    ReasonCodeEnum.OCR_CONFIDENCE_LOW: "Average OCR recognition confidence fell below acceptable threshold.",
    ReasonCodeEnum.OCR_TEXT_CORRUPTED: "Detected characters contain heavy noise or unparseable symbol substitutions.",
    ReasonCodeEnum.INTEGRITY_VERIFIED_CLEAN: "Structural, semantic, and localized visual inspection detected no anomalous signatures.",
}

