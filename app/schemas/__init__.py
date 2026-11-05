"""
Schemas package exporting all CBL-04 data contracts, interfaces, and enums.
"""

from app.schemas.enums import (
    VerdictEnum,
    ReasonCodeEnum,
    ErrorCodeEnum,
    CoordinateSystemEnum,
    REASON_CODE_DESCRIPTIONS,
)
from app.schemas.common import (
    BaseSchema,
    NormalizedBoundingBox,
    PixelBoundingBox,
    ModelVersionInfo,
)
from app.schemas.request import (
    AnalyzeRequest,
    ImageUploadConfig,
    ClientMetadata,
    validate_image_metadata,
)
from app.schemas.ocr import (
    OCRExtractedField,
    ExtractedTransactionData,
    OCRResult,
)
from app.schemas.cnn import (
    CNNSuspiciousRegion,
    CNNModelInput,
    CNNModelOutput,
    CNNInferenceInterface,
)
from app.schemas.semantic import (
    SemanticFeatureVector,
    FieldAnomalyIndicators,
    OCRConfidenceInfo,
    SemanticBranchOutput,
)
from app.schemas.stacker import (
    CalibrationMetadata,
    StackerInput,
    StackerOutput,
    StackerInferenceInterface,
)
from app.schemas.response import (
    SuspiciousRegion,
    ManipulationAnalysisEvidence,
    BranchScores,
    ModelVersionMetadata,
    AnalyzeResponse,
)
from app.schemas.errors import (
    APIErrorResponse,
    create_error_response,
    sanitize_error_detail,
    ERROR_HTTP_STATUS_MAP,
)

__all__ = [
    # Enums
    "VerdictEnum",
    "ReasonCodeEnum",
    "ErrorCodeEnum",
    "CoordinateSystemEnum",
    "REASON_CODE_DESCRIPTIONS",
    # Common
    "BaseSchema",
    "NormalizedBoundingBox",
    "PixelBoundingBox",
    "ModelVersionInfo",
    # Request
    "AnalyzeRequest",
    "ImageUploadConfig",
    "ClientMetadata",
    "validate_image_metadata",
    # OCR
    "OCRExtractedField",
    "ExtractedTransactionData",
    "OCRResult",
    # CNN
    "CNNSuspiciousRegion",
    "CNNModelInput",
    "CNNModelOutput",
    "CNNInferenceInterface",
    # Semantic
    "SemanticFeatureVector",
    "FieldAnomalyIndicators",
    "OCRConfidenceInfo",
    "SemanticBranchOutput",
    # Stacker
    "CalibrationMetadata",
    "StackerInput",
    "StackerOutput",
    "StackerInferenceInterface",
    # Response
    "SuspiciousRegion",
    "ManipulationAnalysisEvidence",
    "BranchScores",
    "ModelVersionMetadata",
    "AnalyzeResponse",
    # Errors
    "APIErrorResponse",
    "create_error_response",
    "sanitize_error_detail",
    "ERROR_HTTP_STATUS_MAP",
]

