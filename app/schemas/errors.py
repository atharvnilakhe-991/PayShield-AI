"""
Standardized API Error Contract for CBL-04.
Enforces consistent error formatting, machine-readable error codes,
request trace correlation, and strict security sanitization (no internal paths or tracebacks).
"""

from datetime import datetime, timezone
import re
from typing import Any, Optional
from pydantic import Field
from app.schemas.common import BaseSchema
from app.schemas.enums import ErrorCodeEnum


def _sanitize_string(val: str) -> str:
    """Removes local filesystem paths and tracebacks from error strings."""
    # Redact Windows paths (e.g. C:\Users\... or D:\...)
    val = re.sub(r"[a-zA-Z]:\\[^\s:<>\"|?*]+", "[REDACTED_PATH]", val)
    # Redact Unix paths (e.g. /home/... or /var/...)
    val = re.sub(r"/(?:[a-zA-Z0-9_\.\-]+/)+[a-zA-Z0-9_\.\-]+", "[REDACTED_PATH]", val)
    # Redact Python traceback line references
    val = re.sub(r'File "[^"]+", line \d+', 'File "[REDACTED]", line 0', val)
    return val


def sanitize_error_detail(detail: Any) -> Any:
    """Recursively sanitizes error details to prevent accidental data leaks."""
    if isinstance(detail, str):
        return _sanitize_string(detail)
    elif isinstance(detail, list):
        return [sanitize_error_detail(item) for item in detail]
    elif isinstance(detail, dict):
        return {k: sanitize_error_detail(v) for k, v in detail.items()}
    return detail


class APIErrorResponse(BaseSchema):
    """
    Standard error response contract returned on 4xx/5xx HTTP responses.
    Guarantees consistent schema across all error conditions.
    """
    error_code: ErrorCodeEnum = Field(
        ...,
        description="Standardized machine-readable error classification code"
    )
    message: str = Field(
        ...,
        description="Safe, human-readable summary of the error condition"
    )
    detail: Optional[Any] = Field(
        default=None,
        description="Additional contextual detail or validation error list (sanitized of sensitive paths/traces)"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Correlation ID of the request for tracing in backend logs"
    )
    status_code: int = Field(
        ...,
        ge=400,
        le=599,
        description="Corresponding HTTP status code (e.g. 400, 413, 415, 422, 500, 503)"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of the error occurrence"
    )


def create_error_response(
    error_code: ErrorCodeEnum,
    message: str,
    status_code: int,
    request_id: Optional[str] = None,
    detail: Optional[Any] = None,
) -> APIErrorResponse:
    """
    Factory function to instantiate sanitized APIErrorResponse objects.
    """
    return APIErrorResponse(
        error_code=error_code,
        message=_sanitize_string(message),
        detail=sanitize_error_detail(detail) if detail is not None else None,
        request_id=request_id,
        status_code=status_code,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# Standard HTTP Status mappings
ERROR_HTTP_STATUS_MAP = {
    ErrorCodeEnum.INVALID_IMAGE_FORMAT: 415,
    ErrorCodeEnum.FILE_TOO_LARGE: 413,
    ErrorCodeEnum.CORRUPT_IMAGE: 400,
    ErrorCodeEnum.IMAGE_UNREADABLE: 400,
    ErrorCodeEnum.IMAGE_DIMENSIONS_OUT_OF_RANGE: 422,
    ErrorCodeEnum.MISSING_REQUIRED_INPUT: 400,
    ErrorCodeEnum.VALIDATION_ERROR: 422,
    ErrorCodeEnum.MODEL_UNAVAILABLE: 503,
    ErrorCodeEnum.INFERENCE_TIMEOUT: 504,
    ErrorCodeEnum.INTERNAL_PROCESSING_FAILURE: 500,
}

