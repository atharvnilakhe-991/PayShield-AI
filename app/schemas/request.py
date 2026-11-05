"""
API Request contract and image upload configuration for CBL-04.
Specifies multipart upload constraints, supported formats, size limits, and validation rules.
"""

from typing import List, Optional, Tuple, Dict, Any
from pydantic import Field, field_validator
from app.schemas.common import BaseSchema
from app.schemas.enums import ErrorCodeEnum


class ImageUploadConfig(BaseSchema):
    """
    Configurable image upload constraints.
    These values define server limits and can be overridden via environment configuration.
    """
    supported_mime_types: List[str] = Field(
        default=["image/jpeg", "image/png", "image/webp"],
        description="Accepted image MIME content types"
    )
    supported_extensions: List[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".webp"],
        description="Accepted file extensions (lowercase)"
    )
    min_file_size_bytes: int = Field(
        default=1,
        ge=1,
        description="Minimum file size in bytes (rejects 0-byte empty files)"
    )
    max_file_size_bytes: int = Field(
        default=10 * 1024 * 1024,  # 10 Megabytes
        ge=1024,
        description="Maximum upload payload size in bytes (Default: 10 MB)"
    )
    min_dimension_px: Tuple[int, int] = Field(
        default=(100, 100),
        description="(min_width, min_height) acceptable pixel dimensions"
    )
    max_dimension_px: Tuple[int, int] = Field(
        default=(4096, 4096),
        description="(max_width, max_height) maximum allowed pixel dimensions"
    )


class ClientMetadata(BaseSchema):
    """Optional client-side telemetry or device metadata supplied with upload."""
    client_version: Optional[str] = Field(default=None, description="Frontend or mobile client version (e.g. '1.0.0')")
    platform: Optional[str] = Field(default=None, description="Client platform (e.g. 'web', 'android', 'ios')")
    device_model: Optional[str] = Field(default=None, description="User device model or user agent snippet")
    custom_tags: Optional[Dict[str, Any]] = Field(default=None, description="Custom key-value tagging for audit logging")


class AnalyzeRequest(BaseSchema):
    """
    Contract schema representing the multipart/form-data upload payload for `/api/v1/analyze`.
    
    Image Transfer Protocol:
      - Transfer Type: `multipart/form-data`
      - Primary File Field: `image` (Binary stream / UploadFile)
      - Form Fields: `request_id` (optional), `return_heatmap` (optional), `metadata` (optional)
      
    Note on Authenticity Assumption:
      Incoming screenshots must never be assumed authentic simply because they visually
      resemble Google Pay. All uploads pass through the full multi-branch verification pipeline.
    """
    request_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Optional client-provided correlation ID. If omitted, the backend generates a UUIDv4."
    )
    return_heatmap: bool = Field(
        default=True,
        description="Whether to generate and return Grad-CAM / ELA heatmap explainability evidence."
    )
    metadata: Optional[ClientMetadata] = Field(
        default=None,
        description="Optional client metadata or telemetry."
    )

    @field_validator("request_id")
    @classmethod
    def sanitize_request_id(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            if not v:
                return None
        return v


def validate_image_metadata(
    filename: Optional[str],
    content_type: Optional[str],
    file_size_bytes: int,
    config: Optional[ImageUploadConfig] = None,
) -> Tuple[bool, Optional[ErrorCodeEnum], Optional[str]]:
    """
    Pure validation helper for incoming upload metadata against contract constraints.
    Returns: (is_valid, error_code, error_message)
    """
    cfg = config or ImageUploadConfig()

    # Check 1: Empty file
    if file_size_bytes < cfg.min_file_size_bytes:
        return (
            False,
            ErrorCodeEnum.CORRUPT_IMAGE,
            f"The uploaded file is empty (size: {file_size_bytes} bytes). A valid screenshot file is required.",
        )

    # Check 2: Payload too large
    if file_size_bytes > cfg.max_file_size_bytes:
        max_mb = cfg.max_file_size_bytes / (1024 * 1024)
        return (
            False,
            ErrorCodeEnum.FILE_TOO_LARGE,
            f"Image file size ({file_size_bytes} bytes) exceeds the maximum allowed limit of {max_mb:.1f} MB.",
        )

    # Check 3: Content-Type / MIME check
    if content_type:
        normalized_mime = content_type.lower().split(";")[0].strip()
        if normalized_mime not in cfg.supported_mime_types:
            return (
                False,
                ErrorCodeEnum.INVALID_IMAGE_FORMAT,
                f"Unsupported media type '{content_type}'. Accepted formats: {', '.join(cfg.supported_mime_types)}.",
            )

    # Check 4: File extension check
    if filename:
        lower_name = filename.lower()
        if not any(lower_name.endswith(ext) for ext in cfg.supported_extensions):
            return (
                False,
                ErrorCodeEnum.INVALID_IMAGE_FORMAT,
                f"Unsupported file extension in '{filename}'. Accepted extensions: {', '.join(cfg.supported_extensions)}.",
            )

    return (True, None, None)

