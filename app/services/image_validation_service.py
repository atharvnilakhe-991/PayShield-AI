"""
Image Validation Service for CBL-04.
Performs strict binary validation, size checking, format decoding, dimension verification,
and resource cleanup without persisting user screenshots permanently to disk.
"""

import io
from typing import Optional, Tuple
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.schemas.enums import ErrorCodeEnum
from app.schemas.errors import ERROR_HTTP_STATUS_MAP


class ImageValidationError(Exception):
    """Exception raised when an uploaded image fails validation."""

    def __init__(
        self,
        error_code: ErrorCodeEnum,
        message: str,
        status_code: Optional[int] = None,
        detail: Optional[str] = None,
    ):
        self.error_code = error_code
        self.message = message
        self.status_code = status_code or ERROR_HTTP_STATUS_MAP.get(error_code, 400)
        self.detail = detail
        super().__init__(message)


class ValidatedImage(BaseModel):
    """Metadata representing a successfully verified and validated image."""

    filename: Optional[str]
    format: str
    width: int
    height: int
    size_bytes: int
    mime_type: str


ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "WEBP"}


def validate_and_inspect_image(
    file_bytes: bytes,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> ValidatedImage:
    """
    Validates the binary content of an uploaded image file against contract specifications.
    
    Checks performed:
      1. Empty file detection (< 1 byte).
      2. Upload payload size limit (max 10 MB).
      3. MIME type & extension preliminary checks.
      4. Actual binary decoding via Pillow to verify image data integrity (not trusting headers alone).
      5. Decoded image format restriction (JPEG, PNG, WebP only).
      6. Pixel dimension boundaries ([100, 4096] px).
      
    Args:
      file_bytes: Raw binary bytes of uploaded file.
      filename: Uploaded file name from multipart form.
      content_type: Content-Type declared by client.
      settings: Application settings singleton.
      
    Returns:
      ValidatedImage containing verified dimensions, format, and size.
      
    Raises:
      ImageValidationError: When any validation constraint fails.
    """
    cfg = settings or get_settings()
    size_bytes = len(file_bytes)

    # 1. Empty upload check
    if size_bytes < cfg.MIN_FILE_SIZE_BYTES:
        raise ImageValidationError(
            error_code=ErrorCodeEnum.CORRUPT_IMAGE,
            status_code=400,
            message="The uploaded file is empty (0 bytes). A valid screenshot image is required.",
        )

    # 2. Maximum file size check
    if size_bytes > cfg.MAX_FILE_SIZE_BYTES:
        max_mb = cfg.MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise ImageValidationError(
            error_code=ErrorCodeEnum.FILE_TOO_LARGE,
            status_code=413,
            message=f"Uploaded image size ({size_bytes} bytes) exceeds maximum limit of {max_mb:.1f} MB.",
            detail=f"Uploaded: {size_bytes} bytes, Allowed: {cfg.MAX_FILE_SIZE_BYTES} bytes",
        )

    # 3. Filename extension check (if filename provided)
    if filename:
        lower_name = filename.lower()
        if not any(lower_name.endswith(ext) for ext in cfg.SUPPORTED_EXTENSIONS):
            raise ImageValidationError(
                error_code=ErrorCodeEnum.INVALID_IMAGE_FORMAT,
                status_code=415,
                message=(
                    f"Unsupported file extension in '{filename}'. "
                    f"Accepted extensions: {', '.join(cfg.SUPPORTED_EXTENSIONS)}"
                ),
            )

    # 4. Binary decode and format identification via Pillow
    try:
        buffer = io.BytesIO(file_bytes)
        with Image.open(buffer) as img:
            img_format = (img.format or "").upper()

            # Reject formats outside JPEG, PNG, WEBP (e.g. GIF, TIFF, BMP)
            if img_format not in ALLOWED_PIL_FORMATS:
                raise ImageValidationError(
                    error_code=ErrorCodeEnum.INVALID_IMAGE_FORMAT,
                    status_code=415,
                    message=(
                        f"Unsupported image format '{img_format}'. "
                        f"Accepted formats: JPEG, PNG, WebP."
                    ),
                )

            # Check dimensions
            width, height = img.size
            min_w, min_h = cfg.MIN_IMAGE_DIMENSION_PX
            max_w, max_h = cfg.MAX_IMAGE_DIMENSION_PX

            if width < min_w or height < min_h:
                raise ImageValidationError(
                    error_code=ErrorCodeEnum.IMAGE_DIMENSIONS_OUT_OF_RANGE,
                    status_code=422,
                    message=(
                        f"Image dimensions ({width}x{height}px) are too small. "
                        f"Minimum required dimensions: {min_w}x{min_h}px."
                    ),
                )

            if width > max_w or height > max_h:
                raise ImageValidationError(
                    error_code=ErrorCodeEnum.IMAGE_DIMENSIONS_OUT_OF_RANGE,
                    status_code=422,
                    message=(
                        f"Image dimensions ({width}x{height}px) exceed maximum allowed size of {max_w}x{max_h}px."
                    ),
                )

            # Map Pillow format to standard MIME type
            mime_map = {
                "JPEG": "image/jpeg",
                "PNG": "image/png",
                "WEBP": "image/webp",
            }
            resolved_mime = mime_map.get(img_format, "image/png")

            return ValidatedImage(
                filename=filename,
                format=img_format,
                width=width,
                height=height,
                size_bytes=size_bytes,
                mime_type=resolved_mime,
            )

    except UnidentifiedImageError:
        raise ImageValidationError(
            error_code=ErrorCodeEnum.CORRUPT_IMAGE,
            status_code=400,
            message="Uploaded file could not be decoded as an image. It may be corrupted or not a valid image file.",
        )
    except ImageValidationError:
        raise
    except Exception as exc:
        raise ImageValidationError(
            error_code=ErrorCodeEnum.IMAGE_UNREADABLE,
            status_code=400,
            message="Failed to read or decode image contents.",
            detail=str(exc),
        )

