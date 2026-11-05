"""
Common schema types, coordinate structures, and model metadata for CBL-04.
Establishes standardized coordinate conventions (normalized vs pixel) and model descriptors.
"""

from typing import Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaseSchema(BaseModel):
    """Base schema enforcing strict typing, immutability defaults, and serialization settings."""
    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
        extra="ignore",
    )


class NormalizedBoundingBox(BaseSchema):
    """
    Normalized bounding box relative to original image dimensions.
    Coordinate system:
      - Origin (0.0, 0.0): Top-Left corner of the image.
      - (1.0, 1.0): Bottom-Right corner of the image.
      - Ranges: 0.0 <= xmin <= xmax <= 1.0, 0.0 <= ymin <= ymax <= 1.0.
    """
    xmin: float = Field(..., ge=0.0, le=1.0, description="Normalized X coordinate of left edge [0.0, 1.0]")
    ymin: float = Field(..., ge=0.0, le=1.0, description="Normalized Y coordinate of top edge [0.0, 1.0]")
    xmax: float = Field(..., ge=0.0, le=1.0, description="Normalized X coordinate of right edge [0.0, 1.0]")
    ymax: float = Field(..., ge=0.0, le=1.0, description="Normalized Y coordinate of bottom edge [0.0, 1.0]")

    @model_validator(mode="after")
    def validate_coordinate_ordering(self) -> "NormalizedBoundingBox":
        if self.xmin > self.xmax:
            raise ValueError(f"xmin ({self.xmin}) must be <= xmax ({self.xmax})")
        if self.ymin > self.ymax:
            raise ValueError(f"ymin ({self.ymin}) must be <= ymax ({self.ymax})")
        return self

    @property
    def width(self) -> float:
        """Normalized width in range [0.0, 1.0]."""
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        """Normalized height in range [0.0, 1.0]."""
        return self.ymax - self.ymin

    def to_pixel(self, image_width: int, image_height: int) -> "PixelBoundingBox":
        """Converts normalized coordinates to absolute integer/float pixels."""
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be positive integers.")
        return PixelBoundingBox(
            x=round(self.xmin * image_width, 2),
            y=round(self.ymin * image_height, 2),
            width=round(self.width * image_width, 2),
            height=round(self.height * image_height, 2),
        )


class PixelBoundingBox(BaseSchema):
    """
    Absolute pixel bounding box relative to the original natural image resolution.
    Coordinate system:
      - Origin (0, 0): Top-Left pixel of original image.
      - x: Left pixel offset.
      - y: Top pixel offset.
      - width: Box width in pixels.
      - height: Box height in pixels.
    """
    x: float = Field(..., ge=0.0, description="Top-left X pixel coordinate")
    y: float = Field(..., ge=0.0, description="Top-left Y pixel coordinate")
    width: float = Field(..., ge=0.0, description="Width in pixels")
    height: float = Field(..., ge=0.0, description="Height in pixels")

    @property
    def area(self) -> float:
        """Bounding box area in pixels squared."""
        return self.width * self.height

    def to_normalized(self, image_width: int, image_height: int) -> NormalizedBoundingBox:
        """Converts pixel bounding box to normalized coordinates in range [0.0, 1.0]."""
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be positive integers.")
        xmin = max(0.0, min(1.0, self.x / image_width))
        ymin = max(0.0, min(1.0, self.y / image_height))
        xmax = max(0.0, min(1.0, (self.x + self.width) / image_width))
        ymax = max(0.0, min(1.0, (self.y + self.height) / image_height))
        return NormalizedBoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


class ModelVersionInfo(BaseSchema):
    """Metadata describing a specific model or sub-pipeline component version."""
    model_name: str = Field(..., description="Canonical name of the model component")
    model_version: str = Field(..., description="Semantic version string (e.g. 'v1.2.0')")
    architecture: Optional[str] = Field(default=None, description="Underlying architecture (e.g. 'EfficientNet-B4')")
    weights_hash: Optional[str] = Field(default=None, description="SHA256 checksum or tag of model weights")
    description: Optional[str] = Field(default=None, description="Component description or notes")

