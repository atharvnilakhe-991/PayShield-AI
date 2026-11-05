"""
CNN Model Interface Contract for Member 2 integration.
Defines the required inputs, output schemas, spatial evidence structures,
and abstract contract for future CNN inference without altering API specifications.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any
from pydantic import Field
from app.schemas.common import (
    BaseSchema,
    ModelVersionInfo,
    NormalizedBoundingBox,
    PixelBoundingBox,
)


class CNNSuspiciousRegion(BaseSchema):
    """
    Localized suspicious region detected by CNN spatial activation / Grad-CAM / ELA analysis.
    Supports both normalized coordinates for resolution invariance and pixel bounding boxes.
    """
    label: str = Field(
        ...,
        description="Descriptive identifier of the localized anomaly (e.g. 'Amount field', 'Timestamp baseline')"
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Localized anomaly / manipulation confidence score [0.0 to 1.0]"
    )
    bbox_normalized: NormalizedBoundingBox = Field(
        ...,
        description="Normalized bounding box relative to image width and height [0.0 to 1.0]"
    )
    bbox_pixel: Optional[PixelBoundingBox] = Field(
        default=None,
        description="Absolute pixel coordinates (x, y, width, height) relative to original image"
    )
    anomaly_type: Optional[str] = Field(
        default=None,
        description="Artifact classification: 'splicing', 'resampling', 'font_tamper', 'ela_compression'"
    )


class CNNModelInput(BaseSchema):
    """
    Standardized input passed to Member 2's CNN inference pipeline.
    Accommodates raw image bytes, in-memory buffers, or filesystem paths.
    """
    image_bytes: Optional[bytes] = Field(
        default=None,
        description="Raw image file binary bytes"
    )
    image_path: Optional[str] = Field(
        default=None,
        description="Optional absolute filesystem path to pre-saved temporary image"
    )
    image_dimensions: Optional[Tuple[int, int]] = Field(
        default=None,
        description="(width, height) of the original unscaled image"
    )
    generate_heatmap: bool = Field(
        default=True,
        description="Whether to generate Grad-CAM / ELA spatial localization heatmap"
    )
    device: Optional[str] = Field(
        default="cpu",
        description="Compute target for inference: 'cpu', 'cuda', 'mps'"
    )


class CNNModelOutput(BaseSchema):
    """
    Standardized output produced by Member 2's CNN model.
    Must be returned by any CNN model integrated into the CBL-04 pipeline.
    """
    manipulation_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="CNN branch probability score that the image is manipulated [0.0 = Authentic, 1.0 = Manipulated]"
    )
    raw_logits: Optional[List[float]] = Field(
        default=None,
        description="Unnormalized class logits from final layer (e.g. [logit_authentic, logit_manipulated])"
    )
    heatmap_data_uri: Optional[str] = Field(
        default=None,
        description="Base64 Data URI of generated Grad-CAM/ELA heatmap ('data:image/png;base64,...')"
    )
    heatmap_relative_path: Optional[str] = Field(
        default=None,
        description="Relative static URL or path to persisted heatmap (e.g. 'static/heatmaps/<id>.png')"
    )
    suspicious_regions: List[CNNSuspiciousRegion] = Field(
        default_factory=list,
        description="Bounding boxes around highest gradient activation or anomalous pixel clusters"
    )
    model_info: ModelVersionInfo = Field(
        default_factory=lambda: ModelVersionInfo(
            model_name="CNN_ManipDetector",
            model_version="v1.0.0",
            architecture="Pending_Member2",
        ),
        description="Model architecture, checkpoint hash, and version provenance"
    )
    inference_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="CNN forward pass wall-clock execution time in milliseconds"
    )
    score_type: str = Field(
        default="uncalibrated_model_score",
        description="Score interpretation: raw softmax/sigmoid score, not a calibrated probability"
    )
    localization_available: bool = Field(
        default=False,
        description="Whether localized spatial bounding boxes/heatmaps are available and reliable"
    )


class CNNInferenceInterface(ABC):
    """
    Abstract contract for Member 2's CNN model integration.
    Any concrete CNN wrapper in Phase 6 must inherit from this interface.
    """
    @abstractmethod
    def predict(self, input_data: CNNModelInput) -> CNNModelOutput:
        """
        Executes CNN inference and spatial evidence generation on the provided image input.
        
        Args:
            input_data: Validated CNNModelInput containing image payload and inference parameters.
            
        Returns:
            CNNModelOutput containing manipulation probability, heatmap, and suspicious regions.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> ModelVersionInfo:
        """Returns the version, architecture, and weights provenance of the loaded model."""
        pass

