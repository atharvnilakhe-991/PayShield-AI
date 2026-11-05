"""
Configuration settings for CBL-04 Google Pay Screenshot Manipulation Detection API.
Loads environment variables using Pydantic Settings v2.
"""

from functools import lru_cache
from typing import List, Tuple, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.schemas.request import ImageUploadConfig


class Settings(BaseSettings):
    """Application configuration parameters with environment variable overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # API Metadata
    API_TITLE: str = "CBL-04 Google Pay Screenshot Manipulation Detection API"
    API_DESCRIPTION: str = (
        "FastAPI backend for detecting tampered or fabricated Google Pay receipts "
        "using dual-branch visual CNN and semantic OCR inspection."
    )
    API_VERSION: str = "0.1.0"
    SERVICE_NAME: str = "cbl04-gpay-detection-api"
    ENVIRONMENT: str = "development"

    # CORS Configuration
    # Safe defaults for local frontend development (Vite 5173, React 3000)
    CORS_ORIGINS: Union[List[str], str] = Field(
        default=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        description="Allowed origins for Cross-Origin Resource Sharing (CORS)",
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Union[List[str], str]) -> List[str]:
        if isinstance(v, str):
            if v.strip() == "*":
                return ["*"]
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    # Image Upload & Validation Limits
    MAX_FILE_SIZE_BYTES: int = Field(
        default=10 * 1024 * 1024,  # 10 MB default
        description="Maximum allowed upload payload size in bytes",
    )
    MIN_FILE_SIZE_BYTES: int = Field(
        default=1,
        description="Minimum upload size in bytes to reject empty files",
    )
    MIN_IMAGE_DIMENSION_PX: Tuple[int, int] = Field(
        default=(100, 100),
        description="Minimum (width, height) pixel dimensions",
    )
    MAX_IMAGE_DIMENSION_PX: Tuple[int, int] = Field(
        default=(4096, 4096),
        description="Maximum (width, height) pixel dimensions",
    )
    SUPPORTED_MIME_TYPES: List[str] = Field(
        default=["image/jpeg", "image/png", "image/webp"],
        description="Accepted image MIME content types",
    )
    SUPPORTED_EXTENSIONS: List[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".webp"],
        description="Accepted image file extensions",
    )

    # Pipeline Operation Mode
    MOCK_MODE: bool = Field(
        default=False,
        description="Whether to run in legacy full-mock mode without model inference",
    )
    DEFAULT_VERDICT_STYLE: str = Field(
        default="canonical",
        description="Verdict formatting style: 'canonical' (GENUINE/EDITED/SUSPICIOUS) or 'frontend' (REAL/FAKE/SUSPICIOUS)",
    )

    # CNN Inference Service Configuration
    CNN_ENABLED: bool = Field(
        default=True,
        description="Whether live CNN model inference is enabled in analysis pipeline",
    )
    CNN_CHECKPOINT_PATH: str = Field(
        default="Hackathon-VIT/CBL04-Dataset/models/phase7/fold_1_resnet18_multitask.pt",
        description="Filesystem path to PyTorch CNN model checkpoint (.pt)",
    )
    CNN_DEVICE: str = Field(
        default="cpu",
        description="Compute device target for CNN model ('cpu', 'mps')",
    )
    CNN_DECISION_THRESHOLD: float = Field(
        default=0.50,
        description="Decision threshold for CNN edited classification",
    )

    # OCR Extraction Service Configuration
    OCR_ENABLED: bool = Field(
        default=True,
        description="Whether live OCR extraction is enabled in analysis pipeline",
    )
    OCR_ENGINE: str = Field(
        default="EasyOCR",
        description="OCR engine implementation to use ('EasyOCR')",
    )
    OCR_MODEL_DIR: str = Field(
        default="models/easyocr",
        description="Directory containing EasyOCR offline model weights",
    )
    OCR_GPU: bool = Field(
        default=False,
        description="Whether to use GPU/MPS acceleration for EasyOCR (default: False for deterministic CPU execution)",
    )
    OCR_LANGUAGES: List[str] = Field(
        default_factory=lambda: ["en"],
        description="Language list for OCR text detection (default: ['en'])",
    )

    # Semantic Feature Engine Configuration
    SEMANTIC_ENABLED: bool = Field(
        default=True,
        description="Whether live semantic feature consistency extraction is enabled in analysis pipeline",
    )
    SEMANTIC_ENGINE_VERSION: str = Field(
        default="v1.0.0-phase6",
        description="Semantic consistency rule engine version string",
    )

    def to_image_upload_config(self) -> ImageUploadConfig:
        """Converts settings to Phase 1 ImageUploadConfig schema."""
        return ImageUploadConfig(
            supported_mime_types=self.SUPPORTED_MIME_TYPES,
            supported_extensions=self.SUPPORTED_EXTENSIONS,
            min_file_size_bytes=self.MIN_FILE_SIZE_BYTES,
            max_file_size_bytes=self.MAX_FILE_SIZE_BYTES,
            min_dimension_px=self.MIN_IMAGE_DIMENSION_PX,
            max_dimension_px=self.MAX_IMAGE_DIMENSION_PX,
        )


@lru_cache()
def get_settings() -> Settings:
    """Returns singleton cached instance of application settings."""
    return Settings()

