"""
CBL-04: Native-Resolution Preprocessing & Transformation Module
Phase 2 - Member 2 (Tile Forensics CNN & Heatmap Lead)

This module provides a reliable, reproducible, native-resolution image and mask
preprocessing pipeline for Google Pay dark-theme screenshot tampering detection.

Key capabilities:
- Native resolution preservation: maintains original canvas width & height (no downsampling or cropping by default).
- Strict RGB standardization: handles RGB, RGBA (dark background compositing), and grayscale (luminance replication).
- Binary mask standardization: exact spatial alignment, background=0, foreground=1 (array) / 255 (PIL).
- Optional robustness transforms (in-memory only, opt-in): mild JPEG recompression (75-95), mild Gaussian blur (0.2-0.6), brightness/contrast.
- Pair consistency: clean and edited pairs receive identical transformation parameters.
- Deterministic reproducibility: seed-controlled transformations for training and testing.
- Seamless compatibility with Phase 1 SingleSample and PairSample structures.
"""

from __future__ import annotations

import io
import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

# Standard Google Pay dark-theme background color used for RGBA compositing (#111111)
DEFAULT_ALPHA_BG_COLOR = (17, 17, 17)


@dataclass
class PreprocessConfig:
    """Configuration for native-resolution preprocessing and robustness transformations."""

    preserve_native_resolution: bool = True
    jpeg_quality: Optional[Union[int, Tuple[int, int]]] = None
    gaussian_sigma: Optional[Union[float, Tuple[float, float]]] = None
    brightness_factor: Optional[Union[float, Tuple[float, float]]] = None
    contrast_factor: Optional[Union[float, Tuple[float, float]]] = None
    alpha_fill_color: Tuple[int, int, int] = DEFAULT_ALPHA_BG_COLOR
    mask_foreground_value: int = 1  # 1 for standard binary array, 0 for background
    normalize_to_float: bool = False  # If True, image array is float32 in [0, 1]; if False, uint8 in [0, 255]
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PreprocessConfig:
        """Instantiate configuration from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, path_or_str: Union[str, Path]) -> PreprocessConfig:
        """Instantiate configuration from JSON file or JSON string."""
        p = Path(path_or_str)
        if p.exists() and p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(str(path_or_str))
        return cls.from_dict(data)


@dataclass
class ProcessedSample:
    """Represents a preprocessed screenshot sample with image and optional mask."""

    image_id: str
    source_id: str
    pair_id: Optional[str]
    label: Optional[int]  # 0: clean, 1: edited, None: unlabeled
    label_name: str  # "clean", "edited", "unlabeled", "unknown"
    sample_type: str  # "synthetic_clean", "synthetic_edited", "real_sensitive", etc.
    original_size: Tuple[int, int]  # (width, height)
    processed_size: Tuple[int, int]  # (width, height)
    image_path: Optional[Path]
    mask_path: Optional[Path]
    image: np.ndarray  # (H, W, 3) uint8 or float32
    mask: Optional[np.ndarray]  # (H, W) uint8 {0, 1} or None for unlabeled
    image_pil: Image.Image  # PIL Image in RGB
    mask_pil: Optional[Image.Image]  # PIL Image in L with {0, 255} or None
    bounding_boxes: List[List[int]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.processed_size[0]

    @property
    def height(self) -> int:
        return self.processed_size[1]

    @property
    def has_mask(self) -> bool:
        return self.mask is not None

    @property
    def tampered_pixel_count(self) -> int:
        if self.mask is None:
            return 0
        return int(np.sum(self.mask > 0))

    @property
    def tampered_pixel_ratio(self) -> float:
        if self.mask is None or self.mask.size == 0:
            return 0.0
        return float(np.sum(self.mask > 0) / self.mask.size)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize sample metadata without raw pixel buffers."""
        return {
            "image_id": self.image_id,
            "source_id": self.source_id,
            "pair_id": self.pair_id,
            "label": self.label,
            "label_name": self.label_name,
            "sample_type": self.sample_type,
            "original_size": self.original_size,
            "processed_size": self.processed_size,
            "image_path": str(self.image_path) if self.image_path else None,
            "mask_path": str(self.mask_path) if self.mask_path else None,
            "has_mask": self.has_mask,
            "tampered_pixel_count": self.tampered_pixel_count,
            "tampered_pixel_ratio": self.tampered_pixel_ratio,
            "bounding_boxes": self.bounding_boxes,
            "metadata": self.metadata,
        }


@dataclass
class ProcessedPair:
    """Represents a matched pair of preprocessed clean and edited screenshots."""

    pair_id: str
    source_id: str
    clean: ProcessedSample
    edited: ProcessedSample
    layout_family: Optional[str] = None
    edit_category: Optional[str] = None
    bounding_boxes: List[List[int]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def verify_alignment(self) -> bool:
        """Verify that clean and edited samples maintain identical dimensions."""
        c_size = self.clean.processed_size
        e_size = self.edited.processed_size
        if c_size != e_size:
            return False
        if self.clean.mask is not None and self.clean.mask.shape != (c_size[1], c_size[0]):
            return False
        if self.edited.mask is not None and self.edited.mask.shape != (e_size[1], e_size[0]):
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize pair metadata."""
        return {
            "pair_id": self.pair_id,
            "source_id": self.source_id,
            "clean": self.clean.to_dict(),
            "edited": self.edited.to_dict(),
            "layout_family": self.layout_family,
            "edit_category": self.edit_category,
            "bounding_boxes": self.bounding_boxes,
            "is_aligned": self.verify_alignment(),
            "metadata": self.metadata,
        }


class Preprocessor:
    """
    Native-resolution preprocessing pipeline for CBL-04 screenshots and tampering masks.

    Usage:
        config = PreprocessConfig(preserve_native_resolution=True, seed=42)
        preprocessor = Preprocessor(config)
        processed_sample = preprocessor.process_sample(sample)
        processed_pair = preprocessor.process_pair(pair)
    """

    def __init__(self, config: Optional[PreprocessConfig] = None) -> None:
        self.config = config or PreprocessConfig()
        self._rng = random.Random(self.config.seed) if self.config.seed is not None else random.Random()

    @staticmethod
    def convert_to_rgb(image: Image.Image, alpha_fill: Tuple[int, int, int] = DEFAULT_ALPHA_BG_COLOR) -> Image.Image:
        """
        Convert any PIL image to standard 3-channel RGB:
        - RGB: preserved as-is.
        - RGBA: composited over alpha_fill (GPay dark background #111111) to eliminate transparency artifacts.
        - L (grayscale): replicated across 3 channels (R=G=B).
        - CMYK / other: converted via Pillow standard converter.
        """
        if image.mode == "RGB":
            return image

        if image.mode == "RGBA":
            # Composite over dark-theme background
            background = Image.new("RGB", image.size, alpha_fill)
            background.paste(image, mask=image.split()[3])  # Alpha channel as mask
            return background

        if image.mode == "L":
            # Grayscale to RGB: replicate luminance across channels
            return image.convert("RGB")

        # Fallback for P, CMYK, etc.
        return image.convert("RGB")

    @staticmethod
    def convert_mask_to_binary(
        mask: Image.Image,
        foreground_val: int = 1,
    ) -> Tuple[np.ndarray, Image.Image]:
        """
        Standardize mask into binary representation:
        - Returns (mask_array, mask_pil)
        - mask_array: uint8 ndarray of shape (H, W) where 0 = background, foreground_val = tampered foreground.
        - mask_pil: PIL Image of mode 'L' where 0 = background, 255 = tampered foreground.
        """
        # Ensure grayscale mode L
        if mask.mode != "L":
            mask_l = mask.convert("L")
        else:
            mask_l = mask

        # Convert to numpy array
        raw_arr = np.array(mask_l, dtype=np.uint8)

        # Strictly threshold: any pixel > 0 is foreground
        binary_mask_arr = np.zeros_like(raw_arr, dtype=np.uint8)
        binary_mask_arr[raw_arr > 0] = foreground_val

        # Display PIL version with 0 and 255
        display_pil = Image.fromarray((binary_mask_arr * (255 // max(1, foreground_val))).astype(np.uint8), mode="L")

        return binary_mask_arr, display_pil

    @staticmethod
    def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
        """Apply mild JPEG compression in-memory without altering source files."""
        if quality < 1 or quality > 100:
            raise ValueError(f"JPEG quality must be between 1 and 100, got {quality}")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer)
        compressed.load()
        return compressed.convert("RGB")

    @staticmethod
    def apply_gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
        """Apply mild Gaussian blur to simulate camera capture or resampling."""
        if sigma <= 0.0:
            return image
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))

    @staticmethod
    def apply_brightness_contrast(
        image: Image.Image,
        brightness: Optional[float] = None,
        contrast: Optional[float] = None,
    ) -> Image.Image:
        """Apply subtle brightness and/or contrast scaling."""
        img = image
        if brightness is not None and abs(brightness - 1.0) > 1e-4:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
        if contrast is not None and abs(contrast - 1.0) > 1e-4:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)
        return img

    def _sample_param(self, param: Optional[Union[float, int, Tuple[Any, Any]]]) -> Optional[Any]:
        """Sample a parameter value from scalar or uniform range."""
        if param is None:
            return None
        if isinstance(param, (list, tuple)) and len(param) == 2:
            low, high = param[0], param[1]
            if isinstance(low, int) and isinstance(high, int):
                return self._rng.randint(low, high)
            return self._rng.uniform(float(low), float(high))
        return param

    def _apply_transforms(
        self,
        image: Image.Image,
        params: Dict[str, Any],
    ) -> Tuple[Image.Image, List[str]]:
        """Apply sampled robustness transformations in a fixed, documented sequence."""
        img = image
        transforms_applied = []

        # 1. Mild brightness / contrast
        b_val = params.get("brightness")
        c_val = params.get("contrast")
        if b_val is not None or c_val is not None:
            img = self.apply_brightness_contrast(img, brightness=b_val, contrast=c_val)
            desc = []
            if b_val is not None:
                desc.append(f"brightness={b_val:.3f}")
            if c_val is not None:
                desc.append(f"contrast={c_val:.3f}")
            transforms_applied.append(f"brightness_contrast({', '.join(desc)})")

        # 2. Mild Gaussian blur
        sigma_val = params.get("gaussian_sigma")
        if sigma_val is not None and sigma_val > 0.0:
            img = self.apply_gaussian_blur(img, sigma=sigma_val)
            transforms_applied.append(f"gaussian_blur(sigma={sigma_val:.3f})")

        # 3. Mild JPEG compression
        jpeg_q = params.get("jpeg_quality")
        if jpeg_q is not None:
            img = self.apply_jpeg_compression(img, quality=int(jpeg_q))
            transforms_applied.append(f"jpeg_compression(quality={int(jpeg_q)})")

        return img, transforms_applied

    def process_sample(
        self,
        sample: Any,
        image_override: Optional[Image.Image] = None,
        mask_override: Optional[Image.Image] = None,
        transform_params_override: Optional[Dict[str, Any]] = None,
    ) -> ProcessedSample:
        """
        Preprocess a SingleSample (or raw image/mask) at native resolution.

        Args:
            sample: SingleSample object, or object with image_path / mask_path attributes.
            image_override: Optional PIL Image to process instead of loading from disk.
            mask_override: Optional PIL mask to process instead of loading from disk.
            transform_params_override: Explicit parameters for transformations.

        Returns:
            ProcessedSample object containing RGB image array and binary mask array.
        """
        # 1. Extract sample attributes
        image_id = getattr(sample, "image_id", "sample")
        source_id = getattr(sample, "source_id", "source")
        pair_id = getattr(sample, "pair_id", None)
        label = getattr(sample, "label", None)
        label_name = getattr(sample, "label_name", "unlabeled" if label is None else ("clean" if label == 0 else "edited"))
        is_synthetic = getattr(sample, "is_synthetic", True)
        sample_type = "synthetic_" + label_name if is_synthetic and label_name in ("clean", "edited") else ("real_sensitive" if not is_synthetic else "unknown")
        image_path = getattr(sample, "image_path", None)
        mask_path = getattr(sample, "mask_path", None)
        bounding_boxes = getattr(sample, "bounding_boxes", [])

        # 2. Load and convert image to RGB
        if image_override is not None:
            raw_img = image_override
        elif hasattr(sample, "load_image"):
            raw_img = sample.load_image()
        elif image_path and Path(image_path).exists():
            with Image.open(image_path) as im:
                raw_img = im.copy()
        else:
            raise ValueError(f"Cannot load image for sample '{image_id}': no valid image source.")

        original_size = raw_img.size
        rgb_img = self.convert_to_rgb(raw_img, alpha_fill=self.config.alpha_fill_color)

        # 3. Load and validate mask
        mask_arr: Optional[np.ndarray] = None
        mask_pil: Optional[Image.Image] = None

        if mask_override is not None:
            raw_mask = mask_override
        elif hasattr(sample, "load_mask"):
            raw_mask = sample.load_mask()
        elif mask_path and Path(mask_path).exists():
            with Image.open(mask_path) as mim:
                raw_mask = mim.copy()
        else:
            raw_mask = None

        if raw_mask is not None:
            # Spatial dimension check: MUST match screenshot
            if raw_mask.size != original_size:
                raise ValueError(
                    f"Dimension mismatch for sample '{image_id}': image is {original_size}, but mask is {raw_mask.size}."
                )
            mask_arr, mask_pil = self.convert_mask_to_binary(
                raw_mask, foreground_val=self.config.mask_foreground_value
            )

        # 4. Determine transformation parameters
        if transform_params_override is not None:
            params = transform_params_override
        else:
            params = {
                "jpeg_quality": self._sample_param(self.config.jpeg_quality),
                "gaussian_sigma": self._sample_param(self.config.gaussian_sigma),
                "brightness": self._sample_param(self.config.brightness_factor),
                "contrast": self._sample_param(self.config.contrast_factor),
            }

        # 5. Apply transformations to image only (never masks!)
        transformed_img, transforms_applied = self._apply_transforms(rgb_img, params)

        # Native resolution check
        if self.config.preserve_native_resolution and transformed_img.size != original_size:
            raise RuntimeError(f"Native resolution violated: expected {original_size}, got {transformed_img.size}")

        # 6. Convert image to numpy array
        np_img = np.array(transformed_img, dtype=np.uint8)
        if self.config.normalize_to_float:
            np_img = (np_img / 255.0).astype(np.float32)

        # 7. Package and return
        meta = {
            "transforms_applied": transforms_applied,
            "transform_params": {k: v for k, v in params.items() if v is not None},
            "seed": self.config.seed,
            "alpha_fill_color": self.config.alpha_fill_color,
            "is_labeled": label is not None,
            "original_image_mode": raw_img.mode,
            "original_mask_mode": raw_mask.mode if raw_mask else None,
        }

        return ProcessedSample(
            image_id=image_id,
            source_id=source_id,
            pair_id=pair_id,
            label=label,
            label_name=label_name,
            sample_type=sample_type,
            original_size=original_size,
            processed_size=transformed_img.size,
            image_path=Path(image_path) if image_path else None,
            mask_path=Path(mask_path) if mask_path else None,
            image=np_img,
            mask=mask_arr,
            image_pil=transformed_img,
            mask_pil=mask_pil,
            bounding_boxes=bounding_boxes,
            metadata=meta,
        )

    def process_pair(self, pair: Any) -> ProcessedPair:
        """
        Preprocess a matched clean-edited PairSample.
        Ensures both clean and edited screenshots receive identical transformation parameters.
        """
        # Sample transformation parameters ONCE for the pair
        pair_params = {
            "jpeg_quality": self._sample_param(self.config.jpeg_quality),
            "gaussian_sigma": self._sample_param(self.config.gaussian_sigma),
            "brightness": self._sample_param(self.config.brightness_factor),
            "contrast": self._sample_param(self.config.contrast_factor),
        }

        # Process clean and edited samples with identical parameters
        clean_processed = self.process_sample(
            pair.clean_sample,
            transform_params_override=pair_params,
        )
        edited_processed = self.process_sample(
            pair.edited_sample,
            transform_params_override=pair_params,
        )

        processed_pair = ProcessedPair(
            pair_id=getattr(pair, "pair_id", clean_processed.pair_id or "pair"),
            source_id=getattr(pair, "source_id", clean_processed.source_id or "source"),
            clean=clean_processed,
            edited=edited_processed,
            layout_family=getattr(pair, "layout_family", None),
            edit_category=getattr(pair, "edit_category", None),
            bounding_boxes=getattr(pair, "bounding_boxes", edited_processed.bounding_boxes),
            metadata={
                "shared_transform_params": {k: v for k, v in pair_params.items() if v is not None},
                "seed": self.config.seed,
            },
        )

        if not processed_pair.verify_alignment():
            raise RuntimeError(
                f"Pair alignment error for '{processed_pair.pair_id}': clean {clean_processed.processed_size} != edited {edited_processed.processed_size}"
            )

        return processed_pair

