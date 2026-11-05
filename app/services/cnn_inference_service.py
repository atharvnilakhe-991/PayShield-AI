"""
CNN Inference Service for CBL-04 Google Pay Screenshot Manipulation Detection.
Integrates Member 2's PyTorch multi-task ResNet-18 model into the FastAPI backend.
Adheres strictly to Phase 1 CNNInferenceInterface and CNNModelOutput contracts.
"""

from __future__ import annotations

import base64
import io
import logging
import sys
import threading
import time
from hashlib import sha256
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from app.core.config import get_settings
from app.schemas.cnn import (
    CNNInferenceInterface,
    CNNModelInput,
    CNNModelOutput,
)
from app.schemas.common import ModelVersionInfo

logger = logging.getLogger("cbl04.cnn")

# Dynamically locate and register Hackathon-VIT model script path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MODEL_SCRIPT_DIR = WORKSPACE_ROOT / "Hackathon-VIT" / "CBL04-Dataset" / "scripts" / "phase7"
if str(MODEL_SCRIPT_DIR) not in sys.path and MODEL_SCRIPT_DIR.exists():
    sys.path.insert(0, str(MODEL_SCRIPT_DIR))

try:
    from model import build_model, GPayTamperModel
except ImportError as err:
    logger.error("Failed to import GPayTamperModel from %s: %s", MODEL_SCRIPT_DIR, err)
    build_model = None
    GPayTamperModel = None


class CNNModelError(Exception):
    """Base exception for CNN model initialization and execution errors."""
    pass


class CNNModelUnavailableError(CNNModelError):
    """Raised when CNN checkpoint is missing, invalid, or cannot be loaded."""
    pass


class CNNInferenceService(CNNInferenceInterface):
    """
    Production-ready CNN inference service wrapping the trained ResNet-18 multi-task model.
    Loads checkpoint into memory once at initialization and reuses across API requests.
    """

    TARGET_HEIGHT = 320
    TARGET_WIDTH = 160
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    BACKGROUND_RGB = (18, 18, 18)

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        auto_load: bool = True,
    ) -> None:
        settings = get_settings()
        raw_path = checkpoint_path or settings.CNN_CHECKPOINT_PATH
        self.checkpoint_path = self._resolve_path(raw_path)
        
        # Device resolution
        requested_device = device or settings.CNN_DEVICE
        if requested_device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model: Optional[torch.nn.Module] = None
        self.model_config: Optional[dict] = None
        self.weights_hash: Optional[str] = None
        self.checkpoint_epoch: Optional[int] = None
        self.is_loaded: bool = False

        # Precompute normalization tensors
        self._mean = torch.tensor(self.IMAGENET_MEAN, device=self.device).view(3, 1, 1)
        self._std = torch.tensor(self.IMAGENET_STD, device=self.device).view(3, 1, 1)

        if auto_load:
            self.load()

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = (WORKSPACE_ROOT / p).resolve()
        return p

    def load(self) -> None:
        """Loads and initializes the PyTorch model from the configured checkpoint."""
        if build_model is None:
            raise CNNModelUnavailableError(
                f"Model architecture module could not be imported from {MODEL_SCRIPT_DIR}"
            )

        if not self.checkpoint_path.exists() or not self.checkpoint_path.is_file():
            raise CNNModelUnavailableError(
                f"CNN model checkpoint not found at {self.checkpoint_path}"
            )

        logger.info("Loading CNN checkpoint from %s on %s...", self.checkpoint_path, self.device)
        try:
            # Read file bytes to compute SHA256 provenance hash safely
            ckpt_bytes = self.checkpoint_path.read_bytes()
            self.full_weights_hash = sha256(ckpt_bytes).hexdigest()
            self.weights_hash = self.full_weights_hash[:16]

            # Secure loading with weights_only=True (safe PyTorch unpickler)
            # Checked in Phase 3 audit: primitive dicts and state_dict load cleanly without unrestricted execution
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )

            if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
                raise CNNModelError(
                    f"Invalid checkpoint format in {self.checkpoint_path.name}: missing 'state_dict'"
                )

            self.model_config = checkpoint.get("model_config", {})
            self.checkpoint_epoch = checkpoint.get("epoch")

            # Build architecture from saved config
            self.model = build_model(self.model_config)
            missing, unexpected = self.model.load_state_dict(checkpoint["state_dict"], strict=True)
            if missing or unexpected:
                logger.warning(
                    "State dict mismatch: missing=%s, unexpected=%s", missing, unexpected
                )

            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info(
                "Successfully loaded CNN model [%s, %s] on %s",
                self.checkpoint_path.name,
                self.weights_hash,
                self.device,
            )
        except Exception as exc:
            self.is_loaded = False
            self.model = None
            logger.error("Failed to load CNN model: %s", exc)
            raise CNNModelUnavailableError(f"Failed to load CNN model: {exc}") from exc

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Executes verified aspect-preserving resize with dark padding and ImageNet normalization.
        
        Args:
            image: Decoded PIL image.
            
        Returns:
            Preprocessed input tensor of shape [1, 3, 320, 160] on configured device.
        """
        rgb_image = image.convert("RGB")
        orig_w, orig_h = rgb_image.size

        # Aspect-preserving scale calculation
        scale = min(self.TARGET_WIDTH / orig_w, self.TARGET_HEIGHT / orig_h)
        new_w = max(1, round(orig_w * scale))
        new_h = max(1, round(orig_h * scale))

        resized = rgb_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (self.TARGET_WIDTH, self.TARGET_HEIGHT), self.BACKGROUND_RGB)
        canvas.paste(resized, (0, 0))

        # Tensor conversion and normalization
        tensor = torch.frombuffer(
            bytearray(canvas.tobytes()), dtype=torch.uint8
        ).reshape(self.TARGET_HEIGHT, self.TARGET_WIDTH, 3)
        
        # Permute from [H, W, C] to [C, H, W] on device
        tensor = tensor.to(self.device).permute(2, 0, 1).float() / 255.0
        normalized = (tensor - self._mean) / self._std
        return normalized.unsqueeze(0)

    def predict(self, input_data: CNNModelInput) -> CNNModelOutput:
        """
        Executes CNN forward pass on input screenshot and returns standardized CNNModelOutput.
        """
        if not self.is_loaded or self.model is None:
            raise CNNModelUnavailableError("CNN model is not loaded or currently unavailable.")

        start_time = time.perf_counter()

        # Resolve image from bytes or path
        image: Optional[Image.Image] = None
        if input_data.image_bytes:
            try:
                image = Image.open(io.BytesIO(input_data.image_bytes))
            except (UnidentifiedImageError, OSError) as exc:
                raise CNNModelError(f"Could not decode image bytes: {exc}") from exc
        elif input_data.image_path:
            img_path = Path(input_data.image_path)
            if not img_path.exists():
                raise CNNModelError(f"Image path does not exist: {input_data.image_path}")
            image = Image.open(img_path)
        else:
            raise CNNModelError("CNNModelInput must supply either image_bytes or image_path.")

        try:
            x = self.preprocess(image)

            with torch.inference_mode():
                output = self.model(x)
                logits_tensor = output["logits"][0]  # Shape: [2]
                scores = torch.softmax(logits_tensor, dim=0)

                raw_clean_score = float(scores[0].item())
                raw_edited_score = float(scores[1].item())
                raw_logits = [float(logits_tensor[0].item()), float(logits_tensor[1].item())]

            # Generate real neural activation heatmap from U-Net decoder mask_logits
            heatmap_data_uri = None
            if "mask_logits" in output:
                try:
                    mask_t = torch.sigmoid(output["mask_logits"][0, 0]).cpu().numpy()
                    orig_w, orig_h = image.size
                    mask_img = Image.fromarray((mask_t * 255.0).astype(np.uint8)).resize(
                        (orig_w, orig_h), Image.Resampling.BILINEAR
                    )
                    mask_norm = np.array(mask_img).astype(np.float32) / 255.0

                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    cm = plt.get_cmap("inferno")
                    heatmap_colored = (cm(mask_norm)[:, :, :3] * 255.0).astype(np.uint8)
                    im_rgb = np.array(image.convert("RGB"))
                    blended = (0.5 * im_rgb.astype(np.float32) + 0.5 * heatmap_colored.astype(np.float32)).astype(np.uint8)

                    buf = io.BytesIO()
                    Image.fromarray(blended).save(buf, format="JPEG", quality=85)
                    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
                    heatmap_data_uri = f"data:image/jpeg;base64,{b64_str}"
                except Exception as h_exc:
                    logger.warning("Could not generate heatmap data URI: %s", h_exc)

            duration_ms = (time.perf_counter() - start_time) * 1000.0

            return CNNModelOutput(
                manipulation_probability=round(raw_edited_score, 6),
                raw_logits=[round(l, 6) for l in raw_logits],
                heatmap_data_uri=heatmap_data_uri,
                heatmap_relative_path=None,
                suspicious_regions=[],
                score_type="uncalibrated_model_score",
                localization_available=bool(heatmap_data_uri),
                model_info=self.get_model_info(),
                inference_time_ms=round(duration_ms, 2),
            )
        finally:
            if image:
                image.close()

    def get_model_info(self) -> ModelVersionInfo:
        """Returns model version metadata and architecture provenance."""
        ckpt_name = self.checkpoint_path.name if self.checkpoint_path else "unknown"
        return ModelVersionInfo(
            model_name="GPayTamperModel_ResNet18",
            model_version=f"v0.7.0-{ckpt_name}",
            architecture="ResNet-18 Multitask (Binary Classifier + Mask Decoder)",
            weights_hash=self.weights_hash or "unhashed",
            description=(
                "Experimental multi-task CNN baseline for Google Pay dark receipts. "
                "Raw score is an uncalibrated model score, not a verified probability. "
                "Mask localization is disabled due to experimental IoU (0.059)."
            ),
        )


# Global singleton instance cache and thread lock
_CNN_SERVICE_LOCK = threading.Lock()
_CNN_SERVICE_INSTANCE: Optional[CNNInferenceService] = None


def get_cnn_service() -> CNNInferenceService:
    """Returns application-level cached singleton of CNNInferenceService with thread safety."""
    global _CNN_SERVICE_INSTANCE
    if _CNN_SERVICE_INSTANCE is None:
        with _CNN_SERVICE_LOCK:
            if _CNN_SERVICE_INSTANCE is None:
                _CNN_SERVICE_INSTANCE = CNNInferenceService()
    return _CNN_SERVICE_INSTANCE


def reset_cnn_service() -> None:
    """Resets cached CNN service singleton (used in testing) with thread safety."""
    global _CNN_SERVICE_INSTANCE
    with _CNN_SERVICE_LOCK:
        _CNN_SERVICE_INSTANCE = None

