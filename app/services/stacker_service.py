"""
Out-Of-Fold (OOF) Stacker Inference Service for CBL-04.
Combines predictions from CNN, OCR, and Semantic consistency branches into a
unified manipulation probability using an interpretable L2-regularized logistic stacker.
Preserves missing branch signals explicitly without fabricating data.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from app.core.config import get_settings
from app.schemas.cnn import CNNModelOutput
from app.schemas.ocr import OCRResult
from app.schemas.semantic import SemanticBranchOutput
from app.schemas.stacker import CalibrationMetadata, StackerInferenceInterface, StackerInput, StackerOutput

logger = logging.getLogger("cbl04.stacker")

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

FEATURE_SCHEMA: List[str] = [
    "cnn_score",
    "cnn_pred_class",
    "ocr_overall_confidence",
    "ocr_amount_extracted",
    "ocr_recipient_extracted",
    "ocr_ref_id_extracted",
    "ocr_date_extracted",
    "semantic_manipulation_score",
    "amount_anomaly",
    "recipient_anomaly",
    "date_anomaly",
    "ref_id_anomaly",
    "status_anomaly",
    "upi_handle_anomaly",
    "missing_critical_fields_count",
    "branch_cnn_available",
    "branch_ocr_available",
    "branch_semantic_available",
]


class StackerService(StackerInferenceInterface):
    """
    Production OOF Stacker service implementing transparent L2 logistic meta-regression.
    Reads trained weights, scaling parameters, and feature schemas from JSON (no unsafe pickle).
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        auto_load: bool = True,
    ) -> None:
        self.model_path = Path(model_path) if model_path else WORKSPACE_ROOT / "models" / "phase8" / "stacker_model.json"
        self.is_loaded: bool = False
        self.model_version: str = "v1.0.0-phase8"
        self.feature_schema: List[str] = list(FEATURE_SCHEMA)
        self.weights: Dict[str, float] = {}
        self.scaler: Dict[str, Dict[str, float]] = {}
        self.intercept: float = 0.0
        self.operating_threshold: float = 0.50
        self.calibration_status: str = "uncalibrated"
        self.l2_reg: float = 0.01

        if auto_load:
            self.load()

    def load(self) -> None:
        """Loads trained model weights and scaling parameters from JSON artifact."""
        if self.model_path.exists():
            try:
                with open(self.model_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self.model_version = data.get("stacker_version", "v1.0.0-phase8")
                self.feature_schema = data.get("feature_schema", list(FEATURE_SCHEMA))
                self.intercept = float(data.get("intercept", 0.0))
                self.weights = {k: float(v) for k, v in data.get("weights", {}).items()}
                self.scaler = data.get("scaler", {})
                self.operating_threshold = float(data.get("operating_threshold", 0.50))
                self.calibration_status = data.get("calibration_status", "uncalibrated")
                self.l2_reg = float(data.get("l2_regularization", 0.01))
                self.is_loaded = True
                logger.info("Stacker model loaded successfully from %s (version: %s)", self.model_path, self.model_version)
                return
            except Exception as exc:
                logger.warning("Failed to parse stacker model JSON at %s: %s. Using default neutral weights.", self.model_path, exc)

        # Fallback neutral weights if model file not yet written or during testing
        self.model_version = "v1.0.0-phase8-fallback"
        self.intercept = -0.5
        self.weights = {col: 0.0 for col in self.feature_schema}
        self.weights["cnn_score"] = 2.0
        self.weights["semantic_manipulation_score"] = 1.5
        self.scaler = {col: {"mean": 0.0, "std": 1.0} for col in self.feature_schema}
        self.is_loaded = True
        logger.info("Initialized stacker service with baseline weights (model file not found at %s)", self.model_path)

    def extract_feature_vector(
        self,
        cnn_result: Optional[CNNModelOutput] = None,
        ocr_result: Optional[OCRResult] = None,
        semantic_result: Optional[SemanticBranchOutput] = None,
    ) -> Tuple[np.ndarray, Dict[str, float], Dict[str, bool]]:
        """
        Extracts and standardizes the input vector from live branch results.
        Returns scaled feature array, raw feature dict, and branch availability flags.
        """
        raw_features: Dict[str, float] = {}
        branch_avail: Dict[str, bool] = {
            "cnn": False,
            "ocr": False,
            "semantic": False,
        }

        # 1. CNN Branch
        if cnn_result is not None and getattr(cnn_result, "execution_status", "live") in ["live", "evaluated"]:
            cnn_score = float(getattr(cnn_result, "manipulation_probability", 0.5))
            cnn_score = max(0.0, min(1.0, cnn_score))
            cnn_pred = 1.0 if cnn_score >= 0.5 else 0.0
            cnn_avail = 1.0
            branch_avail["cnn"] = True
        else:
            cnn_score = 0.5  # Neutral prior when CNN unavailable
            cnn_pred = 0.0
            cnn_avail = 0.0

        raw_features["cnn_score"] = cnn_score
        raw_features["cnn_pred_class"] = cnn_pred
        raw_features["branch_cnn_available"] = cnn_avail

        # 2. OCR Branch
        if ocr_result is not None and getattr(ocr_result, "ocr_engine", None) is not None:
            ocr_conf = float(getattr(ocr_result, "overall_confidence", 0.0))
            ocr_conf = max(0.0, min(1.0, ocr_conf))
            fields = getattr(ocr_result, "fields", None)
            has_amt = 1.0 if fields and (fields.amount is not None or fields.amount_numeric is not None) else 0.0
            has_rec = 1.0 if fields and (fields.receiver is not None or fields.receiver_upi is not None) else 0.0
            has_ref = 1.0 if fields and (fields.transaction_id is not None or fields.google_txn_id is not None) else 0.0
            has_date = 1.0 if fields and (fields.date is not None or fields.time is not None) else 0.0
            ocr_avail = 1.0
            branch_avail["ocr"] = True
        else:
            ocr_conf = 0.0
            has_amt, has_rec, has_ref, has_date = 0.0, 0.0, 0.0, 0.0
            ocr_avail = 0.0

        raw_features["ocr_overall_confidence"] = ocr_conf
        raw_features["ocr_amount_extracted"] = has_amt
        raw_features["ocr_recipient_extracted"] = has_rec
        raw_features["ocr_ref_id_extracted"] = has_ref
        raw_features["ocr_date_extracted"] = has_date
        raw_features["branch_ocr_available"] = ocr_avail

        # 3. Semantic Branch
        if semantic_result is not None and getattr(semantic_result, "semantic_status", "live") in ["live", "evaluated"]:
            sem_score = float(getattr(semantic_result, "semantic_manipulation_score", 0.0))
            sem_score = max(0.0, min(1.0, sem_score))
            anom = getattr(semantic_result, "field_anomalies", None)
            amt_anom = 1.0 if anom and getattr(anom, "amount_anomaly", False) else 0.0
            rec_anom = 1.0 if anom and getattr(anom, "recipient_anomaly", False) else 0.0
            date_anom = 1.0 if anom and getattr(anom, "date_anomaly", False) else 0.0
            ref_anom = 1.0 if anom and getattr(anom, "ref_id_anomaly", False) else 0.0
            stat_anom = 1.0 if anom and getattr(anom, "status_anomaly", False) else 0.0
            upi_anom = 1.0 if anom and getattr(anom, "upi_handle_anomaly", False) else 0.0

            fv = getattr(semantic_result, "feature_vector", None)
            missing_cnt = float(getattr(fv, "missing_critical_fields_count", 0)) / 4.0 if fv else 0.0
            sem_avail = 1.0
            branch_avail["semantic"] = True
        else:
            sem_score = 0.0
            amt_anom, rec_anom, date_anom, ref_anom, stat_anom, upi_anom = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            missing_cnt = 1.0  # 4 / 4 missing when semantic branch failed
            sem_avail = 0.0

        raw_features["semantic_manipulation_score"] = sem_score
        raw_features["amount_anomaly"] = amt_anom
        raw_features["recipient_anomaly"] = rec_anom
        raw_features["date_anomaly"] = date_anom
        raw_features["ref_id_anomaly"] = ref_anom
        raw_features["status_anomaly"] = stat_anom
        raw_features["upi_handle_anomaly"] = upi_anom
        raw_features["missing_critical_fields_count"] = missing_cnt
        raw_features["branch_semantic_available"] = sem_avail

        # Apply Standardization
        scaled_array = np.zeros(len(self.feature_schema), dtype=np.float64)
        for i, col in enumerate(self.feature_schema):
            val = raw_features.get(col, 0.0)
            sc = self.scaler.get(col, {"mean": 0.0, "std": 1.0})
            mean = sc.get("mean", 0.0)
            std = sc.get("std", 1.0)
            std = std if std > 1e-6 else 1.0
            scaled_array[i] = (val - mean) / std

        return scaled_array, raw_features, branch_avail

    def predict_branches(
        self,
        cnn_result: Optional[CNNModelOutput] = None,
        ocr_result: Optional[OCRResult] = None,
        semantic_result: Optional[SemanticBranchOutput] = None,
    ) -> StackerOutput:
        """
        Executes stacker forward pass combining live branch outputs.
        Computes feature contributions for transparent interpretability.
        """
        start_time = time.perf_counter()

        scaled_vec, raw_feats, branch_avail = self.extract_feature_vector(
            cnn_result=cnn_result,
            ocr_result=ocr_result,
            semantic_result=semantic_result,
        )

        # Compute dot product and sigmoid
        z = self.intercept
        for i, col in enumerate(self.feature_schema):
            w = self.weights.get(col, 0.0)
            z += w * scaled_vec[i]

        prob = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
        prob = round(prob, 4)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        metadata = CalibrationMetadata(
            calibration_method=None,  # Explicitly uncalibrated
            cnn_weight=round(self.weights.get("cnn_score", 0.0), 4),
            semantic_weight=round(self.weights.get("semantic_manipulation_score", 0.0), 4),
            uncertainty_score=round(1.0 - abs(prob - 0.5) * 2.0, 4),
            decision_threshold=self.operating_threshold,
        )

        return StackerOutput(
            final_manipulation_probability=prob,
            calibrated_confidence=round(abs(prob - 0.5) * 2.0, 4),
            stacker_model_version=self.model_version,
            decision_metadata=metadata,
            inference_time_ms=round(elapsed_ms, 2),
        )

    def predict(self, input_data: StackerInput) -> StackerOutput:
        """
        Implements abstract StackerInferenceInterface contract.
        Wraps StackerInput into dummy branch results for compatibility.
        """
        start_time = time.perf_counter()
        cnn_score = input_data.cnn_score
        sem_score = input_data.semantic_score

        # Approximate forward pass from 2 main scores
        w_cnn = self.weights.get("cnn_score", 2.0)
        w_sem = self.weights.get("semantic_manipulation_score", 1.5)
        z = self.intercept + w_cnn * cnn_score + w_sem * sem_score
        prob = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
        prob = round(prob, 4)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        metadata = CalibrationMetadata(
            calibration_method=None,
            cnn_weight=round(w_cnn, 4),
            semantic_weight=round(w_sem, 4),
            uncertainty_score=round(1.0 - abs(prob - 0.5) * 2.0, 4),
            decision_threshold=self.operating_threshold,
        )

        return StackerOutput(
            final_manipulation_probability=prob,
            calibrated_confidence=round(abs(prob - 0.5) * 2.0, 4),
            stacker_model_version=self.model_version,
            decision_metadata=metadata,
            inference_time_ms=round(elapsed_ms, 2),
        )


# Thread-safe Singleton Management
_STACKER_INSTANCE: Optional[StackerService] = None
_STACKER_LOCK = threading.Lock()


def get_stacker_service() -> StackerService:
    """Returns thread-safe singleton instance of StackerService."""
    global _STACKER_INSTANCE
    if _STACKER_INSTANCE is None:
        with _STACKER_LOCK:
            if _STACKER_INSTANCE is None:
                _STACKER_INSTANCE = StackerService()
    return _STACKER_INSTANCE


def reset_stacker_service() -> None:
    """Resets singleton instance for testing and dynamic reloading."""
    global _STACKER_INSTANCE
    with _STACKER_LOCK:
        _STACKER_INSTANCE = None
