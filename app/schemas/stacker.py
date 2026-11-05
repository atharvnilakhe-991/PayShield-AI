"""
Out-Of-Fold (OOF) Stacker Model Contract for Phase 6.
Defines input and output schemas for combining predictions from the CNN branch
and the OCR/Semantic branch into a unified, calibrated manipulation probability.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from pydantic import Field
from app.schemas.common import BaseSchema


class CalibrationMetadata(BaseSchema):
    """
    Metadata capturing probability calibration parameters and meta-model decision details.
    Allows downstream auditing of how final probabilities were combined and calibrated.
    """
    calibration_method: Optional[str] = Field(
        default=None,
        description="Calibration strategy applied: 'platt_scaling', 'isotonic_regression', 'temperature_scaling'"
    )
    cnn_weight: Optional[float] = Field(
        default=None,
        description="Ensemble weight or regression coefficient assigned to the CNN branch"
    )
    semantic_weight: Optional[float] = Field(
        default=None,
        description="Ensemble weight or regression coefficient assigned to the semantic branch"
    )
    uncertainty_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Model disagreement or prediction entropy metric [0.0 = High consensus, 1.0 = High disagreement]"
    )
    decision_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Configured classification threshold calibrated against validation data"
    )


class StackerInput(BaseSchema):
    """
    Standard input payload for the OOF Stacker meta-learner.
    Combines the first-stage probabilities and telemetry from both specialized branches.
    """
    cnn_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Manipulation probability score produced by CNN branch [0.0 to 1.0]"
    )
    semantic_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Manipulation probability score produced by OCR/Semantic branch [0.0 to 1.0]"
    )
    additional_branch_scores: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional auxiliary scores (e.g. {'ela_score': 0.72, 'exif_integrity_score': 0.15})"
    )
    feature_vector: Optional[List[float]] = Field(
        default=None,
        description="Optional flattened feature vector from the semantic engine if meta-learner is a tree or neural net"
    )
    cnn_model_version: str = Field(
        ...,
        description="Version string of the CNN model providing cnn_score"
    )
    semantic_model_version: str = Field(
        ...,
        description="Version string of the semantic engine providing semantic_score"
    )


class StackerOutput(BaseSchema):
    """
    Standard output produced by the OOF Stacker.
    Supplies the unified final manipulation probability and calibration provenance.
    """
    final_manipulation_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final ensembled manipulation probability [0.0 = Authentic, 1.0 = Manipulated]"
    )
    calibrated_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated confidence score representing certainty in the assigned verdict [0.0 to 1.0]"
    )
    stacker_model_version: str = Field(
        ...,
        description="Version identifier of the trained stacker model (e.g. 'oof_stacker_lgbm_v1.0')"
    )
    decision_metadata: Optional[CalibrationMetadata] = Field(
        default=None,
        description="Calibration parameters and ensemble weighting telemetry"
    )
    inference_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock time taken by stacker forward pass in milliseconds"
    )


class StackerInferenceInterface(ABC):
    """
    Abstract contract for the OOF Stacker model implementation in Phase 6.
    Any concrete ensemble stacker must implement this interface.
    """
    @abstractmethod
    def predict(self, input_data: StackerInput) -> StackerOutput:
        """
        Combines branch inputs and returns calibrated final probability and metadata.
        
        Args:
            input_data: Validated StackerInput with CNN and Semantic outputs.
            
        Returns:
            StackerOutput with final combined probability and version provenance.
        """
        pass

