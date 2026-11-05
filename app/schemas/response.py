"""
Final API Response Contract for `/api/v1/analyze` and `/analyze`.
Standardizes the complete verification result, branch breakdown scores,
OCR metadata, localized evidence, model provenance, and legal disclaimers.
"""

from typing import Dict, List, Optional
from pydantic import Field, model_validator
from app.schemas.common import BaseSchema, NormalizedBoundingBox, PixelBoundingBox
from app.schemas.enums import ReasonCodeEnum, VerdictEnum
from app.schemas.ocr import ExtractedTransactionData
from app.schemas.semantic import SemanticBranchOutput


class SuspiciousRegion(BaseSchema):
    """
    Localized suspicious region formatted for both frontend canvas rendering and ML auditing.
    Supports both pixel coordinates for direct UI overlay and normalized coordinates for resolution invariance.
    """
    label: str = Field(..., description="Human-readable label (e.g. 'Amount field', 'Transaction ID alignment')")
    score: float = Field(..., ge=0.0, le=1.0, description="Localized manipulation anomaly score [0.0 to 1.0]")
    x: Optional[float] = Field(default=None, ge=0.0, description="Top-left X coordinate in pixels")
    y: Optional[float] = Field(default=None, ge=0.0, description="Top-left Y coordinate in pixels")
    width: Optional[float] = Field(default=None, ge=0.0, description="Bounding box width in pixels")
    height: Optional[float] = Field(default=None, ge=0.0, description="Bounding box height in pixels")
    box_normalized: Optional[NormalizedBoundingBox] = Field(
        default=None,
        description="Resolution-invariant normalized bounding box [0.0 to 1.0]"
    )


class ManipulationAnalysisEvidence(BaseSchema):
    """
    Spatial explainability and localization evidence container.
    Directly aligns with the frontend ResultsDashboard expectation.
    """
    detected: bool = Field(
        ...,
        description="Whether spatial tampering or anomalous regions were detected in screenshot"
    )
    suspicious_regions: List[SuspiciousRegion] = Field(
        default_factory=list,
        description="List of localized bounding boxes identifying suspected tampered zones"
    )
    heatmap_url: Optional[str] = Field(
        default=None,
        description="Static URL or Base64 Data URI of the generated Grad-CAM/ELA heatmap (null if authentic or disabled)"
    )


class BranchScores(BaseSchema):
    """
    Detailed breakdown of individual model branch predictions prior to / alongside ensemble stacking.
    Provides complete multi-model transparency without hardcoded branch coupling.
    """
    cnn_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Visual / Pixel manipulation probability from CNN branch"
    )
    semantic_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Layout / Text manipulation probability from OCR and semantic branch"
    )
    stacker_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Ensemble probability output from the OOF Stacker meta-learner"
    )
    additional_scores: Optional[Dict[str, float]] = Field(
        default=None,
        description="Auxiliary branch or filter scores (e.g. {'ela_score': 0.65, 'noise_score': 0.32})"
    )


class ModelVersionMetadata(BaseSchema):
    """Complete provenance of all pipeline models and engines involved in generating the response."""
    pipeline_version: str = Field(default="v1.0.0", description="Overall CBL-04 pipeline version")
    cnn_model_version: Optional[str] = Field(default=None, description="Member 2 CNN model architecture and weights version")
    ocr_engine_version: Optional[str] = Field(default=None, description="OCR extraction engine version")
    semantic_engine_version: Optional[str] = Field(default=None, description="Semantic rule engine version")
    stacker_model_version: Optional[str] = Field(default=None, description="OOF ensemble meta-learner version")


class AnalyzeResponse(BaseSchema):
    """
    Standardized API response for the `/api/v1/analyze` endpoint.
    Combines high-level verdict and confidence with explainable OCR metadata,
    localized visual evidence, branch score breakdown, and legal settlement disclaimer.
    """
    request_id: str = Field(
        ...,
        description="Unique request correlation ID for tracing and audit logging"
    )
    verdict: VerdictEnum = Field(
        ...,
        description="Overall authenticity verdict: GENUINE, SUSPICIOUS, EDITED, INCONCLUSIVE"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional heuristic decision margin score in assigned verdict [0.0 to 1.0] for decisive verdicts, or null for INCONCLUSIVE. Uncalibrated heuristic metric; not a statistical probability."
    )
    decision_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Primary uncalibrated decision score from the multi-branch ensemble pipeline [0.0 to 1.0]. Represents continuous decision logic score; not a calibrated probability."
    )
    uncertainty_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Boundary-based normalized ambiguity index [0.0 to 1.0] measuring proximity to decision boundary; 1.0 indicates maximum ambiguity. Populated for indeterminate states."
    )
    manipulation_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Raw ensemble manipulation score from stacker [0.0 to 1.0]; uncalibrated experimental indicator"
    )
    message: str = Field(
        ...,
        description="Human-readable summary of analysis findings"
    )
    ocr_data: ExtractedTransactionData = Field(
        default_factory=ExtractedTransactionData,
        description="Extracted and normalized Google Pay transaction metadata"
    )
    manipulation_analysis: ManipulationAnalysisEvidence = Field(
        ...,
        description="Spatial explainability container with suspicious regions and heatmap URL"
    )
    branch_scores: BranchScores = Field(
        default_factory=BranchScores,
        description="Granular score breakdown from each independent analysis branch"
    )
    # Convenience top-level accessors for branch scores
    cnn_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Convenience top-level access to CNN branch score"
    )
    semantic_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Convenience top-level access to Semantic branch score"
    )
    stacker_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Convenience top-level access to Stacker score"
    )
    semantic_details: Optional[SemanticBranchOutput] = Field(
        default=None,
        description="Detailed semantic consistency feature analysis output and rule-level evidence"
    )
    reason_codes: List[ReasonCodeEnum] = Field(
        default_factory=list,
        description="List of machine-readable structured anomaly identifiers"
    )
    model_versions: ModelVersionMetadata = Field(
        default_factory=ModelVersionMetadata,
        description="Provenance of all participating model and pipeline versions"
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total end-to-end pipeline processing time in milliseconds"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal warnings or limitations encountered during processing (e.g. heavy image compression)"
    )
    pipeline_status: str = Field(
        default="phase3_cnn_integrated",
        description="Current pipeline integration state (e.g. 'phase3_cnn_integrated')"
    )
    component_status: Optional[Dict[str, str]] = Field(
        default_factory=lambda: {
            "cnn": "live",
            "ocr": "mock",
            "semantic": "mock",
            "stacker": "mock"
        },
        description="Operational status of each branch component ('live', 'mock', 'disabled')"
    )
    disclaimer: str = Field(
        default=(
            "Notice: Screenshot analysis performs digital artifact and layout integrity checks only. "
            "It does not verify bank-side settlement, banking ledger entries, or prove that a financial payment occurred."
        ),
        description="Mandatory disclaimer clarifying the scope and limitations of receipt image analysis."
    )

    @model_validator(mode="after")
    def sync_branch_scores(self) -> "AnalyzeResponse":
        """Ensures consistency between branch_scores object and convenience top-level fields."""
        if self.cnn_score is None and self.branch_scores.cnn_score is not None:
            self.cnn_score = self.branch_scores.cnn_score
        elif self.cnn_score is not None and self.branch_scores.cnn_score is None:
            self.branch_scores.cnn_score = self.cnn_score

        if self.semantic_score is None and self.branch_scores.semantic_score is not None:
            self.semantic_score = self.branch_scores.semantic_score
        elif self.semantic_score is not None and self.branch_scores.semantic_score is None:
            self.branch_scores.semantic_score = self.semantic_score

        if self.stacker_score is None and self.branch_scores.stacker_score is not None:
            self.stacker_score = self.branch_scores.stacker_score
        elif self.stacker_score is not None and self.branch_scores.stacker_score is None:
            self.branch_scores.stacker_score = self.stacker_score

        if self.decision_score is None:
            if self.stacker_score is not None:
                self.decision_score = self.stacker_score
            elif self.branch_scores.stacker_score is not None:
                self.decision_score = self.branch_scores.stacker_score
            elif self.manipulation_probability is not None:
                self.decision_score = self.manipulation_probability

        return self

