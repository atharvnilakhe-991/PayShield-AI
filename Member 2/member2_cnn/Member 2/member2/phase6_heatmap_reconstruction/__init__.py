"""
CBL-04: Heatmap Reconstruction & Image-Level Localization Package
Phase 6 - Member 2 (Tile Forensics CNN & Heatmap Lead)
"""

from .reconstructor import HeatmapReconstructor, generate_blending_window
from .inference import TileInferencer
from .thresholding import ThresholdOptimizer, ThresholdOptimizationResult
from .evaluator import Phase6Evaluator, Phase6EvaluationResult, ImageEvaluationRecord
from .visualizer import Phase6Visualizer, create_heatmap_overlay, create_error_overlay

__all__ = [
    "HeatmapReconstructor",
    "generate_blending_window",
    "TileInferencer",
    "ThresholdOptimizer",
    "ThresholdOptimizationResult",
    "Phase6Evaluator",
    "Phase6EvaluationResult",
    "ImageEvaluationRecord",
    "Phase6Visualizer",
    "create_heatmap_overlay",
    "create_error_overlay",
]

