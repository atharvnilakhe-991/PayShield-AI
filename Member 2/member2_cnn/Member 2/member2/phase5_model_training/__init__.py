"""
CBL-04: Tile Forensics CNN Model Training Package
Phase 5 - Member 2 (Tile Forensics CNN & Heatmap Lead)
"""

from .split_dataset import GroupSplitter, DatasetSplit, SplitConfig
from .model import TileResNet18, SimpleTileCNN, get_model
from .training_dataset import TileForensicsDataset, compute_class_weights
from .train import Trainer, TrainingResult
from .evaluate import Evaluator, EvaluationResult

__all__ = [
    "GroupSplitter",
    "DatasetSplit",
    "SplitConfig",
    "TileResNet18",
    "SimpleTileCNN",
    "get_model",
    "TileForensicsDataset",
    "compute_class_weights",
    "Trainer",
    "TrainingResult",
    "Evaluator",
    "EvaluationResult",
]

