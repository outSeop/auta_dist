"""
YOLOv11 Knowledge Distillation for Single-Class Detection
Optimized for Figma UI Component Detection
"""

from .models.figma_ui_distillation import FigmaUIDistillation
from .losses.single_class_loss import SingleClassDistillationLoss
from .losses.feature_alignment_loss import FeatureAlignmentLoss

__version__ = "1.0.0"
__author__ = "YOLO Distillation Team"

__all__ = [
    "FigmaUIDistillation",
    "SingleClassDistillationLoss", 
    "FeatureAlignmentLoss"
]
