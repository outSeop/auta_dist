"""
손실 함수 모듈
"""

from .single_class_loss import SingleClassDistillationLoss
from .feature_alignment_loss import FeatureAlignmentLoss

__all__ = [
    "SingleClassDistillationLoss",
    "FeatureAlignmentLoss"
]
