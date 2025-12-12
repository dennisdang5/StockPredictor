"""
Loss function module for model-specific loss functions.

Supports both simple losses (standard PyTorch losses) and complex losses
that require intermediate layer outputs.
"""

from .base_loss import BaseLoss, SimpleLoss
from .registry import LossRegistry, get_loss
from .intermediate_losses import ClassificationLoss, AutoEncoderLoss, AELSTMLoss, CAELSTMLoss, CompositeLoss

__all__ = [
    'BaseLoss',
    'SimpleLoss',
    'LossRegistry',
    'get_loss',
    'ClassificationLoss',
    'AutoEncoderLoss',
    'AELSTMLoss',
    'CAELSTMLoss',
    'CompositeLoss',
]
