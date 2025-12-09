"""
Base classes for loss functions.
"""

import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any


class BaseLoss(ABC):
    """
    Abstract base class for all loss functions.
    
    Loss functions can optionally use intermediate layer outputs from the model.
    """
    
    @abstractmethod
    def __call__(self, predictions, targets, intermediates: Optional[Dict[str, Any]] = None):
        """
        Compute loss.
        
        Args:
            predictions: Model predictions tensor
            targets: Target values tensor
            intermediates: Optional dictionary of intermediate layer outputs
                         Keyed by layer name/path
        
        Returns:
            Loss tensor (scalar)
        """
        raise NotImplementedError("Subclasses must implement __call__")
    
    def requires_intermediates(self) -> bool:
        """
        Whether this loss function requires intermediate layer outputs.
        
        Returns:
            True if loss needs intermediates, False otherwise
        """
        return False
    
    def get_hook_targets(self) -> List[str]:
        """
        Get list of layer names/paths that need to be hooked.
        
        Returns:
            List of layer names/paths (e.g., ["encoder", "decoder", "AE.encoder"])
        """
        return []


class SimpleLoss(BaseLoss):
    """
    Wrapper for standard PyTorch loss functions that don't need intermediates.
    
    Usage:
        loss_fn = SimpleLoss(nn.MSELoss())
        loss = loss_fn(predictions, targets)
    """
    
    def __init__(self, loss_fn: nn.Module):
        """
        Initialize SimpleLoss wrapper.
        
        Args:
            loss_fn: PyTorch loss function (e.g., nn.MSELoss(), nn.CrossEntropyLoss())
        """
        super().__init__()
        self.loss_fn = loss_fn
    
    def __call__(self, predictions, targets, intermediates: Optional[Dict[str, Any]] = None):
        """
        Compute loss using wrapped PyTorch loss function.
        
        Args:
            predictions: Model predictions tensor
            targets: Target values tensor
            intermediates: Ignored (for compatibility)
        
        Returns:
            Loss tensor
        """
        return self.loss_fn(predictions, targets)
    
    def requires_intermediates(self) -> bool:
        """Simple losses don't require intermediates."""
        return False
