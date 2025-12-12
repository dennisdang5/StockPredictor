"""
Loss function registry for managing and creating loss functions.
"""

from typing import Dict, Callable, Optional, Any
import warnings
import torch.nn as nn
from .base_loss import BaseLoss, SimpleLoss
from .intermediate_losses import ClassificationLoss, AutoEncoderLoss, AELSTMLoss, CAELSTMLoss, CompositeLoss


class LossRegistry:
    """
    Central registry for loss functions.
    Maps loss names to factory functions or loss instances.
    """
    _registry: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str, factory: Callable):
        """
        Register a loss function factory.
        
        Args:
            name: Loss name (e.g., "mse", "autoencoder_loss")
            factory: Function that creates the loss: factory(**kwargs) -> BaseLoss
        """
        name_lower = name.lower()
        if name_lower in cls._registry:
            warnings.warn(f"Loss '{name_lower}' already registered. Overwriting.")
        cls._registry[name_lower] = factory
    
    @classmethod
    def get_loss(cls, loss_name: str, loss_kwargs: Optional[Dict[str, Any]] = None) -> BaseLoss:
        """
        Get a loss function by name.
        
        Args:
            loss_name: Name of the loss function
            loss_kwargs: Optional keyword arguments to pass to loss factory
        
        Returns:
            BaseLoss instance
        
        Raises:
            ValueError: If loss name not found
        """
        if loss_kwargs is None:
            loss_kwargs = {}
        
        name_lower = loss_name.lower()
        if name_lower not in cls._registry:
            available = ", ".join(cls.list_losses())
            raise ValueError(
                f"Loss '{loss_name}' not found. Available losses: {available}"
            )
        
        factory = cls._registry[name_lower]
        return factory(**loss_kwargs)
    
    @classmethod
    def list_losses(cls):
        """Return list of all registered loss names."""
        return list(cls._registry.keys())


# Register default losses
def _create_mse(**kwargs):
    """Create MSE loss."""
    return SimpleLoss(nn.MSELoss())

def _create_mae(**kwargs):
    """Create MAE loss."""
    return SimpleLoss(nn.L1Loss())

def _create_crossentropy(**kwargs):
    """Create CrossEntropy loss."""
    return SimpleLoss(nn.CrossEntropyLoss())

def _create_bce(**kwargs):
    """Create BCE loss."""
    return SimpleLoss(nn.BCELoss())

def _create_bce_with_logits(**kwargs):
    """Create BCEWithLogits loss."""
    return SimpleLoss(nn.BCEWithLogitsLoss())

# Register default losses
LossRegistry.register("mse", _create_mse)
LossRegistry.register("mae", _create_mae)
LossRegistry.register("l1", _create_mae)  # Alias
LossRegistry.register("crossentropy", _create_crossentropy)
LossRegistry.register("bce", _create_bce)
LossRegistry.register("bce_with_logits", _create_bce_with_logits)

# Register intermediate-aware losses
LossRegistry.register("classification_loss", lambda **kwargs: ClassificationLoss(**kwargs))
LossRegistry.register("autoencoder_loss", lambda **kwargs: AutoEncoderLoss(**kwargs))
LossRegistry.register("ae_loss", lambda **kwargs: AutoEncoderLoss(**kwargs))  # Alias
LossRegistry.register("aelstm_loss", lambda **kwargs: AELSTMLoss(**kwargs))
LossRegistry.register("caelstm_loss", lambda **kwargs: CAELSTMLoss(**kwargs))
LossRegistry.register("composite_loss", lambda **kwargs: CompositeLoss(**kwargs))


def get_loss(loss_name: str, loss_kwargs: Optional[Dict[str, Any]] = None) -> BaseLoss:
    """
    Convenience function to get a loss function.
    
    Usage:
        loss_fn = get_loss("mse")
        loss_fn = get_loss("autoencoder_loss", {"reconstruction_weight": 0.5})
    """
    return LossRegistry.get_loss(loss_name, loss_kwargs)
