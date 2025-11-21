import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models in the system.
    All models (custom and external) must inherit from this.
    """
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
    
    @abstractmethod
    def forward(self, x, params=None):
        """Forward pass - must be implemented by all models."""
        raise NotImplementedError("Subclasses must implement forward method")
    
    def get_config(self):
        """Return the model's configuration."""
        return self.model_config
    
    @classmethod
    @abstractmethod
    def from_config(cls, model_config):
        """Create model instance from config - must be implemented."""
        raise NotImplementedError("Subclasses must implement from_config method")