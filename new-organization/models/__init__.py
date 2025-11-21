"""
Models package - unified interface for all models.
"""

# Import registry system
from .registry import (
    ModelRegistry,
    register_model,
    create_model,
)

# Import base classes
from .base import BaseModel

# Import configs (will be created in configs/__init__.py)
# This will import all config classes
from .configs import (
    BaseModelConfig,
    LSTMConfig,
    CNNLSTMConfig,
    AELSTMConfig,
    CNNAELSTMConfig,
    TimesNetConfig,
)

# Import custom models (this triggers their @register_model decorators)
from . import custom

# Import external model adapters (this triggers their registration)
from . import external

# Define public API
__all__ = [
    # Registry
    'ModelRegistry',
    'register_model',
    'create_model',
    # Base classes
    'BaseModel',
    'BaseModelConfig',
    # Configs
    'LSTMConfig',
    'CNNLSTMConfig',
    'AELSTMConfig',
    'CNNAELSTMConfig',
    'TimesNetConfig',
    # Convenience functions
    'get_available_models',
    'list_models',
]


def get_available_models():
    """Return list of all registered model names."""
    return ModelRegistry.list_models()


def list_models():
    """Alias for get_available_models()."""
    return get_available_models()