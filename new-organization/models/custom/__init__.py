"""
Custom model implementations.
Importing this module registers all custom models.
"""

# Import all custom model modules
# This triggers their @register_model decorators
from . import lstm
from . import autoencoder

__all__ = ['lstm', 'autoencoder']