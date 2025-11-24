"""
Custom model implementations.
Importing this module registers all custom models.
"""

# Import all custom model modules
# This triggers their @register_model decorators
from . import lstm
from . import autoencoder
from . import autoencoder_lstm
from . import portfolio

__all__ = ['lstm', 'autoencoder', 'autoencoder_lstm', 'portfolio']