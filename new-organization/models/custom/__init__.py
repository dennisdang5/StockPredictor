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
from . import cae_lstm
from . import cnn_autoencoder
__all__ = ['lstm', 'cae_lstm', 'autoencoder', 'autoencoder_lstm', 'portfolio', 'cnn_autoencoder']