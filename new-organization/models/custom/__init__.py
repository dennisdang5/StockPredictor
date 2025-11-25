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
from . import cnn_autoencoder_lstm

__all__ = ['lstm', 'autoencoder', 'autoencoder_lstm', 'portfolio', 'cnn_autoencoder_lstm']