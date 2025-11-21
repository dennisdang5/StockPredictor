"""
External model adapters.
Importing this module registers all external model wrappers.
"""

# Import external model adapters
# This triggers their @register_model decorators
from . import timesnet
__all__ = ['timesnet']