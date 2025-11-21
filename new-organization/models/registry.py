from typing import Dict, Callable, Optional, List
import warnings
import torch.nn as nn
from .configs.base_config import BaseModelConfig

class ModelRegistry:
    """
    Central registry for all models.
    Maps model names to factory functions.
    """
    _registry: Dict[str, Callable] = {}
    _config_classes: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, factory: Callable, config_class: Optional[type] = None):
        """
        Register a model factory function.
        
        Args:
            name: Model name (e.g., "LSTM", "TimesNet")
            factory: Function that creates the model: factory(config) -> model
            config_class: Optional config class for this model
        """
        name_upper = name.upper()
        if name_upper in cls._registry:
            warnings.warn(f"Model '{name_upper}' already registered. Overwriting.")
        cls._registry[name_upper] = factory
        if config_class:
            cls._config_classes[name_upper] = config_class
    
    @classmethod
    def create(cls, name: str, config: BaseModelConfig) -> nn.Module:
        """
        Create a model instance by name.
        
        Args:
            name: Model name (case-insensitive)
            config: Model configuration object
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model name not found
        """
        name_upper = name.upper()
        if name_upper not in cls._registry:
            available = ", ".join(cls.list_models())
            raise ValueError(
                f"Model '{name}' not found. Available models: {available}"
            )
        
        factory = cls._registry[name_upper]
        return factory(config)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """Return list of all registered model names."""
        return list(cls._registry.keys())
    
    @classmethod
    def get_config_class(cls, name: str) -> Optional[type]:
        """Get the config class for a model."""
        return cls._config_classes.get(name.upper())


def register_model(name: str, config_class: Optional[type] = None):
    """
    Decorator to register a model factory function.
    
    Usage:
        @register_model("LSTM", LSTMConfig)
        def create_lstm(config):
            return LSTMModel(config)
    """
    def decorator(factory: Callable):
        ModelRegistry.register(name, factory, config_class)
        return factory
    return decorator


def create_model(name: str, config: BaseModelConfig) -> nn.Module:
    """
    Convenience function to create a model.
    
    Usage:
        model = create_model("LSTM", lstm_config)
    """
    return ModelRegistry.create(name, config)