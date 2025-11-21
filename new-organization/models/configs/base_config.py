class BaseModelConfig:
    """
    Base configuration class for all models.
    
    All model configs inherit from this base class and must include parameters.
    Parameters is a dictionary of parameters for the model.
    """
    def __init__(self, parameters=None):
        """
        Initialize base model configuration.
        
        Args:
            parameters: Dictionary of parameters for the model.
        """
        if parameters is None:
            raise ValueError("parameters is required")
        self.parameters = parameters
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def __repr__(self):
        """String representation of config."""
        attrs = ', '.join(f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith('_'))
        return f"{self.__class__.__name__}({attrs})"