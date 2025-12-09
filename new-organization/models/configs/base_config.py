class BaseModelConfig:
    """
    Base configuration class for all models.
    
    All model configs inherit from this base class and must include parameters.
    Parameters is a dictionary of parameters for the model.
    
    Optional loss configuration can be provided via parameters['loss_config']:
        loss_config = {
            'loss_name': str,  # e.g., "mse", "autoencoder_loss"
            'loss_kwargs': dict,  # e.g., {"reconstruction_weight": 0.5}
            'intermediate_layers': list  # e.g., ["encoder", "decoder"]
        }
    """
    def __init__(self, parameters=None):
        """
        Initialize base model configuration.
        
        Args:
            parameters: Dictionary of parameters for the model.
                        Can optionally include 'loss_config' for model-specific loss functions.
        """
        if parameters is None:
            raise ValueError("parameters is required")
        self.parameters = parameters
        
        # Extract loss_config if provided
        loss_config = parameters.get('loss_config', None)
        if loss_config is not None:
            self.loss_config = loss_config
        else:
            self.loss_config = None
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def __repr__(self):
        """String representation of config."""
        attrs = ', '.join(f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith('_'))
        return f"{self.__class__.__name__}({attrs})"