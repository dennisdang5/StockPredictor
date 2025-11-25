from ..base import BaseModel
from ..configs import MLPConfig
import torch
import torch.nn as nn
from torch.nn import init


class MLPModel(BaseModel):
    """
    MLP Model for time series prediction.
    
    Args:
        model_config: Configuration object containing model parameters.
                     Must have the following attributes (all optional with defaults):
    
    Expected model_config attributes:
        - input_dim (int): Dimension of input data.
        - hidden_dims (list[int]): List of hidden dimensions.
        - activation (str): Activation name supported by torch.nn (default: "relu").
        - dropout (float): Dropout applied between MLP layers.
    """
    def __init__(self, model_config=None):
        super().__init__(model_config)
        if model_config is None:
            raise ValueError("model_config is required for MLPModel")
        
        # Type checking: ensure model_config is LSTMConfig or compatible
        if not isinstance(model_config, MLPConfig):
            raise TypeError(
                f"MLPModel requires MLPConfig instance, "
                f"got {type(model_config).__name__}. "
                f"Use MLPConfig(input_dim=..., hidden_dims=..., etc.) to create the config."
            )
        
        # Extract all parameters from model_config with defaults
        self.input_dim = model_config.to_dict().get('input_dim')
        self.hidden_dims = model_config.to_dict().get('hidden_dims', [self.input_dim//2])
        self.activation = model_config.to_dict().get('activation', 'relu')
        self.dropout = model_config.to_dict().get('dropout', 0.1)
        self.output_dim = model_config.to_dict().get('output_dim', 1)
        
        # MLP layers
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        # Build hidden layers: connect each hidden dim to the next
        for i in range(len(self.hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
            self.hidden_layers.append(getattr(nn, self.activation.capitalize(), nn.ReLU)())
            self.hidden_layers.append(nn.LayerNorm(self.hidden_dims[i+1]))
            self.hidden_layers.append(nn.Dropout(p=self.dropout))
        
        self.output_layer = nn.Linear(self.hidden_dims[-1], self.output_dim)
        
        # Initialize weights properly
        self._initialize_weights()
        
    @classmethod
    def from_config(cls, model_config):
        """Create model instance from config."""
        return cls(model_config)
    
    def _initialize_weights(self):
        """Initialize weights with proper scaling to prevent NaN/exploding gradients."""
        # Initialize input layer weights
        init.xavier_uniform_(self.input_layer.weight.data)
        self.input_layer.bias.data.fill_(0)
        
        # Initialize hidden layer weights
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight.data)
                layer.bias.data.fill_(0)
        
        # Initialize output layer weights
        init.xavier_uniform_(self.output_layer.weight.data)
        self.output_layer.bias.data.fill_(0)

    def forward(self, x, params=None):
        # MLP acts as a classifier on outputs from individual models (e.g., portfolio architecture)
        # Input x should be feature vectors from individual model outputs, not raw time series data
        # Flatten input if it's 3D (batch, seq_len, features) -> (batch, seq_len * features)
        if x.dim() == 3:
            batch_size = x.size(0)
            x = x.view(batch_size, -1)  # Flatten sequence dimension
        
        # Pass through input layer
        x = self.input_layer(x)
        x = getattr(nn, self.activation.capitalize(), nn.ReLU)()(x)
        
        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Final output layer (no normalization - output should be in natural scale)
        x = self.output_layer(x)
        return x


# Register the model
from ..registry import ModelRegistry
ModelRegistry.register("MLP", lambda config: MLPModel(config), MLPConfig)