from ..base import BaseModel
from ..configs import LSTMConfig
import torch
import torch.nn as nn
from torch.nn import init


class LSTMModel(BaseModel):
    """
    LSTM Model for time series prediction.
    
    Args:
        model_config: Configuration object containing model parameters.
                     Must have the following attributes (all optional with defaults):
    
    Expected model_config attributes:
        - input_shape (tuple): Shape of input data (timesteps, features). 
                              Format: (sequence_length, num_features)
                              Default: (31, 3)
        - hidden_size (int): Hidden dimension of LSTM layers. 
                            Controls the size of hidden state and cell state.
                            Default: 25
        - num_layers (int): Number of stacked LSTM layers. 
                          More layers = deeper model, but slower training.
                          Default: 1
        - batch_first (bool): If True, input/output tensors are (batch, seq, feature).
                             If False, tensors are (seq, batch, feature).
                             Default: True
        - dropout (float): Dropout rate applied after LSTM layers (0.0 to 1.0).
                          Higher values = more regularization.
                          Default: 0.1
    """
    def __init__(self, model_config=None):
        super().__init__(model_config)
        if model_config is None:
            raise ValueError("model_config is required for LSTMModel")
        
        # Type checking: ensure model_config is LSTMConfig or compatible
        if not isinstance(model_config, LSTMConfig):
            raise TypeError(
                f"LSTMModel requires LSTMConfig instance, "
                f"got {type(model_config).__name__}. "
                f"Use LSTMConfig(input_shape=..., hidden_size=..., etc.) to create the config."
            )
        
        # Extract all parameters from model_config with defaults
        self.input_shape = model_config.to_dict().get('input_shape', (31, 3))
        self.hidden_size = model_config.to_dict().get('hidden_size', 25)
        self.num_layers = model_config.to_dict().get('num_layers', 1)
        self.batch_first = model_config.to_dict().get('batch_first', True)
        self.dropout = model_config.to_dict().get('dropout', 0.1)
        
        self.input_dim = self.input_shape[1]
        
        # Input normalization layer - helps stabilize inputs
        # Normalizes across features at each time step
        self.input_norm = nn.LayerNorm(self.input_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dtype=torch.float32)
        
        # Layer normalization after LSTM - stabilizes activations and gradients
        self.lstm_norm = nn.LayerNorm(self.hidden_size)
        
        self.dropout = nn.Dropout(p=self.dropout)
        self.linear = nn.Linear(self.hidden_size, 1)
        # Note: No normalization after final output layer - output should be in natural scale
        
        # Initialize weights properly to prevent NaN
        self._initialize_weights()
    
    @classmethod
    def from_config(cls, model_config):
        """Create model instance from config."""
        return cls(model_config)
    
    def _initialize_weights(self):
        """Initialize weights with proper scaling to prevent NaN/exploding gradients."""
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                # Input-to-hidden weights: use Xavier uniform
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Hidden-to-hidden weights: use orthogonal initialization (better for RNNs)
                init.orthogonal_(param.data)
            elif 'bias' in name:
                # Initialize biases to zero, except forget gate bias (set to 1 for better gradient flow)
                param.data.fill_(0)
                # LSTM has 4 gates: input, forget, cell, output
                # Set forget gate bias to 1 (helps with gradient flow)
                n = param.size(0)
                start, end = n // 4, n // 2  # Forget gate is second quarter
                param.data[start:end].fill_(1)
        
        # Initialize linear layer weights
        init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        # Normalize input features at each time step
        # This helps stabilize training even if input normalization varies
        x = self.input_norm(x)
        
        # Pass through LSTM
        x, _ = self.lstm(x)
        
        # Normalize LSTM output (applied to last time step only)
        # This stabilizes the hidden state before the final linear layer
        x = self.lstm_norm(x[:,-1,:])
        
        # Apply dropout for regularization
        x = self.dropout(x)
        
        # Final linear layer (no normalization - output should be in natural scale)
        x = self.linear(x)
        return x


# Register the model
from ..registry import ModelRegistry
ModelRegistry.register("LSTM", lambda config: LSTMModel(config), LSTMConfig)