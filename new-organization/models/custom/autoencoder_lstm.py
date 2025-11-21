from ..base import BaseModel
from ..configs import AELSTMConfig
import torch
import torch.nn as nn
from .autoencoder import AutoEncoder
from .lstm import LSTMModel

class AELSTM(BaseModel):
    """
    AutoEncoder-LSTM Model combining autoencoder feature extraction with LSTM prediction.
    
    Architecture: AutoEncoder extracts features from input, then LSTM processes
    the encoded features for final prediction.
    
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
    def __init__(self, model_config):
        super(AELSTM, self).__init__(model_config)
        if model_config is None:
            raise ValueError("model_config is required for AELSTM")
        
        # Type checking: ensure model_config is AELSTMConfig or compatible
        if not isinstance(model_config, AELSTMConfig):
            # Allow SimpleNamespace for backward compatibility but warn
            raise TypeError(
                f"AELSTM requires AELSTMConfig instance, "
                f"got {type(model_config).__name__}. "
                f"Use AELSTMConfig(input_shape=..., hidden_size=..., etc.) to create the config."
            )
        
        self.input_shape = model_config.to_dict().get('input_shape', (31, 3))
        self.AE = AutoEncoder(model_config=model_config)
        self.LSTM = LSTMModel(model_config=model_config)

    def forward(self, x):
        x = self.AE(x)
        x = self.LSTM(x)
        return x