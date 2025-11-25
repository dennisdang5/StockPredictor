from ..base import BaseModel
from ..configs import AELSTMConfig, AutoEncoderConfig, LSTMConfig
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
        # Create AutoEncoderConfig from AELSTMConfig for the AutoEncoder component
        ae_config_dict = model_config.to_dict()
        ae_config = AutoEncoderConfig(parameters={'input_shape': ae_config_dict.get('input_shape', (31, 3))})
        self.AE = AutoEncoder(model_config=ae_config)
        # Create LSTMConfig from AELSTMConfig for the LSTM component
        lstm_config = LSTMConfig(parameters={
            'input_shape': ae_config_dict.get('input_shape', (31, 3)),
            'hidden_size': ae_config_dict.get('hidden_size', 25),
            'num_layers': ae_config_dict.get('num_layers', 1),
            'batch_first': ae_config_dict.get('batch_first', True),
            'dropout': ae_config_dict.get('dropout', 0.1)
        })
        self.LSTM = LSTMModel(model_config=lstm_config)

    def forward(self, x, params=None):
        x = self.AE(x)
        x = self.LSTM(x)
        return x

    @classmethod
    def from_config(cls, model_config):
        """Factory hook so the registry can instantiate the model."""
        return cls(model_config)


# Register the model with the central registry
from ..registry import ModelRegistry
ModelRegistry.register("AELSTM", lambda config: AELSTM(config), AELSTMConfig)