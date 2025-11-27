from ..base import BaseModel
from ..configs import CNNAELSTMConfig, CNNAutoEncoderConfig, LSTMConfig
from .cnn_autoencoder import CNNAutoEncoder
from .lstm import LSTMModel
import torch
import torch.nn as nn

class CNNAELSTM(BaseModel):
    """
    CNN-AutoEncoder-LSTM Model combining CNN autoencoder feature extraction with LSTM prediction.
    
    Architecture: CNN-AutoEncoder extracts features from input (processing first 11
    and last 20 timesteps separately), then LSTM processes the encoded features for final prediction.
    
    Args:
        model_config: Configuration object containing model parameters.
                     Must have the following attributes (all optional with defaults):
    
    Expected model_config attributes:
        - input_shape (tuple): Shape of input data (timesteps, features).
                              Format: (sequence_length, num_features)
                              Default: (31, 3)
        - kernel_size (int): Size of CNN convolution kernel (must be odd for padding).
                            Controls receptive field of convolution in CNN-AutoEncoder.
                            Default: 3
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
    def __init__(self, model_config) -> None:
        super(CNNAELSTM, self).__init__(model_config)
        if model_config is None:
            raise ValueError("model_config is required for CNNAELSTM")
        
        # Type checking: ensure model_config is CNNAELSTMConfig or compatible
        if not isinstance(model_config, CNNAELSTMConfig):
            # Allow SimpleNamespace for backward compatibility but warn
            raise TypeError(
                f"CNNAELSTM requires CNNAELSTMConfig instance, "
                f"got {type(model_config).__name__}. "
                f"Use CNNAELSTMConfig(input_shape=..., kernel_size=..., etc.) to create the config."
            )
        self.model_config = model_config
        self.input_shape = model_config.to_dict().get('input_shape', (31, 3))
        self.kernel_size = model_config.to_dict().get('kernel_size', 3)
        
        # Create CNNAutoEncoderConfig from CNNAELSTMConfig for the CNNAutoEncoder component
        cnn_ae_config = model_config.cnn_ae_config
        lstm_config = model_config.lstm_config
        self.CNNAE = CNNAutoEncoder(model_config=cnn_ae_config)
        self.LSTM = LSTMModel(model_config=lstm_config)

    def forward(self, x, params=None):
        x = self.CNNAE(x)
        x = self.LSTM(x)
        return x

    @classmethod
    def from_config(cls, model_config):
        """Factory hook so the registry can instantiate the model."""
        return cls(model_config)

# Register the model
from ..registry import ModelRegistry
ModelRegistry.register("CNNAELSTM", lambda config: CNNAELSTM(config), CNNAELSTMConfig)