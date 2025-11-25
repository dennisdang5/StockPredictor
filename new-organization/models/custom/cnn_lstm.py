from ..base import BaseModel
from ..configs import CNNLSTMConfig, LSTMConfig
from .lstm import LSTMModel
import torch
import torch.nn as nn

class CNNLSTMModel(BaseModel):
    """
    CNN-LSTM Model for time series prediction using short/long convolution method.
    
    Architecture: Processes first 11 and last 20 timesteps separately with CNN,
    then concatenates and passes through LSTM for final prediction.
    
    Args:
        model_config: Configuration object containing model parameters.
                     Must have the following attributes (all optional with defaults):
    
    Expected model_config attributes:
        - input_shape (tuple): Shape of input data (timesteps, features).
                              Format: (sequence_length, num_features)
                              Default: (31, 3)
        - kernel_size (int): Size of CNN convolution kernel (must be odd for padding).
                            Controls receptive field of convolution.
                            Applied to both short (first 11) and long (last 20) sequences.
                            Default: 3
        - hidden_size (int): Hidden dimension of LSTM layers. Default: 25
        - num_layers (int): Number of stacked LSTM layers. Default: 1
        - batch_first (bool): Whether batch is first dimension. Default: True
        - dropout (float): Dropout rate (0.0 to 1.0). Default: 0.1
    """
    def __init__(self, model_config) -> None:
        super(CNNLSTMModel, self).__init__(model_config)
        if model_config is None:
            raise ValueError("model_config is required for CNNLSTMModel")
        
        # Type checking: ensure model_config is CNNLSTMConfig or compatible
        if not isinstance(model_config, CNNLSTMConfig):
            raise TypeError(
                f"CNNLSTMModel requires CNNLSTMConfig instance, "
                f"got {type(model_config).__name__}. "
                f"Use CNNLSTMConfig(input_shape=..., kernel_size=..., etc.) to create the config."
            )
        
        self.model_config = model_config
        self.input_shape = model_config.to_dict().get('input_shape', (31, 3))
        self.kernel_size = model_config.to_dict().get('kernel_size', 3)
        
        self.num_features = self.input_shape[1]  # Can be 3 (price only), 7 (price + 4 NLP), 13 (price + 10 NLP), etc.
        
        # Input normalization
        self.input_norm = nn.LayerNorm(self.num_features)
        
        # Short sequence CNN (first 11 timesteps)
        # Encoder: input features -> 2*input_shape[1] channels
        self.short_enc_conv = nn.Conv1d(self.num_features, 2*self.input_shape[1], self.kernel_size, padding=self.kernel_size//2)
        self.short_enc_norm = nn.LayerNorm(2*self.input_shape[1])
        # Decoder: 2*input_shape[1] channels -> input features
        self.short_dec_conv = nn.Conv1d(2*self.input_shape[1], self.num_features, self.kernel_size, padding=self.kernel_size//2)
        self.short_dec_norm = nn.LayerNorm(self.num_features)
        
        # Long sequence CNN (last 20 timesteps)
        # Encoder: input features -> 2*input_shape[1] channels
        self.long_enc_conv = nn.Conv1d(self.num_features, 2*self.input_shape[1], self.kernel_size, padding=self.kernel_size//2)
        self.long_enc_norm = nn.LayerNorm(2*self.input_shape[1])
        # Decoder: 2*input_shape[1] channels -> input features
        self.long_dec_conv = nn.Conv1d(2*self.input_shape[1], self.num_features, self.kernel_size, padding=self.kernel_size//2)
        self.long_dec_norm = nn.LayerNorm(self.num_features)
        
        # Create LSTM config and model
        lstm_config = LSTMConfig(parameters={
            'input_shape': self.input_shape,
            'hidden_size': model_config.to_dict().get('hidden_size', 25),
            'num_layers': model_config.to_dict().get('num_layers', 1),
            'batch_first': model_config.to_dict().get('batch_first', True),
            'dropout': model_config.to_dict().get('dropout', 0.1)
        })
        self.LSTM = LSTMModel(model_config=lstm_config)

    def forward(self, x, params=None):
        # assume x in shape (batch, 31, num_features) = (batch, time_steps, features)
        # num_features can be 3 (price only), 7 (price + 4 NLP), 13 (price + 10 NLP), etc.
        # Normalize input
        x = self.input_norm(x)
        
        # Process short sequence (first 11 timesteps)
        # Conv1d expects [batch, channels, length], so transpose from [batch, time, features] to [batch, features, time]
        short_enc = self.short_enc_conv(x[:, :11, :].transpose(1, 2))  # [batch, num_features, 11] -> [batch, channels, 11]
        # Transpose to [batch, time, channels] for normalization
        short_enc = short_enc.transpose(1, 2)  # [batch, channels, 11] -> [batch, 11, channels]
        # Normalize after short encoder
        short_enc = self.short_enc_norm(short_enc)
        # Transpose back to [batch, channels, time] for decoder
        short_enc = short_enc.transpose(1, 2)  # [batch, 11, channels] -> [batch, channels, 11]
        
        # Process long sequence (last 20 timesteps)
        long_enc = self.long_enc_conv(x[:, 11:, :].transpose(1, 2))   # [batch, num_features, 20] -> [batch, channels, 20]
        # Transpose to [batch, time, channels] for normalization
        long_enc = long_enc.transpose(1, 2)  # [batch, channels, 20] -> [batch, 20, channels]
        # Normalize after long encoder
        long_enc = self.long_enc_norm(long_enc)
        # Transpose back to [batch, channels, time] for decoder
        long_enc = long_enc.transpose(1, 2)  # [batch, 20, channels] -> [batch, channels, 20]
        
        # Decode short sequence
        short_dec = self.short_dec_conv(short_enc)  # [batch, channels, 11] -> [batch, num_features, 11]
        # Transpose to [batch, time, channels] for normalization
        short_dec = short_dec.transpose(1, 2)  # [batch, num_features, 11] -> [batch, 11, num_features]
        # Normalize after short decoder
        short_dec = self.short_dec_norm(short_dec)
        
        # Decode long sequence
        long_dec = self.long_dec_conv(long_enc)    # [batch, channels, 20] -> [batch, num_features, 20]
        # Transpose to [batch, time, channels] for normalization
        long_dec = long_dec.transpose(1, 2)  # [batch, num_features, 20] -> [batch, 20, num_features] 
        # Normalize after long decoder
        long_dec = self.long_dec_norm(long_dec)
        
        # Concatenate short and long sequences
        x = torch.cat((short_dec, long_dec), dim=1)  # [batch, 31, num_features]
        
        # Pass through LSTM
        x = self.LSTM(x)
        return x

    @classmethod
    def from_config(cls, model_config):
        """Factory hook so the registry can instantiate the model."""
        return cls(model_config)