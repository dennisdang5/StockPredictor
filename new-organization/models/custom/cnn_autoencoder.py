from ..base import BaseModel
from ..configs import CNNAutoEncoderConfig
import torch
import torch.nn as nn

class CNNAutoEncoder(BaseModel):
    """
    CNN-based AutoEncoder for feature extraction using convolutional layers.
    
    Architecture: Processes first 11 and last 20 timesteps separately with CNN,
    encodes to 2x channels, then decodes back to original feature dimensions.
    
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
    """
    def __init__(self, model_config):
        super(CNNAutoEncoder,self).__init__(model_config)
        if model_config is None:
            raise ValueError("model_config is required for CNNAutoEncoder")
        
        # Type checking: ensure model_config is CNNAutoEncoderConfig or compatible
        if not isinstance(model_config, CNNAutoEncoderConfig):
            # Allow SimpleNamespace for backward compatibility but warn
            raise TypeError(
                f"CNNAutoEncoder requires CNNAutoEncoderConfig instance, "
                f"got {type(model_config).__name__}. "
                f"Use CNNAutoEncoderConfig(input_shape=..., kernel_size=...) to create the config."
            )
        
        self.input_shape = model_config.to_dict().get('input_shape', (31, 3))
        self.kernel_size = model_config.to_dict().get('kernel_size', 3)
        
        self.num_features = self.input_shape[1]  # Can be 3 (price only), 7 (price + 4 NLP), 13 (price + 10 NLP), etc.
        # Input normalization
        self.input_norm = nn.LayerNorm(self.num_features)
        # Encoder: input features -> 2*input_shape[1] channels
        self.short_enc_conv = nn.Conv1d(self.num_features, 2*self.input_shape[1], self.kernel_size, padding=self.kernel_size//2)
        # Normalization after short encoder conv (normalize across channels)
        self.short_enc_norm = nn.LayerNorm(2*self.input_shape[1])
        self.long_enc_conv = nn.Conv1d(self.num_features, 2*self.input_shape[1], self.kernel_size, padding=self.kernel_size//2)
        # Normalization after long encoder conv (normalize across channels)
        self.long_enc_norm = nn.LayerNorm(2*self.input_shape[1])

        # Decoder: 2*input_shape[1] channels -> input features
        self.short_dec_conv = nn.Conv1d(2*self.input_shape[1], self.num_features, self.kernel_size, padding=self.kernel_size//2)
        # Normalization after short decoder conv (normalize across channels)
        self.short_dec_norm = nn.LayerNorm(self.num_features)
        self.long_dec_conv = nn.Conv1d(2*self.input_shape[1], self.num_features, self.kernel_size, padding=self.kernel_size//2)
        # Normalization after long decoder conv (normalize across channels)
        self.long_dec_norm = nn.LayerNorm(self.num_features)

        ###### other parameters ####
        # nn.MSELoss()
        # optimizer = torch.optim.Adam()

    def forward(self, x):
        # assume x in shape (batch, 31, num_features) = (batch, time_steps, features)
        # num_features can be 3 (price only), 7 (price + 4 NLP), 13 (price + 10 NLP), etc.
        # Normalize input
        x = self.input_norm(x)
        # Conv1d expects [batch, channels, length], so transpose from [batch, time, features] to [batch, features, time]
        short_enc = self.short_enc_conv(x[:, :11, :].transpose(1, 2))  # [batch, num_features, 11] -> [batch, channels, 11]
        # Transpose to [batch, time, channels] for normalization
        short_enc = short_enc.transpose(1, 2)  # [batch, channels, 11] -> [batch, 11, channels]
        # Normalize after short encoder
        short_enc = self.short_enc_norm(short_enc)
        # Transpose back to [batch, channels, time] for decoder
        short_enc = short_enc.transpose(1, 2)  # [batch, 11, channels] -> [batch, channels, 11]
        
        long_enc = self.long_enc_conv(x[:, 11:, :].transpose(1, 2))   # [batch, num_features, 20] -> [batch, channels, 20]
        # Transpose to [batch, time, channels] for normalization
        long_enc = long_enc.transpose(1, 2)  # [batch, channels, 20] -> [batch, 20, channels]
        # Normalize after long encoder
        long_enc = self.long_enc_norm(long_enc)
        # Transpose back to [batch, channels, time] for decoder
        long_enc = long_enc.transpose(1, 2)  # [batch, 20, channels] -> [batch, channels, 20]
        
        short_dec = self.short_dec_conv(short_enc)  # [batch, channels, 11] -> [batch, num_features, 11]
        # Transpose to [batch, time, channels] for normalization
        short_dec = short_dec.transpose(1, 2)  # [batch, num_features, 11] -> [batch, 11, num_features]
        # Normalize after short decoder
        short_dec = self.short_dec_norm(short_dec)
        
        long_dec = self.long_dec_conv(long_enc)    # [batch, channels, 20] -> [batch, num_features, 20]
        # Transpose to [batch, time, channels] for normalization
        long_dec = long_dec.transpose(1, 2)  # [batch, num_features, 20] -> [batch, 20, num_features] 
        # Normalize after long decoder
        long_dec = self.long_dec_norm(long_dec)

        x = torch.cat((short_dec, long_dec), dim=1)  # [batch, 31, num_features]
        return x