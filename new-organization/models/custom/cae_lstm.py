from ..base import BaseModel
from ..configs import CAELSTMConfig, LSTMConfig
from .lstm import LSTMModel
import torch
import torch.nn as nn

class CAELSTMModel(BaseModel):
    """
    CAE-LSTM Model for time series prediction using short/long convolution method.
    
    Architecture: Processes first 20 and last 11 timesteps separately with CAE (Convolutional AutoEncoder),
    then concatenates encoder outputs and passes through LSTM for final prediction.
    
    Args:
        model_config: Configuration object containing model parameters.
                     Must have the following attributes (all optional with defaults):
    
    Expected model_config attributes:
        - input_shape (tuple): Shape of input data (timesteps, features).
                              Format: (sequence_length, num_features)
                              Default: (31, 3)
        - kernel_size (int): Size of CNN convolution kernel (must be odd for padding).
                            Controls receptive field of convolution.
                            Applied to both short (first 20) and long (last 11) sequences.
                            Default: 3
        - hidden_size (int): Hidden dimension of LSTM layers. Default: 25
        - num_layers (int): Number of stacked LSTM layers. Default: 1
        - batch_first (bool): Whether batch is first dimension. Default: True
        - dropout (float): Dropout rate (0.0 to 1.0). Default: 0.1
    """
    def __init__(self, model_config) -> None:
        super(CAELSTMModel, self).__init__(model_config)
        if model_config is None:
            raise ValueError("model_config is required for CAELSTMModel")
        
        # Type checking: ensure model_config is CAELSTMConfig or compatible
        if not isinstance(model_config, CAELSTMConfig):
            raise TypeError(
                f"CAELSTMModel requires CAELSTMConfig instance, "
                f"got {type(model_config).__name__}. "
                f"Use CAELSTMConfig(input_shape=..., kernel_size=..., etc.) to create the config."
            )
        
        self.model_config = model_config
        self.input_shape = model_config.to_dict().get('input_shape', (31, 3))

        if self.input_shape[0] == 31:
            self.short_kernel_size = 3
            self.short_padding = 1
            self.short_stride = 1
            self.short_use_transpose = False

            self.long_kernel_size = 3
            self.long_padding = 1
            self.long_stride = 1
            self.long_use_transpose = False

            self.barrier = 20
        else:
            self.short_kernel_size = 5
            self.short_padding = 2
            self.short_stride = 2
            self.short_use_transpose = True
            # For ConvTranspose1d: output = (input-1)*stride - 2*padding + kernel + output_padding
            # Encoder: 80 -> 40, Decoder: 40 -> 80
            # 80 = (40-1)*2 - 2*2 + 5 + output_padding = 78 - 4 + 5 + output_padding = 79 + output_padding
            # output_padding = 1
            self.short_output_padding = 1

            self.long_kernel_size = 21
            self.long_padding = 10
            self.long_stride = 21
            self.long_use_transpose = True
            # Encoder: 160 -> 8, Decoder: 8 -> 160
            # 160 = (8-1)*21 - 2*10 + 21 + output_padding = 147 - 20 + 21 + output_padding = 148 + output_padding
            # output_padding = 12
            self.long_output_padding = 12

            self.barrier = 80
        
        #self.kernel_size = model_config.to_dict().get('kernel_size', 3)
        
        self.num_features = self.input_shape[1]  # Can be 3 (price only), 7 (price + 4 NLP), 13 (price + 10 NLP), etc.
        
        # Input normalization
        self.input_norm = nn.LayerNorm(self.num_features)
        
        # Short sequence CAE (first 20 timesteps)
        # Encoder: input features -> 2*input_shape[1] channels
        self.short_enc_conv = nn.Conv1d(self.num_features, 2*self.input_shape[1], self.short_kernel_size, padding=self.short_padding, stride=self.short_stride)
        self.short_enc_norm = nn.LayerNorm(2*self.input_shape[1])
        # Decoder: 2*input_shape[1] channels -> input features
        # Use ConvTranspose1d when stride > 1 to restore original length
        if self.short_use_transpose:
            self.short_dec_conv = nn.ConvTranspose1d(2*self.input_shape[1], self.num_features, self.short_kernel_size, padding=self.short_padding, stride=self.short_stride, output_padding=self.short_output_padding)
        else:
            self.short_dec_conv = nn.Conv1d(2*self.input_shape[1], self.num_features, self.short_kernel_size, padding=self.short_padding, stride=self.short_stride)
        self.short_dec_norm = nn.LayerNorm(self.num_features)
        
        # Long sequence CAE (last 11 timesteps)
        # Encoder: input features -> 2*input_shape[1] channels
        self.long_enc_conv = nn.Conv1d(self.num_features, 2*self.input_shape[1], self.long_kernel_size, padding=self.long_padding, stride=self.long_stride)
        self.long_enc_norm = nn.LayerNorm(2*self.input_shape[1])
        # Decoder: 2*input_shape[1] channels -> input features
        # Use ConvTranspose1d when stride > 1 to restore original length
        if self.long_use_transpose:
            self.long_dec_conv = nn.ConvTranspose1d(2*self.input_shape[1], self.num_features, self.long_kernel_size, padding=self.long_padding, stride=self.long_stride, output_padding=self.long_output_padding)
        else:
            self.long_dec_conv = nn.Conv1d(2*self.input_shape[1], self.num_features, self.long_kernel_size, padding=self.long_padding, stride=self.long_stride)
        self.long_dec_norm = nn.LayerNorm(self.num_features)
        
        # Use lstm_config from model_config - it's already calculated with correct input_shape
        # The config automatically calculates lstm_input_shape from encoder output shape
        self.LSTM = LSTMModel(model_config=model_config.lstm_config)

    def forward(self, x, params=None):
        # assume x in shape (batch, 31, num_features) = (batch, time_steps, features)
        # num_features can be 3 (price only), 7 (price + 4 NLP), 13 (price + 10 NLP), etc.
        # Normalize input
        x = self.input_norm(x)
        
        # #region agent log
        try:
            import json, time
            log_data = {'sessionId': 'debug-session', 'runId': 'run1', 'hypothesisId': 'A', 'location': 'cae_lstm.py:99', 'message': 'forward pass start', 'data': {'input_shape': list(x.shape), 'barrier': self.barrier, 'input_shape_config': list(self.input_shape)}, 'timestamp': int(time.time() * 1000)}
            with open('/Users/loganyamamoto/Desktop/class/CSCI/566/project/StockPredictor/.cursor/debug.log', 'a') as f:
                f.write(json.dumps(log_data) + '\n')
        except: pass
        # #endregion
        
        # Process short sequence (first 20 timesteps)
        # Conv1d expects [batch, channels, length], so transpose from [batch, time, features] to [batch, features, time]
        short_input = x[:, :self.barrier, :]
        # #region agent log
        try:
            import json, time
            log_data = {'sessionId': 'debug-session', 'runId': 'run1', 'hypothesisId': 'A', 'location': 'cae_lstm.py:107', 'message': 'short_input shape', 'data': {'short_input_shape': list(short_input.shape), 'barrier': self.barrier}, 'timestamp': int(time.time() * 1000)}
            with open('/Users/loganyamamoto/Desktop/class/CSCI/566/project/StockPredictor/.cursor/debug.log', 'a') as f:
                f.write(json.dumps(log_data) + '\n')
        except: pass
        # #endregion
        short_enc = self.short_enc_conv(short_input.transpose(1, 2))  # [batch, num_features, 20] -> [batch, 2*num_features, 20]
        # #region agent log
        try:
            import json, time
            log_data = {'sessionId': 'debug-session', 'runId': 'run1', 'hypothesisId': 'A', 'location': 'cae_lstm.py:113', 'message': 'short_enc after conv', 'data': {'short_enc_shape': list(short_enc.shape)}, 'timestamp': int(time.time() * 1000)}
            with open('/Users/loganyamamoto/Desktop/class/CSCI/566/project/StockPredictor/.cursor/debug.log', 'a') as f:
                f.write(json.dumps(log_data) + '\n')
        except: pass
        # #endregion
        # Transpose to [batch, time, channels] for normalization
        short_enc = short_enc.transpose(1, 2)  # [batch, 2*num_features, 20] -> [batch, 20, 2*num_features]
        # Normalize after short encoder
        short_enc = self.short_enc_norm(short_enc)  # [batch, 20, 2*num_features]
        
        # Process long sequence (last 11 timesteps)
        long_input = x[:, self.barrier:, :]
        # #region agent log
        try:
            import json, time
            log_data = {'sessionId': 'debug-session', 'runId': 'run1', 'hypothesisId': 'A', 'location': 'cae_lstm.py:125', 'message': 'long_input shape', 'data': {'long_input_shape': list(long_input.shape), 'barrier': self.barrier, 'total_input_len': x.shape[1]}, 'timestamp': int(time.time() * 1000)}
            with open('/Users/loganyamamoto/Desktop/class/CSCI/566/project/StockPredictor/.cursor/debug.log', 'a') as f:
                f.write(json.dumps(log_data) + '\n')
        except: pass
        # #endregion
        long_enc = self.long_enc_conv(long_input.transpose(1, 2))   # [batch, num_features, 11] -> [batch, 2*num_features, 11]
        # #region agent log
        try:
            import json, time
            log_data = {'sessionId': 'debug-session', 'runId': 'run1', 'hypothesisId': 'A', 'location': 'cae_lstm.py:131', 'message': 'long_enc after conv', 'data': {'long_enc_shape': list(long_enc.shape)}, 'timestamp': int(time.time() * 1000)}
            with open('/Users/loganyamamoto/Desktop/class/CSCI/566/project/StockPredictor/.cursor/debug.log', 'a') as f:
                f.write(json.dumps(log_data) + '\n')
        except: pass
        # #endregion
        # Transpose to [batch, time, channels] for normalization
        long_enc = long_enc.transpose(1, 2)  # [batch, 2*num_features, 11] -> [batch, 11, 2*num_features]
        # Normalize after long encoder
        long_enc = self.long_enc_norm(long_enc)  # [batch, 11, 2*num_features]
        
        # Concatenate encoder outputs for LSTM: [batch, seq_len, 2*num_features]
        x = torch.cat((short_enc, long_enc), dim=1)  # [batch, 31, 2*num_features]
        
        # Store encoder outputs for decoder/reconstruction (needed for reconstruction loss)
        # Transpose back to [batch, channels, time] for decoder operations
        short_enc_for_decoder = short_enc.transpose(1, 2)  # [batch, 2*num_features, 20]
        long_enc_for_decoder = long_enc.transpose(1, 2)  # [batch, 2*num_features, 11]
        
        # Decode short sequence (for reconstruction loss - enables multi-task learning)
        short_dec = self.short_dec_conv(short_enc_for_decoder)  # [batch, channels, 20] -> [batch, num_features, 20]
        # #region agent log
        try:
            import json, time
            log_data = {'sessionId': 'debug-session', 'runId': 'run1', 'hypothesisId': 'A', 'location': 'cae_lstm.py:145', 'message': 'short_dec after conv', 'data': {'short_dec_shape': list(short_dec.shape), 'short_input_len': short_input.shape[1]}, 'timestamp': int(time.time() * 1000)}
            with open('/Users/loganyamamoto/Desktop/class/CSCI/566/project/StockPredictor/.cursor/debug.log', 'a') as f:
                f.write(json.dumps(log_data) + '\n')
        except: pass
        # #endregion
        # Transpose to [batch, time, channels] for normalization
        short_dec = short_dec.transpose(1, 2)  # [batch, num_features, 20] -> [batch, 20, num_features]
        # Normalize after short decoder
        short_dec = self.short_dec_norm(short_dec)
        
        # Decode long sequence (for reconstruction loss - enables multi-task learning)
        long_dec = self.long_dec_conv(long_enc_for_decoder)    # [batch, channels, 11] -> [batch, num_features, 11]
        # #region agent log
        try:
            import json, time
            log_data = {'sessionId': 'debug-session', 'runId': 'run1', 'hypothesisId': 'A', 'location': 'cae_lstm.py:155', 'message': 'long_dec after conv', 'data': {'long_dec_shape': list(long_dec.shape), 'long_input_len': long_input.shape[1]}, 'timestamp': int(time.time() * 1000)}
            with open('/Users/loganyamamoto/Desktop/class/CSCI/566/project/StockPredictor/.cursor/debug.log', 'a') as f:
                f.write(json.dumps(log_data) + '\n')
        except: pass
        # #endregion
        # Transpose to [batch, time, channels] for normalization
        long_dec = long_dec.transpose(1, 2)  # [batch, num_features, 11] -> [batch, 11, num_features] 
        # Normalize after long decoder
        long_dec = self.long_dec_norm(long_dec)
        
        # Pass through LSTM
        x = self.LSTM(x)
        return x

    @classmethod
    def from_config(cls, model_config):
        """Factory hook so the registry can instantiate the model."""
        return cls(model_config)

# Register the model
from ..registry import ModelRegistry
ModelRegistry.register("CAELSTM", lambda config: CAELSTMModel(config), CAELSTMConfig)
