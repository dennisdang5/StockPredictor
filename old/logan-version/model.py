"""
Model definitions for stock prediction.

All models require a model_config object containing configuration parameters.
The model_config can be created using ModelConfig classes, types.SimpleNamespace, or any object with attributes.

Common model_config parameters across all models:
    - input_shape (tuple): (timesteps, features) - Default: (31, 3)

Important Notes:
    - seq_len (sequence length) is NOT a model_config parameter. It should be set in TrainerConfig.seq_len
      and controls both data generation (feature extraction window) and model architecture.
    - For TimesNet specifically, seq_len must be provided via TrainerConfig.seq_len, not in TimesNetConfig.

Model-specific parameters are documented in each class's docstring.
"""

import pandas as pd
import torch.nn as nn
import torch
import torch.nn.init as init


# ============================================================================
# Model Configuration Classes
# ============================================================================

class ModelConfig:
    """
    Base configuration class for all models.
    
    All model configs inherit from this base class and must include input_shape.
    
    Note: seq_len (sequence length) is NOT a parameter in ModelConfig classes.
    It should be set in TrainerConfig.seq_len, which controls both data generation
    (feature extraction window) and model architecture.
    """
    def __init__(self, input_shape=(31, 3)):
        """
        Initialize base model configuration.
        
        Args:
            input_shape (tuple): Shape of input data (timesteps, features).
                               Format: (sequence_length, num_features)
                               Default: (31, 3)
                               Note: The sequence_length in input_shape should match
                               TrainerConfig.seq_len for consistency.
        """
        self.input_shape = input_shape
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def __repr__(self):
        """String representation of config."""
        attrs = ', '.join(f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith('_'))
        return f"{self.__class__.__name__}({attrs})"




# ============================================================================
# Model Classes
# ============================================================================

class LSTMModel(nn.Module):
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
    def __init__(self, model_config):
        super().__init__()
        if model_config is None:
            raise ValueError("model_config is required for LSTMModel")
        
        # Type checking: ensure model_config is LSTMConfig or compatible
        if not isinstance(model_config, (LSTMConfig, ModelConfig)):
            # Allow SimpleNamespace for backward compatibility but warn
            from types import SimpleNamespace
            if not isinstance(model_config, SimpleNamespace):
                raise TypeError(
                    f"LSTMModel requires LSTMConfig or ModelConfig instance, "
                    f"got {type(model_config).__name__}. "
                    f"Use LSTMConfig(input_shape=..., hidden_size=..., etc.) to create the config."
                )
        
        # Extract all parameters from model_config with defaults
        input_shape = getattr(model_config, 'input_shape', (31, 3))
        hidden_size = getattr(model_config, 'hidden_size', 25)
        num_layers = getattr(model_config, 'num_layers', 1)
        batch_first = getattr(model_config, 'batch_first', True)
        dropout = getattr(model_config, 'dropout', 0.1)
        
        self.input_dim = input_shape
        self.hidden_size = hidden_size
        
        # Input normalization layer - helps stabilize inputs
        # Normalizes across features at each time step
        self.input_norm = nn.LayerNorm(input_shape[1])
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size=input_shape[1], hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first, dtype=torch.float32)
        
        # Layer normalization after LSTM - stabilizes activations and gradients
        self.lstm_norm = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, 1)
        # Note: No normalization after final output layer - output should be in natural scale
        
        # Initialize weights properly to prevent NaN
        self._initialize_weights()
    
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

class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM Model combining convolutional and LSTM layers.
    
    Architecture: CNN layers process first 11 and last 20 timesteps separately,
    then concatenated output is fed to LSTM layers.
    
    Args:
        model_config: Configuration object containing model parameters.
                     Must have the following attributes (all optional with defaults):
    
    Expected model_config attributes:
        - input_shape (tuple): Shape of input data (timesteps, features).
                              Format: (sequence_length, num_features)
                              Default: (31, 3)
        - kernel_size (int): Size of CNN convolution kernel (must be odd for padding).
                            Controls receptive field of convolution.
                            Default: 3
        - hidden_size (int): Hidden dimension used for both CNN and LSTM layers.
                            Controls the size of CNN output channels and LSTM hidden state.
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
        super().__init__()
        if model_config is None:
            raise ValueError("model_config is required for CNNLSTMModel")
        
        # Type checking: ensure model_config is CNNLSTMConfig or compatible
        if not isinstance(model_config, (CNNLSTMConfig, ModelConfig)):
            # Allow SimpleNamespace for backward compatibility but warn
            from types import SimpleNamespace
            if not isinstance(model_config, SimpleNamespace):
                raise TypeError(
                    f"CNNLSTMModel requires CNNLSTMConfig or ModelConfig instance, "
                    f"got {type(model_config).__name__}. "
                    f"Use CNNLSTMConfig(input_shape=..., kernel_size=..., etc.) to create the config."
                )
        
        # Extract all parameters from model_config with defaults
        input_shape = getattr(model_config, 'input_shape', (31, 3))
        kernel_size = getattr(model_config, 'kernel_size', 3)
        hidden_size = getattr(model_config, 'hidden_size', 25)
        num_layers = getattr(model_config, 'num_layers', 1)
        batch_first = getattr(model_config, 'batch_first', True)
        dropout = getattr(model_config, 'dropout', 0.1)
        
        self.input_dim = input_shape
        self.hidden_size = hidden_size
        self.input_norm = nn.LayerNorm(input_shape[1])
        # first 11 of input should get one cnn kernel and last 20 should have a different kernel size
        self.cnn1 = nn.Conv1d(input_shape[1], hidden_size, kernel_size=kernel_size, padding=kernel_size//2)
        # Normalization after CNN1 (normalize across channels for each time step)
        self.cnn1_norm = nn.LayerNorm(hidden_size)
        self.cnn2 = nn.Conv1d(input_shape[1], hidden_size, kernel_size=kernel_size, padding=kernel_size//2)
        # Normalization after CNN2 (normalize across channels for each time step)
        self.cnn2_norm = nn.LayerNorm(hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=batch_first, dtype=torch.float32)
        self.lstm_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, 1)
        # Note: No normalization after final output layer - output should be in natural scale
        
        
        
    def forward(self, x):
        x = self.input_norm(x)
        # first 11 of input should get one cnn kernel and last 20 should have a different kernel size
        # Conv1d expects [batch, channels, length], so transpose from [batch, time, features] to [batch, features, time]
        # num_features can be 3 (price only), 7 (price + 4 NLP), 13 (price + 10 NLP), etc.
        x1 = self.cnn1(x[:, :11, :].transpose(1, 2))  # [batch, num_features, 11] -> [batch, hidden_size, 11]
        # Transpose back to [batch, time, features] for normalization
        x1 = x1.transpose(1, 2)  # [batch, hidden_size, 11] -> [batch, 11, hidden_size]
        # Normalize after CNN1
        x1 = self.cnn1_norm(x1)
        
        x2 = self.cnn2(x[:, 11:, :].transpose(1, 2))  # [batch, num_features, 20] -> [batch, hidden_size, 20]
        # Transpose back to [batch, time, features] for normalization
        x2 = x2.transpose(1, 2)  # [batch, hidden_size, 20] -> [batch, 20, hidden_size]
        # Normalize after CNN2
        x2 = self.cnn2_norm(x2)
        
        x = torch.cat((x1, x2), dim=1)  # [batch, 31, hidden_size]
        x, _ = self.lstm(x)
        x = self.lstm_norm(x[:,-1,:]) # apply layer normalization to last time step only
        x = self.dropout(x)
        # Final linear layer (no normalization - output should be in natural scale)
        x = self.linear(x)
        return x


class AutoEncoder(nn.Module):
    """
    AutoEncoder for feature extraction and dimensionality reduction.
    
    Architecture: Encodes input to 2x dimensions, then decodes back to original shape.
    Used for feature extraction and denoising.
    
    Args:
        model_config: Configuration object containing model parameters.
                     Must have the following attributes (all optional with defaults):
    
    Expected model_config attributes:
        - input_shape (tuple): Shape of input data (timesteps, features).
                              Format: (sequence_length, num_features)
                              Used to calculate total input dimensions (timesteps * features).
                              Default: (31, 3)
    """
    def __init__(self, model_config):
        super(AutoEncoder,self).__init__()
        if model_config is None:
            raise ValueError("model_config is required for AutoEncoder")
        
        # Type checking: ensure model_config is AutoEncoderConfig or compatible
        if not isinstance(model_config, (AutoEncoderConfig, ModelConfig)):
            # Allow SimpleNamespace for backward compatibility but warn
            from types import SimpleNamespace
            if not isinstance(model_config, SimpleNamespace):
                raise TypeError(
                    f"AutoEncoder requires AutoEncoderConfig or ModelConfig instance, "
                    f"got {type(model_config).__name__}. "
                    f"Use AutoEncoderConfig(input_shape=...) to create the config."
                )
        
        # Extract all parameters from model_config with defaults
        input_shape = getattr(model_config, 'input_shape', (31, 3))
        
        self.input_shape = input_shape
        def _dof(x):
            output = 1
            for val in x:
                output *= val
            return output
        self.dof = _dof(self.input_shape)
        # Input normalization
        self.input_norm = nn.LayerNorm(self.dof)
        self.encoder=nn.Sequential(
            # naive concatenation
            nn.Linear(self.dof, 2*self.dof),
            nn.LayerNorm(2*self.dof),  # Normalization after encoder linear
            nn.ReLU()
        )
        self.decoder=nn.Sequential(
            nn.Linear(2*self.dof, self.dof),
            nn.LayerNorm(self.dof),  # Normalization after decoder linear
            nn.ReLU()
        )

        ###### other parameters ####
        # nn.MSELoss()
        # optimizer = torch.optim.Adam()

    def forward(self, x):
        # assume x in shape (batch, 31, 3) or (31, 3)
        if x.dim() == 2:
            # Single sample, add batch dimension
            x = x.unsqueeze(0)
        original_shape = x.shape
        x = torch.flatten(x, start_dim=1)  # Flatten spatial dimensions, keep batch
        # Normalize input
        x = self.input_norm(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Unflatten back to original spatial shape
        x = torch.unflatten(decoded, dim=1, sizes=(self.input_shape[0], self.input_shape[1]))
        # Remove batch dimension if it was added
        if original_shape[0] == 1 and len(original_shape) == 2:
            x = x.squeeze(0)
        return x


class CNNAutoEncoder(nn.Module):
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
        super(CNNAutoEncoder,self).__init__()
        if model_config is None:
            raise ValueError("model_config is required for CNNAutoEncoder")
        
        # Type checking: ensure model_config is CNNAutoEncoderConfig or compatible
        if not isinstance(model_config, (CNNAutoEncoderConfig, ModelConfig)):
            # Allow SimpleNamespace for backward compatibility but warn
            from types import SimpleNamespace
            if not isinstance(model_config, SimpleNamespace):
                raise TypeError(
                    f"CNNAutoEncoder requires CNNAutoEncoderConfig or ModelConfig instance, "
                    f"got {type(model_config).__name__}. "
                    f"Use CNNAutoEncoderConfig(input_shape=..., kernel_size=...) to create the config."
                )
        
        # Extract all parameters from model_config with defaults
        input_shape = getattr(model_config, 'input_shape', (31, 3))
        kernel_size = getattr(model_config, 'kernel_size', 3)
        
        self.input_shape = input_shape
        self.num_features = input_shape[1]  # Can be 3 (price only), 7 (price + 4 NLP), 13 (price + 10 NLP), etc.
        # Input normalization
        self.input_norm = nn.LayerNorm(self.num_features)
        # Encoder: input features -> 2*input_shape[1] channels
        self.short_enc_conv = nn.Conv1d(self.num_features, 2*self.input_shape[1], kernel_size, padding=kernel_size//2)
        # Normalization after short encoder conv (normalize across channels)
        self.short_enc_norm = nn.LayerNorm(2*self.input_shape[1])
        self.long_enc_conv = nn.Conv1d(self.num_features, 2*self.input_shape[1], kernel_size, padding=kernel_size//2)
        # Normalization after long encoder conv (normalize across channels)
        self.long_enc_norm = nn.LayerNorm(2*self.input_shape[1])

        # Decoder: 2*input_shape[1] channels -> input features
        self.short_dec_conv = nn.Conv1d(2*self.input_shape[1], self.num_features, kernel_size, padding=kernel_size//2)
        # Normalization after short decoder conv (normalize across channels)
        self.short_dec_norm = nn.LayerNorm(self.num_features)
        self.long_dec_conv = nn.Conv1d(2*self.input_shape[1], self.num_features, kernel_size, padding=kernel_size//2)
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

class AELSTM(nn.Module):
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
        super(AELSTM, self).__init__()
        if model_config is None:
            raise ValueError("model_config is required for AELSTM")
        
        # Type checking: ensure model_config is AELSTMConfig or compatible
        if not isinstance(model_config, (AELSTMConfig, ModelConfig)):
            # Allow SimpleNamespace for backward compatibility but warn
            from types import SimpleNamespace
            if not isinstance(model_config, SimpleNamespace):
                raise TypeError(
                    f"AELSTM requires AELSTMConfig or ModelConfig instance, "
                    f"got {type(model_config).__name__}. "
                    f"Use AELSTMConfig(input_shape=..., hidden_size=..., etc.) to create the config."
                )
        
        # Extract all parameters from model_config with defaults
        input_shape = getattr(model_config, 'input_shape', (31, 3))
        
        self.input_shape = input_shape
        self.AE = AutoEncoder(model_config=model_config)
        self.LSTM = LSTMModel(model_config=model_config)

    def forward(self, x):
        x = self.AE(x)
        x = self.LSTM(x)
        return x

class CNNAELSTM(nn.Module):
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
        super(CNNAELSTM, self).__init__()
        if model_config is None:
            raise ValueError("model_config is required for CNNAELSTM")
        
        # Type checking: ensure model_config is CNNAELSTMConfig or compatible
        if not isinstance(model_config, (CNNAELSTMConfig, ModelConfig)):
            # Allow SimpleNamespace for backward compatibility but warn
            from types import SimpleNamespace
            if not isinstance(model_config, SimpleNamespace):
                raise TypeError(
                    f"CNNAELSTM requires CNNAELSTMConfig or ModelConfig instance, "
                    f"got {type(model_config).__name__}. "
                    f"Use CNNAELSTMConfig(input_shape=..., kernel_size=..., etc.) to create the config."
                )
        
        # Extract all parameters from model_config with defaults
        input_shape = getattr(model_config, 'input_shape', (31, 3))
        kernel_size = getattr(model_config, 'kernel_size', 3)
        
        self.CNNAE = CNNAutoEncoder(model_config=model_config)
        self.LSTM = LSTMModel(model_config=model_config)

    def forward(self, x):
        x = self.CNNAE(x)
        x = self.LSTM(x)
        return x