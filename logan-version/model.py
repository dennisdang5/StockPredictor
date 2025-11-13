
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.init as init

 
class LSTMModel(nn.Module):
    def __init__(self, input_dim=(31,3), hidden_size=25, num_layers=1, batch_first=True, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        
        # Input normalization layer - helps stabilize inputs
        # Normalizes across features at each time step
        self.input_norm = nn.LayerNorm(input_dim[1])
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size=input_dim[1], hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first, dtype=torch.float32)
        
        # Layer normalization after LSTM - stabilizes activations and gradients
        self.lstm_norm = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, 1)
        
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
        
        # Final linear layer
        x = self.linear(x)
        return x

class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim=(31,3), kernel_size=3, hidden_size=25, num_layers=1, batch_first=True, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.input_norm = nn.LayerNorm(input_dim[1])
        # first 11 of input should get one cnn kernel and last 20 should have a different kernel size
        self.cnn1 = nn.Conv1d(input_dim[1], hidden_size, kernel_size=kernel_size, padding=kernel_size//2)
        self.cnn2 = nn.Conv1d(input_dim[1], hidden_size, kernel_size=kernel_size, padding=kernel_size//2)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=batch_first, dtype=torch.float32)
        self.lstm_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, 1)
        
        
        
    def forward(self, x):
        x = self.input_norm(x)
        # first 11 of input should get one cnn kernel and last 20 should have a different kernel size
        # Conv1d expects [batch, channels, length], so transpose from [batch, time, features] to [batch, features, time]
        x1 = self.cnn1(x[:, :11, :].transpose(1, 2))  # [batch, 3, 11] -> [batch, hidden_size, 11]
        x2 = self.cnn2(x[:, 11:, :].transpose(1, 2))  # [batch, 3, 20] -> [batch, hidden_size, 20]
        # Transpose back to [batch, time, features] for LSTM
        x1 = x1.transpose(1, 2)  # [batch, hidden_size, 11] -> [batch, 11, hidden_size]
        x2 = x2.transpose(1, 2)  # [batch, hidden_size, 20] -> [batch, 20, hidden_size]
        x = torch.cat((x1, x2), dim=1)  # [batch, 31, hidden_size]
        x, _ = self.lstm(x)
        x = self.lstm_norm(x[:,-1,:]) # apply layer normalization to last time step only
        x = self.dropout(x)
        x = self.linear(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_shape=(31,3), ) -> None:
        super(AutoEncoder,self).__init__()
        self.input_shape = input_shape
        def _dof(x):
            output = 1
            for val in x:
                output *= x
            return output
        self.dof = _dof(self.input_shape)
        self.encoder=nn.Sequential(
            # naive concatenation
            nn.Linear(self.dof, 2*self.dof),
            nn.Sigmoid()
        )
        self.decoder=nn.Sequential(
            nn.Linear(2*self.dof, self.dof),
            nn.Sigmoid()
        )

        ###### other parameters ####
        # nn.MSELoss()
        # optimizer = torch.optim.Adam()

    def forward(self, x):
        # assume x in shape (31,3)
        x = torch.flatten(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        x = torch.unflatten(decoded, dim=1, sizes=(31,3))
        return x


class CNNAutoEncoder(nn.Module):
    def __init__(self, input_shape=(31,3), ) -> None:
        super(CNNAutoEncoder,self).__init__()
        self.input_shape = input_shape
        self.short_enc_conv = nn.Conv1d(3, 2*self.input_shape[1], 3, padding=1)
        self.long_enc_conv = nn.Conv1d(3, 2*self.input_shape[1], 3, padding=1)

        self.short_dec_conv = nn.Conv1d(2*self.input_shape[1], 3, 3, padding=1)
        self.long_dec_conv = nn.Conv1d(2*self.input_shape[1], 3, 3, padding=1)

        ###### other parameters ####
        # nn.MSELoss()
        # optimizer = torch.optim.Adam()

    def forward(self, x):
        # assume x in shape (batch, 31, 3) = (batch, time_steps, features)
        # Conv1d expects [batch, channels, length], so transpose from [batch, time, features] to [batch, features, time]
        short_enc = self.short_enc_conv(x[:, :11, :].transpose(1, 2))  # [batch, 3, 11] -> [batch, channels, 11]
        long_enc = self.long_enc_conv(x[:, 11:, :].transpose(1, 2))   # [batch, 3, 20] -> [batch, channels, 20]
        
        short_dec = self.short_dec_conv(short_enc)  # [batch, channels, 11] -> [batch, 3, 11]
        long_dec = self.long_dec_conv(long_enc)    # [batch, channels, 20] -> [batch, 3, 20]

        # Transpose back to [batch, time, features] for concatenation
        short_dec = short_dec.transpose(1, 2)  # [batch, 3, 11] -> [batch, 11, 3]
        long_dec = long_dec.transpose(1, 2)    # [batch, 3, 20] -> [batch, 20, 3]
        x = torch.cat((short_dec, long_dec), dim=1)  # [batch, 31, 3]
        return x

class AELSTM(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(AELSTM, self).__init__(*args, **kwargs)
        self.AE = AutoEncoder()

class CNNAELSTM(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)