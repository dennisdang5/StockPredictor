
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
        x1 = self.cnn1(x[:, :11, :])
        x2 = self.cnn2(x[:, 11:, :])
        x = torch.cat((x1, x2), dim=1)
        x, _ = self.lstm(x)
        x = self.lstm_norm(x[:,-1,:]) # apply layer normalization to last time step only
        x = self.dropout(x)
        x = self.linear(x)
