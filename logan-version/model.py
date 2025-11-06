
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.init as init

 
class LSTMModel(nn.Module):
    def __init__(self, input_dim=(31,3), hidden_size=25, num_layers=1, batch_first=True, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(input_size=input_dim[1], hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first, dtype=torch.float32)
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
        x, _ = self.lstm(x)
        x = self.linear(x[:,-1,:])
        return x
