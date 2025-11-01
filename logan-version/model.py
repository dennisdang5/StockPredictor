
import pandas as pd
import torch.nn as nn
import torch

 
class LSTMModel(nn.Module):
    def __init__(self, input_dim=(31,3), hidden_size=25, num_layers=1, batch_first=True, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(input_size=input_dim[1], hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first, dtype=torch.float32)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:,-1,:])
        return x
