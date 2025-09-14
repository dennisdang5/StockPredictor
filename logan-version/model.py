
import pandas as pd
import torch.nn as nn
 
class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=240*3, hidden_size=25, num_layers=1, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(25, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x