import torch 
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create vector of shape (Seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (- math.log(10000.0) / d_model))
        # Apply the sin to even position and cos to odd position
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueze(0)

        self.register_buffer("pe", pe) # register as buffer 
    def forward(self, x): 
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)