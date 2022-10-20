import torch
import torch.nn as nn
import math

from .mlp import MLP


class PositionalEncoding(nn.Module):
    """https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""
    def __init__(self, dim, max_pos):
        super().__init__()
        self.max_pos = max_pos
        position = torch.arange(max_pos).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_pos, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        return self.pe[x]


class PositionalMLP(nn.Module):
    def __init__(self, dim_out, max_pos, dim_pe, dim_hidden, n_layers, dropout):
        super().__init__()
        self.max_pos = max_pos
        self.pe = PositionalEncoding(dim_pe, max_pos)
        self.input = nn.Linear(dim_pe, dim_hidden)
        self.mlp = MLP(dim_hidden, dim_out, dim_hidden, n_layers, dropout)

    def reset_parameters(self):
        self.input.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        # LayerNorm: OK
        # Dropout: OK
        # Skip connections: not OK
        x = self.input(self.pe(x))
        x = self.mlp(x)
        return x


class PositionalWeight(nn.Module):
    def __init__(self, max_pos, dim_in, dim_out):
        super().__init__()
        self.max_pos = max_pos
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weights = nn.Parameter(torch.Tensor(max_pos + 1, dim_in, dim_out))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dim_out)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        return self.weights[x].view(x.size(0), -1)
