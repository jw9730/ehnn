from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, n_layers, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.f = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        assert n_layers > 0
        self.layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(nn.Linear(dim_in, dim_out))
        else:
            for layer_idx in range(n_layers):
                self.layers.append(
                    nn.Linear(dim_hidden if layer_idx > 0 else dim_in,
                              dim_hidden if layer_idx < n_layers - 1 else dim_out)
                )

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for idx in range(self.n_layers - 1):
            x = self.layers[idx](x)
            x = self.f(x)
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x

    def multi_forward(self, xs: List) -> List:
        for x in xs:
            assert len(x.size()) == 2
        return self.forward(torch.cat(xs)).split([x.size(0) for x in xs])
