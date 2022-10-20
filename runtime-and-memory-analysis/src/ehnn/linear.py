import torch
import torch.nn as nn

import math

from .mlp import MLP
from .hypernetwork import PositionalEncoding, PositionalMLP, PositionalWeight


class BiasE(nn.Module):
    def __init__(self, dim_out,
                 max_l, pe_dim, hyper_dim, hyper_layers, hyper_dropout, force_broadcast):
        super().__init__()
        self.dim_out = dim_out
        self.max_l = max_l
        self.pe_dim = pe_dim
        self.hyper_dim = hyper_dim
        self.hyper_layers = hyper_layers
        self.hyper_dropout = hyper_dropout
        self.force_broadcast = force_broadcast
        self.b = PositionalMLP(dim_out, max_l + 1, pe_dim, hyper_dim, hyper_layers, hyper_dropout)

    def reset_parameters(self):
        self.b.reset_parameters()

    def forward(self, x, edge_orders):
        x_v, x_e = x
        if (self.max_l and self.hyper_dropout == 0) or self.force_broadcast:
            # do not use this when hypernetwork is stochastic
            indices = torch.arange(self.b.max_pos, device=x_v.device)
            b = self.b(indices)[edge_orders]
        else:
            b = self.b(edge_orders)
        b_1 = self.b(torch.ones(1, dtype=torch.long, device=x_v.device)).view(1, self.dim_out)  # [D']
        return x_v + b_1, x_e + b


class BiasV(nn.Module):
    def __init__(self, dim_out):
        super().__init__()
        self.dim_out = dim_out
        self.b = nn.Parameter(torch.Tensor(1, dim_out))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dim_out)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return x + self.b


class LinearV2E(nn.Module):
    def __init__(self, dim_in, dim_out,
                 hypernet_info, inner_dim, pe_dim, hyper_dim, hyper_layers, hyper_dropout, force_broadcast):
        super().__init__()
        assert dim_in == dim_out == inner_dim
        _, max_l, _ = hypernet_info
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.inner_dim = inner_dim
        self.max_l = max_l
        self.pe_dim = pe_dim
        self.hyper_dim = hyper_dim
        self.hyper_layers = hyper_layers
        self.hyper_dropout = hyper_dropout
        self.force_broadcast = force_broadcast
        # node-input only, don't need pe1
        self.mlp1 = MLP(dim_in, inner_dim, hyper_dim, hyper_layers, hyper_dropout)
        self.mlp2 = MLP(inner_dim*2, inner_dim, hyper_dim, hyper_layers, hyper_dropout)
        self.mlp3 = MLP(inner_dim*2, dim_out, hyper_dim, hyper_layers, hyper_dropout)
        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm2 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)
        self.pe2 = PositionalEncoding(inner_dim, 2)
        self.pe3 = PositionalEncoding(inner_dim, max_l + 1)
        self.b = BiasE(dim_out, max_l, pe_dim, hyper_dim, hyper_layers, hyper_dropout, force_broadcast)

    def reset_parameters(self):
        self.mlp1.reset_parameters()
        self.mlp2.reset_parameters()
        self.mlp3.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm3.reset_parameters()
        self.b.reset_parameters()

    def forward(self, x, ehnn_cache) -> [torch.Tensor, torch.Tensor]:
        incidence = ehnn_cache['incidence']
        edge_orders = ehnn_cache['edge_orders']
        prefix_normalizer = ehnn_cache['prefix_normalizer']
        # MLP1
        x = x + self.mlp1(self.norm1(x))  # [N, D]
        # aggregation
        x0 = x.mean(dim=0, keepdim=True)  # [1, D]
        x1_v = x  # [N, D]
        x1_e = (incidence.t() @ x / prefix_normalizer[:, None])  # [|E|, D]
        # MLP2
        pe2_s0 = self.pe2(torch.zeros(1, dtype=torch.long, device=x.device)).view(1, self.inner_dim)  # [1, D]
        pe2_s1 = self.pe2(torch.ones(1, dtype=torch.long, device=x.device)).view(1, self.inner_dim)  # [1, D]

        t0 = self.norm2(x0)
        x0 = x0 + self.mlp2(torch.cat((t0 , pe2_s0), dim=-1))

        t1_v = self.norm2(x1_v)
        t1_e = self.norm2(x1_e)
        x1_v = x1_v + self.mlp2(torch.cat((t1_v , pe2_s1.expand(t1_v.shape)), dim=-1))
        x1_e = x1_e + self.mlp2(torch.cat((t1_e , pe2_s1.expand(t1_e.shape)), dim=-1))

        x_v = x0 + x1_v
        x_e = x0 + x1_e
        # MLP3
        pe3_l1 = self.pe3(torch.ones(1, dtype=torch.long, device=x.device)).view(1, self.inner_dim)
        if (self.max_l < len(edge_orders) and self.hyper_dropout == 0) or self.force_broadcast:
            # do not use this when hypernetwork is stochastic
            indices = torch.arange(self.pe3.max_pos, device=x.device)
            pe3_l = self.pe3(indices)[edge_orders]  # [|E|, D]
        else:
            pe3_l = self.pe3(edge_orders)  # [|E|, D]

        t_v = self.norm3(x_v)
        t_e = self.norm3(x_e)
        x_v = x_v + self.mlp3(torch.cat((t_v , pe3_l1.expand(t_v.shape)), dim=-1))
        x_e = x_e + self.mlp3(torch.cat((t_e , pe3_l.expand(t_e.shape)), dim=-1))

        x = x_v, x_e
        x = self.b(x, edge_orders)
        return x


class LinearE2V(nn.Module):
    def __init__(self, dim_in, dim_out,
                 hypernet_info, inner_dim, pe_dim, hyper_dim, hyper_layers, hyper_dropout, force_broadcast):
        super().__init__()
        assert dim_in == dim_out == inner_dim
        max_k, _, _ = hypernet_info
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.inner_dim = inner_dim
        self.max_k = max_k
        self.pe_dim = pe_dim
        self.hyper_dim = hyper_dim
        self.hyper_layers = hyper_layers
        self.hyper_dropout = hyper_dropout
        self.force_broadcast = force_broadcast
        self.mlp1 = MLP(dim_in*2, inner_dim, hyper_dim, hyper_layers, hyper_dropout)
        self.mlp2 = MLP(inner_dim*2, inner_dim, hyper_dim, hyper_layers, hyper_dropout)
        self.mlp3 = MLP(inner_dim, dim_out, hyper_dim, hyper_layers, hyper_dropout)
        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm2 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)
        # node-input only, don't need pe3
        self.pe1 = PositionalEncoding(dim_in, max_k + 1)
        self.pe2 = PositionalEncoding(inner_dim, 2)
        self.b = BiasV(dim_out)

    def reset_parameters(self):
        self.mlp1.reset_parameters()
        self.mlp2.reset_parameters()
        self.mlp3.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm3.reset_parameters()
        self.b.reset_parameters()

    def forward(self, x, ehnn_cache) -> torch.Tensor:
        incidence = ehnn_cache['incidence']
        edge_orders = ehnn_cache['edge_orders']
        suffix_normalizer = ehnn_cache['suffix_normalizer']
        x_v, x_e = x
        # MLP1
        pe1_k1 = self.pe1(torch.ones(1, dtype=torch.long, device=x_v.device)).view(1, self.dim_in)
        if (self.max_k < len(edge_orders) and self.hyper_dropout == 0) or self.force_broadcast:
            # do not use this when hypernetwork is stochastic
            indices = torch.arange(self.pe1.max_pos, device=x_v.device)
            pe1_k = self.pe1(indices)[edge_orders]  # [|E|, D]
        else:
            pe1_k = self.pe1(edge_orders)  # [|E|, D]

        t_v = self.norm1(x_v)
        t_e = self.norm1(x_e)
        x_v = x_v + self.mlp1(torch.cat((t_v, pe1_k1.expand(t_v.shape)), dim=-1))
        x_e = x_e + self.mlp1(torch.cat((t_e , pe1_k.expand(t_e.shape)), dim=-1))

        # aggregation
        x0 = torch.cat((x_v, x_e)).mean(dim=0, keepdim=True)  # [1, D]
        x1 = ((x_v + incidence @ x_e) / (1 + suffix_normalizer[:, None]))  # [N, D]
        # MLP2
        pe2_s0 = self.pe2(torch.zeros(1, dtype=torch.long, device=x_v.device)).view(1, self.inner_dim)  # [1, D]
        pe2_s1 = self.pe2(torch.ones(1, dtype=torch.long, device=x_v.device)).view(1, self.inner_dim)  # [1, D]

        t0 = self.norm2(x0)
        t1 = self.norm2(x1)
        x0 = x0 + self.mlp2(torch.cat((t0 , pe2_s0.expand(t0.shape)), dim=-1))
        x1 = x1 + self.mlp2(torch.cat((t1 , pe2_s1.expand(t1.shape)), dim=-1))

        x = x0 + x1
        # MLP3
        x = x + self.mlp3(self.norm3(x))  # [N, D']
        x = self.b(x)
        return x
