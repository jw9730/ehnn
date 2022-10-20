import torch
import torch.nn as nn

import math

from .mlp import MLP
from .hypernetwork import PositionalEncoding, PositionalMLP, PositionalWeight


CONCAT_PE = False


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

    def forward(self, x, ehnn_cache):
        """
        :param x: ([B, N, D], [B, |E|, D])
        :param ehnn_cache:
        :return:
        """
        edge_orders = ehnn_cache['edge_orders']
        node_mask = ehnn_cache['node_mask']
        edge_mask = ehnn_cache['edge_mask']

        x_v, x_e = x
        if (self.max_l and self.hyper_dropout == 0) or self.force_broadcast:
            # do not use this when hypernetwork is stochastic
            indices = torch.arange(self.b.max_pos, device=x_v.device)
            b = self.b(indices)[edge_orders]
        else:
            b = self.b(edge_orders)
        b_1 = self.b(torch.ones(1, dtype=torch.long, device=x_v.device)).view(1, 1, self.dim_out)  # [D']
        x_v = x_v + b_1
        x_e = x_e + b
        x_v = x_v.masked_fill(~node_mask[..., None], 0)  # [B, N, D]
        x_e = x_e.masked_fill(~edge_mask[..., None], 0)  # [B, |E|, D]
        return x_v, x_e


class BiasV(nn.Module):
    def __init__(self, dim_out):
        super().__init__()
        self.dim_out = dim_out
        self.b = nn.Parameter(torch.Tensor(1, dim_out))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dim_out)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, x, ehnn_cache):
        node_mask = ehnn_cache['node_mask']
        x = x + self.b
        x = x.masked_fill(~node_mask[..., None], 0)
        return x


class LinearV2E_wo_global(nn.Module):
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
        self.mlp2 = MLP(inner_dim * (2 if CONCAT_PE else 1), inner_dim, hyper_dim, hyper_layers, hyper_dropout)
        self.mlp3 = MLP(inner_dim * (2 if CONCAT_PE else 1), dim_out, hyper_dim, hyper_layers, hyper_dropout)
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
        incidence = ehnn_cache['incidence']  # [B, N, |E|]
        edge_orders = ehnn_cache['edge_orders']  # [B, |E|]
        prefix_normalizer = ehnn_cache['prefix_normalizer']  # [B, |E|]
        node_mask = ehnn_cache['node_mask']  # [B, N]
        edge_mask = ehnn_cache['edge_mask']  # [B, |E|]
        n_nodes = ehnn_cache['n_nodes']  # [B,]

        x = x.clone().masked_fill(~node_mask[..., None], 0)  # [B, N, D]

        # MLP1
        x = x + self.mlp1(self.norm1(x))  # [B, N, D]
        x = x.clone().masked_fill(~node_mask[..., None], 0)  # [B, N, D]
        # aggregation
        x1_v = x  # [B, N, D]
        x1_e = torch.einsum('bne,bnd->bed', incidence, x) / prefix_normalizer[..., None]  # [B, |E|, D]
        x1_e = x1_e.clone().masked_fill(~edge_mask[..., None], 0)  # [B, |E|, D]
        # MLP2
        pe2_s1 = self.pe2(torch.ones(1, dtype=torch.long, device=x.device)).view(1, 1, self.inner_dim)  # [1, 1, D]
        if CONCAT_PE:
            x1_v = x1_v + self.mlp2(torch.cat((self.norm2(x1_v), pe2_s1.expand(x1_v.shape)), dim=-1))
            x1_e = x1_e + self.mlp2(torch.cat((self.norm2(x1_e), pe2_s1.expand(x1_e.shape)), dim=-1))
        else:
            x1_v = x1_v + self.mlp2(self.norm2(x1_v) + pe2_s1)
            x1_e = x1_e + self.mlp2(self.norm2(x1_e) + pe2_s1)

        x_v = x1_v
        x_e = x1_e
        x_v = x_v.clone().masked_fill(~node_mask[..., None], 0)  # [B, N, D]
        x_e = x_e.clone().masked_fill(~edge_mask[..., None], 0)  # [B, |E|, D]
        # MLP3
        pe3_l1 = self.pe3(torch.ones(1, dtype=torch.long, device=x.device)).view(1, 1, self.inner_dim)
        if (self.max_l < edge_orders.size(1) and self.hyper_dropout == 0) or self.force_broadcast:
            # do not use this when hypernetwork is stochastic
            indices = torch.arange(self.pe3.max_pos, device=x.device)
            pe3_l = self.pe3(indices)[edge_orders]  # [B, |E|, D]
        else:
            pe3_l = self.pe3(edge_orders)  # [B, |E|, D]
        if CONCAT_PE:
            x_v = x_v + self.mlp3(torch.cat((self.norm3(x_v), pe3_l1.expand(x_v.shape)), dim=-1))
            x_e = x_e + self.mlp3(torch.cat((self.norm3(x_e), pe3_l.expand(x_e.shape)), dim=-1))
        else:
            x_v = x_v + self.mlp3(self.norm3(x_v) + pe3_l1)
            x_e = x_e + self.mlp3(self.norm3(x_e) + pe3_l)

        x = x_v, x_e
        x = self.b(x, ehnn_cache)
        return x


class LinearE2V_wo_global(nn.Module):
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
        self.mlp1 = MLP(dim_in * (2 if CONCAT_PE else 1), inner_dim, hyper_dim, hyper_layers, hyper_dropout)
        self.mlp2 = MLP(inner_dim * (2 if CONCAT_PE else 1), inner_dim, hyper_dim, hyper_layers, hyper_dropout)
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
        incidence = ehnn_cache['incidence']  # [B, N, |E|]
        edge_orders = ehnn_cache['edge_orders']  # [B, |E|]
        suffix_normalizer = ehnn_cache['suffix_normalizer']  # [B, N]
        node_mask = ehnn_cache['node_mask']  # [B, N]
        edge_mask = ehnn_cache['edge_mask']  # [B, |E|]
        n_nodes = ehnn_cache['n_nodes']  # [B,]
        n_edges = ehnn_cache['n_edges']  # [B,]

        x_v, x_e = x
        x_v = x_v.clone().masked_fill(~node_mask[..., None], 0)  # [B, N, D]
        x_e = x_e.clone().masked_fill(~edge_mask[..., None], 0)  # [B, N, D]

        # MLP1
        pe1_k1 = self.pe1(torch.ones(1, dtype=torch.long, device=x_v.device)).view(1, 1, self.dim_in)
        if (self.max_k < edge_orders.size(1) and self.hyper_dropout == 0) or self.force_broadcast:
            # do not use this when hypernetwork is stochastic
            indices = torch.arange(self.pe1.max_pos, device=x_v.device)
            pe1_k = self.pe1(indices)[edge_orders]  # [B, |E|, D]
        else:
            pe1_k = self.pe1(edge_orders)  # [B, |E|, D]
        if CONCAT_PE:
            x_v = x_v + self.mlp1(torch.cat((self.norm1(x_v), pe1_k1.expand(x_v.shape)), dim=-1))
            x_e = x_e + self.mlp1(torch.cat((self.norm1(x_e), pe1_k.expand(x_e.shape)), dim=-1))
        else:
            x_v = x_v + self.mlp1(self.norm1(x_v) + pe1_k1)
            x_e = x_e + self.mlp1(self.norm1(x_e) + pe1_k)
        x_v = x_v.clone().masked_fill(~node_mask[..., None], 0)  # [B, N, D]
        x_e = x_e.clone().masked_fill(~edge_mask[..., None], 0)  # [B, |E|, D]

        # aggregation
        x1 = x_v + torch.einsum('bne,bed->bnd', incidence, x_e) / (1 + suffix_normalizer[..., None])  # [B, N, D]
        # MLP2
        pe2_s1 = self.pe2(torch.ones(1, dtype=torch.long, device=x_v.device)).view(1, 1, self.inner_dim)  # [1, 1, D]
        if CONCAT_PE:
            x1 = x1 + self.mlp2(torch.cat((self.norm2(x1), pe2_s1.expand(x1.shape)), dim=-1))
        else:
            x1 = x1 + self.mlp2(self.norm2(x1) + pe2_s1)

        x = x1
        x = x.clone().masked_fill(~node_mask[..., None], 0)  # [B, N, D]

        # MLP3
        x = x + self.mlp3(self.norm3(x))  # [N, D']
        x = self.b(x, ehnn_cache)
        return x
