import torch
import torch.nn as nn

import math

from .hypernetwork import PositionalMLP, PositionalWeight, Positional2DMLP, Positional2DWeight


CONCAT_PE = False


class BiasE(nn.Module):
    def __init__(
            self,
            dim_out,
            max_l,
            pe_dim,
            hyper_dim,
            hyper_layers,
            hyper_dropout,
            force_broadcast,
            use_hypernet=False
    ):
        super().__init__()
        self.dim_out = dim_out
        self.max_l = max_l
        self.pe_dim = pe_dim
        self.hyper_dim = hyper_dim
        self.hyper_layers = hyper_layers
        self.hyper_dropout = hyper_dropout
        self.force_broadcast = force_broadcast
        if use_hypernet:
            self.b = PositionalMLP(dim_out, max_l + 1, pe_dim, hyper_dim, hyper_layers, hyper_dropout)
        else:
            self.b = PositionalWeight(max_l + 1, 1, dim_out)

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
            indices = torch.arange(self.max_l + 1, device=x_v.device)
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


class NaiveV2E(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_out,
            hypernet_info,
            inner_dim,
            pe_dim,
            hyper_dim,
            hyper_layers,
            hyper_dropout,
            force_broadcast,
            use_hypernet=False
    ):
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
        if use_hypernet:
            self.w = Positional2DMLP(dim_in * dim_out, 2, max_l + 1, pe_dim, hyper_dim, hyper_layers, hyper_dropout)
        else:
            self.w = Positional2DWeight(2, max_l + 1, dim_in, dim_out)
        self.b = BiasE(dim_out, max_l, pe_dim, hyper_dim, hyper_layers, hyper_dropout, force_broadcast)

    def reset_parameters(self):
        self.w.reset_parameters()
        self.b.reset_parameters()

    def forward(self, x, ehnn_cache) -> [torch.Tensor, torch.Tensor]:
        incidence = ehnn_cache['incidence']  # [B, N, |E|]
        edge_orders = ehnn_cache['edge_orders']  # [B, |E|]
        prefix_normalizer = ehnn_cache['prefix_normalizer']  # [B, |E|]
        node_mask = ehnn_cache['node_mask']  # [B, N]
        edge_mask = ehnn_cache['edge_mask']  # [B, |E|]
        n_nodes = ehnn_cache['n_nodes']  # [B,]

        x = x.clone().masked_fill(~node_mask[..., None], 0)  # [B, N, D]
        b, n, e = incidence.size()

        if (self.max_l and self.hyper_dropout == 0) or self.force_broadcast:
            # do not use this when hypernetwork is stochastic
            indices = torch.arange(self.max_l + 1, device=x.device)
            w0 = self.w(torch.zeros_like(indices), indices)[edge_orders]
            w1 = self.w(torch.ones_like(indices), indices)[edge_orders]
        else:
            w0 = self.w(torch.zeros_like(edge_orders), edge_orders)
            w1 = self.w(torch.ones_like(edge_orders), edge_orders)
        w0 = w0.view(b, e, self.dim_in, self.dim_out)  # [B, |E|, D, D']
        w1 = w1.view(b, e, self.dim_in, self.dim_out)  # [B, |E|, D, D']
        ones = torch.ones(1, dtype=torch.long, device=x.device)
        w0_1 = self.w(torch.zeros_like(ones), ones).view(self.dim_in, self.dim_out)  # [D, D']
        w1_1 = self.w(torch.ones_like(ones), ones).view(self.dim_in, self.dim_out)  # [D, D']

        # aggregation
        x0 = x.sum(dim=1, keepdim=True) / n_nodes[:, None, None]  # [B, 1, D]
        x1_v = x  # [B, N, D]
        x1_e = torch.einsum('bne,bnd->bed', incidence, x) / prefix_normalizer[..., None]  # [B, |E|, D]
        x1_e = x1_e.clone().masked_fill(~edge_mask[..., None], 0)  # [B, |E|, D]

        # weight application
        x_v = torch.einsum('...i,ij->...j', x0, w0_1) + torch.einsum('...i,ij->...j', x1_v, w1_1)
        x_e = torch.einsum('...i,...ij->...j', x0, w0) + torch.einsum('...i,...ij->...j', x1_e, w1)
        x_v = x_v.clone().masked_fill(~node_mask[..., None], 0)  # [B, N, D]
        x_e = x_e.clone().masked_fill(~edge_mask[..., None], 0)  # [B, |E|, D]

        x = x_v, x_e
        x = self.b(x, ehnn_cache)
        return x


class NaiveE2V(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_out,
            hypernet_info,
            inner_dim,
            pe_dim,
            hyper_dim,
            hyper_layers,
            hyper_dropout,
            force_broadcast,
            use_hypernet=False
    ):
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
        if use_hypernet:
            self.w = Positional2DMLP(dim_in * dim_out, 2, max_k + 1, pe_dim, hyper_dim, hyper_layers, hyper_dropout)
        else:
            self.w = Positional2DWeight(2, max_k + 1, dim_in, dim_out)
        self.b = BiasV(dim_out)

    def reset_parameters(self):
        self.w.reset_parameters()
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
        b, n, e = incidence.size()

        if (self.max_k and self.hyper_dropout == 0) or self.force_broadcast:
            # do not use this when hypernetwork is stochastic
            indices = torch.arange(self.max_k + 1, device=x_v.device)
            w0 = self.w(torch.zeros_like(indices), indices)[edge_orders]
            w1 = self.w(torch.ones_like(indices), indices)[edge_orders]
        else:
            w0 = self.w(torch.zeros_like(edge_orders), edge_orders)
            w1 = self.w(torch.ones_like(edge_orders), edge_orders)
        w0 = w0.view(b, e, self.dim_in, self.dim_out)  # [B, |E|, D, D']
        w1 = w1.view(b, e, self.dim_in, self.dim_out)  # [B, |E|, D, D']
        ones = torch.ones(1, dtype=torch.long, device=x_v.device)
        w0_1 = self.w(torch.zeros_like(ones), ones).view(self.dim_in, self.dim_out)  # [D, D']
        w1_1 = self.w(torch.ones_like(ones), ones).view(self.dim_in, self.dim_out)  # [D, D']

        x_v = x_v.clone().masked_fill(~node_mask[..., None], 0)  # [B, N, D]
        x_e = x_e.clone().masked_fill(~edge_mask[..., None], 0)  # [B, |E|, D]

        # weight application
        x0_v = torch.einsum('...i,ij->...j', x_v, w0_1)
        x0_e = torch.einsum('...i,...ij->...j', x_e, w0)
        x1_v = torch.einsum('...i,ij->...j', x_v, w1_1)
        x1_e = torch.einsum('...i,...ij->...j', x_e, w1)

        # aggregation
        x0 = torch.cat((x0_v, x0_e), dim=1).sum(dim=1, keepdim=True) / (n_nodes + n_edges)[:, None, None]  # [B, 1, D]
        x1_e = torch.einsum('bne,bed->bnd', incidence, x1_e) / (1 + suffix_normalizer[..., None])  # [B, N, D]
        x1_v = x1_v  # [B, N, D]

        x = x0 + x1_e + x1_v
        x = x.clone().masked_fill(~node_mask[..., None], 0)  # [B, N, D]

        x = self.b(x, ehnn_cache)
        return x
