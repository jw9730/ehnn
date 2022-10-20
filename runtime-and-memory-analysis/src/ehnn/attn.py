import torch
import torch.nn as nn

import math

from .linear import BiasE, BiasV
from .hypernetwork import PositionalEncoding, PositionalMLP, PositionalWeight
from .mlp import MLP


def util_sparse_einsum(abh: torch.sparse.Tensor, bhd: torch.Tensor) -> torch.Tensor:
    """abh,bhd->ahd"""
    a, b, h = abh.size()
    b_, h_, d = bhd.size()
    assert b == b_ and h == h_
    indices, values = abh.indices(), abh.values()  # [2, |indices|], [|indices|, h]
    # elementwise multiplication -> row summation
    res = torch.sparse_coo_tensor(indices[0:1], values[..., None] * bhd[indices[1]], size=(a, h, d))  # [a, h, d]
    return res.coalesce().to_dense()  # [a, h, d]


class SelfAttnV2E(nn.Module):
    def __init__(self, dim_in, dim_qk, n_heads, hypernet_info, inner_dim, pe_dim, hyper_dim, hyper_layers,
                 hyper_dropout, force_broadcast, att0_dropout, att1_dropout):
        super().__init__()
        _, max_l, _ = hypernet_info
        self.max_l = max_l
        self.dim_in = dim_in
        self.dim_qk = dim_qk
        self.n_heads = n_heads
        self.inner_dim = inner_dim
        self.dim_v = inner_dim
        self.pe_dim = pe_dim
        self.hyper_dim = hyper_dim
        self.hyper_layers = hyper_layers
        self.hyper_dropout = hyper_dropout
        self.force_broadcast = force_broadcast
        self.dim_qk_head = dim_qk // n_heads if dim_qk >= n_heads else 1
        self.dim_v_head = self.dim_v // n_heads if self.dim_v >= n_heads else 1
        self.q = PositionalMLP(n_heads * self.dim_qk_head, 2, pe_dim, hyper_dim, hyper_layers, hyper_dropout)
        self.k = nn.Linear(self.dim_v, 2 * dim_qk)
        self.v = nn.Linear(self.dim_v, self.dim_v)

        self.mlp1 = MLP(dim_in, self.dim_v, hyper_dim, hyper_layers, hyper_dropout)
        self.mlp2 = MLP(self.dim_v * 2, self.dim_v, hyper_dim, hyper_layers, hyper_dropout)
        self.mlp3 = MLP(self.dim_v * 2, dim_in, hyper_dim, hyper_layers, hyper_dropout)
        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm2 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)
        self.pe2 = PositionalEncoding(inner_dim, 2)
        self.pe3 = PositionalEncoding(inner_dim, max_l + 1)
        self.b = BiasE(dim_in, max_l, pe_dim, hyper_dim, hyper_layers, hyper_dropout, force_broadcast)

        self.att0_dropout = nn.Dropout(att0_dropout)
        self.att1_dropout = nn.Dropout(att1_dropout)

    def reset_parameters(self):
        self.q.reset_parameters()
        self.k.reset_parameters()
        self.v.reset_parameters()
        self.mlp1.reset_parameters()
        self.mlp2.reset_parameters()
        self.mlp3.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm3.reset_parameters()
        self.b.reset_parameters()

    def forward(self, x, ehnn_cache) -> [torch.Tensor, torch.Tensor]:
        """forward method
        :param x: [N, D]
        :param ehnn_cache:
        :return: [N, D], [|E|, D]
        """
        incidence = ehnn_cache['incidence']
        edge_orders = ehnn_cache['edge_orders']

        # MLP1
        x = x + self.mlp1(self.norm1(x))  # [N, D]

        q_0 = self.q(torch.zeros(1, dtype=torch.long, device=x.device)).view(self.n_heads, self.dim_qk_head)
        q_1 = self.q(torch.ones(1, dtype=torch.long, device=x.device)).view(self.n_heads, self.dim_qk_head)
        k = self.k(x)
        k_0, k_1 = k.split(self.dim_qk, -1)
        k_0 = torch.stack(k_0.split(self.dim_qk_head, -1), 1)  # [N, H, D/H]
        k_1 = torch.stack(k_1.split(self.dim_qk_head, -1), 1)  # [N, H, D/H]
        v = self.v(x)
        v = torch.stack(v.split(self.dim_v_head, -1), 1)  # [N, H, Dv/H]

        # aggregation
        # either 0-overlap (global), or 1-overlap (local)
        # 0-overlap
        logit0 = torch.einsum('hd,nhd->nh', q_0, k_0) / math.sqrt(self.dim_qk_head)  # [N, H]
        alpha0 = torch.softmax(logit0, dim=0)  # [N, H]
        att0 = torch.einsum('nh,nhd->hd', alpha0, v)  # [H, Dv/H]
        att0 = torch.cat(att0.unbind(-2), -1)  # [Dv,]
        att0 = att0[None, :]  # [1, Dv]
        # 1-overlap
        # 1-overlap node-to-node
        att1_v = torch.cat(v.unbind(-2), -1)  # [N, Dv]
        # 1-overlap node-to-edge
        logit1 = torch.einsum('hd,nhd->nh', q_1, k_1) / math.sqrt(self.dim_qk_head)  # [N, H]
        # broadcast logits over incidence matrix
        n, e = incidence.size()
        indices = incidence.indices()[[1, 0]]  # [2, |indices|], indices of transposed incidence matrix

        logit1 = torch.sparse_coo_tensor(indices, logit1[indices[1]], size=(e, n, self.n_heads))  # [|E|, N, H]
        alpha1 = torch.sparse.softmax(logit1, dim=1).coalesce()  # [|E|, N, H]
        att1_e = util_sparse_einsum(alpha1, v)  # [|E|, H, Dv/H]
        att1_e = torch.cat(att1_e.unbind(-2), -1)  # [|E|, Dv]

        # MLP2
        pe2_s0 = self.pe2(torch.zeros(1, dtype=torch.long, device=x.device)).view(1, self.inner_dim)  # [1, D]
        pe2_s1 = self.pe2(torch.ones(1, dtype=torch.long, device=x.device)).view(1, self.inner_dim)  # [1, D]
        att0 = att0 + self.mlp2(torch.cat((self.norm2(att0), pe2_s0.expand(att0.shape)), dim=-1))
        att1_v = att1_v + self.mlp2(torch.cat((self.norm2(att1_v), pe2_s1.expand(att1_v.shape)), dim=-1))
        att1_e = att1_e + self.mlp2(torch.cat((self.norm2(att1_e), pe2_s1.expand(att1_e.shape)), dim=-1))

        x_v = self.att0_dropout(att0) + self.att1_dropout(att1_v)
        x_e = self.att0_dropout(att0) + self.att1_dropout(att1_e)
        # MLP3
        pe3_l1 = self.pe3(torch.ones(1, dtype=torch.long, device=x.device)).view(1, self.inner_dim)
        if (self.max_l < len(edge_orders) and self.hyper_dropout == 0) or self.force_broadcast:
            # do not use this when hypernetwork is stochastic
            indices = torch.arange(self.pe3.max_pos, device=x.device)
            pe3_l = self.pe3(indices)[edge_orders]  # [|E|, D]
        else:
            pe3_l = self.pe3(edge_orders)  # [|E|, D]
        x_v = x_v + self.mlp3(torch.cat((self.norm3(x_v), pe3_l1.expand(x_v.shape)), dim=-1))
        x_e = x_e + self.mlp3(torch.cat((self.norm3(x_e), pe3_l.expand(x_e.shape)), dim=-1))

        x = x_v, x_e
        x = self.b(x, edge_orders)
        return x


class SelfAttnE2V(nn.Module):
    def __init__(self, dim_in, dim_qk, n_heads, hypernet_info, inner_dim, pe_dim, hyper_dim, hyper_layers,
                 hyper_dropout, force_broadcast, att0_dropout, att1_dropout):
        super().__init__()
        max_k, _, _ = hypernet_info
        self.max_k = max_k
        self.dim_in = dim_in
        self.dim_qk = dim_qk
        self.n_heads = n_heads
        self.dim_v = inner_dim
        self.inner_dim = inner_dim
        self.pe_dim = pe_dim
        self.hyper_dim = hyper_dim
        self.hyper_layers = hyper_layers
        self.hyper_dropout = hyper_dropout
        self.force_broadcast = force_broadcast
        self.dim_qk_head = dim_qk // n_heads if dim_qk >= n_heads else 1
        self.dim_v_head = self.dim_v // n_heads if self.dim_v >= n_heads else 1
        self.q = PositionalMLP(n_heads * self.dim_qk_head, 2, pe_dim, hyper_dim, hyper_layers, hyper_dropout)
        self.k = nn.Linear(self.dim_v, 2 * dim_qk)
        self.v = nn.Linear(self.dim_v, self.dim_v)

        self.mlp1 = MLP(dim_in * 2, self.dim_v, hyper_dim, hyper_layers, hyper_dropout)
        self.mlp2 = MLP(self.dim_v * 2, self.dim_v, hyper_dim, hyper_layers, hyper_dropout)
        self.mlp3 = MLP(self.dim_v, dim_in, hyper_dim, hyper_layers, hyper_dropout)
        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm2 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)
        # node-input only, don't need pe3
        self.pe1 = PositionalEncoding(dim_in, max_k + 1)
        self.pe2 = PositionalEncoding(inner_dim, 2)
        self.b = BiasV(dim_in)

        self.att0_dropout = nn.Dropout(att0_dropout)
        self.att1_dropout = nn.Dropout(att1_dropout)

    def reset_parameters(self):
        self.q.reset_parameters()
        self.k.reset_parameters()
        self.v.reset_parameters()
        self.mlp1.reset_parameters()
        self.mlp2.reset_parameters()
        self.mlp3.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm3.reset_parameters()
        self.b.reset_parameters()

    def forward(self, x, ehnn_cache) -> [torch.Tensor, torch.Tensor]:
        """forward method
        :param x: [N, D]
        :param ehnn_cache:
        :return: [N, D], [|E|, D]
        """
        incidence = ehnn_cache['incidence']
        edge_orders = ehnn_cache['edge_orders']
        indices_with_nodes = ehnn_cache['indices_with_nodes']

        # MLP1
        x_v, x_e = x
        pe1_k1 = self.pe1(torch.ones(1, dtype=torch.long, device=x_v.device)).view(1, self.dim_in)
        if (self.max_k < len(edge_orders) and self.hyper_dropout == 0) or self.force_broadcast:
            # do not use this when hypernetwork is stochastic
            indices = torch.arange(self.pe1.max_pos, device=x_v.device)
            pe1_k = self.pe1(indices)[edge_orders]  # [|E|, D]
        else:
            pe1_k = self.pe1(edge_orders)  # [|E|, D]
        x_v = x_v + self.mlp1(torch.cat((self.norm1(x_v), pe1_k1.expand(x_v.shape)), dim=-1))
        x_e = x_e + self.mlp1(torch.cat((self.norm1(x_e), pe1_k.expand(x_e.shape)), dim=-1))

        q_0 = self.q(torch.zeros(1, dtype=torch.long, device=x_v.device)).view(self.n_heads, self.dim_qk_head)
        q_1 = self.q(torch.ones(1, dtype=torch.long, device=x_v.device)).view(self.n_heads, self.dim_qk_head)
        k_v, k_e = self.k(x_v), self.k(x_e)
        k_v_0, k_v_1 = k_v.split(self.dim_qk, -1)
        k_e_0, k_e_1 = k_e.split(self.dim_qk, -1)
        k_v_0 = torch.stack(k_v_0.split(self.dim_qk_head, -1), 1)  # [N, H, D/H]
        k_v_1 = torch.stack(k_v_1.split(self.dim_qk_head, -1), 1)  # [N, H, D/H]
        k_e_0 = torch.stack(k_e_0.split(self.dim_qk_head, -1), 1)  # [|E|, H, D/H]
        k_e_1 = torch.stack(k_e_1.split(self.dim_qk_head, -1), 1)  # [|E|, H, D/H]
        k_0 = torch.cat([k_e_0, k_v_0])
        v_v, v_e = self.v(x_v), self.v(x_e)
        v_v = torch.stack(v_v.split(self.dim_v_head, -1), 1)  # [N, H, Dv/H]
        v_e = torch.stack(v_e.split(self.dim_v_head, -1), 1)  # [|E|, H, Dv/H]

        # aggregation
        # either 0-overlap (global), or 1-overlap (local)
        v = torch.cat([v_e, v_v])  # [|E| + N, H, Dv/H]
        # 0-overlap
        # softmax (not using kernel)
        logit0 = torch.einsum('hd,ehd->eh', q_0, k_0) / math.sqrt(self.dim_qk_head)  # [|E| + N, H]
        alpha0 = torch.softmax(logit0, dim=0)  # [|E| + N, H]
        att0 = torch.einsum('eh,ehd->hd', alpha0, v)  # [H, Dv/H]
        att0 = torch.cat(att0.unbind(-2), -1)  # [Dv,]
        att0 = att0[None, :]  # [1, Dv]
        # 1-overlap
        logit1 = torch.einsum('hd,nhd->nh', q_1, torch.cat([k_e_1, k_v_1])) / math.sqrt(
            self.dim_qk_head)  # [|E| + N, H]
        n, e = incidence.size()
        logit1 = torch.sparse_coo_tensor(indices_with_nodes, logit1[indices_with_nodes[1]], size=(n, e + n, self.n_heads))  # [N, |E| + N, H]
        alpha1 = torch.sparse.softmax(logit1, dim=1).coalesce()  # [N, |E| + N, H]
        att1 = util_sparse_einsum(alpha1, v)  # [N, H, Dv/H]
        att1 = torch.cat(att1.unbind(-2), -1)  # [N, Dv]

        # MLP2
        pe2_s0 = self.pe2(torch.zeros(1, dtype=torch.long, device=x_v.device)).view(1, self.inner_dim)  # [1, D]
        pe2_s1 = self.pe2(torch.ones(1, dtype=torch.long, device=x_v.device)).view(1, self.inner_dim)  # [1, D]
        att0 = att0 + self.mlp2(torch.cat((self.norm2(att0), pe2_s0.expand(att0.shape)), dim=-1))
        att1 = att1 + self.mlp2(torch.cat((self.norm2(att1), pe2_s1.expand(att1.shape)), dim=-1))

        # output has only 1-edges
        x = self.att0_dropout(att0) + self.att1_dropout(att1)
        # MLP3
        x = x + self.mlp3(self.norm3(x))  # [N, D']
        x = self.b(x)
        return x


class TransformerV2E(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_ff, n_heads, dropout,
                 hypernet_info, inner_dim, pe_dim, hyper_dim, hyper_layers, hyper_dropout, force_broadcast, att0_dropout, att1_dropout):
        super().__init__()
        self.ln = nn.LayerNorm(dim_in)
        self.attn = SelfAttnV2E(dim_in, dim_qk, n_heads, hypernet_info, inner_dim, pe_dim, hyper_dim,
                                hyper_layers, hyper_dropout, force_broadcast, att0_dropout, att1_dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, dim_ff),
            nn.Dropout(dropout, inplace=True),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_in)
        )

    def reset_parameters(self):
        self.ln.reset_parameters()
        self.attn.reset_parameters()
        for layer in self.ffn:
            if isinstance(layer, nn.LayerNorm):
                layer.reset_parameters()
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x, ehnn_cache) -> [torch.Tensor, torch.Tensor]:
        h = self.ln(x)
        h_v, h_e = self.attn(h, ehnn_cache)
        x_v, x_e = (x + h_v, h_e)
        h_v, h_e = (self.ffn(x_v), self.ffn(x_e))
        x_v, x_e = (x_v + h_v, x_e + h_e)
        return x_v, x_e


class TransformerE2V(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_ff, n_heads, dropout,
                 hypernet_info, inner_dim, pe_dim, hyper_dim, hyper_layers, hyper_dropout, force_broadcast, att0_dropout, att1_dropout):
        super().__init__()
        self.ln = nn.LayerNorm(dim_in)
        self.attn = SelfAttnE2V(dim_in, dim_qk, n_heads, hypernet_info, inner_dim, pe_dim, hyper_dim,
                                hyper_layers, hyper_dropout, force_broadcast, att0_dropout, att1_dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, dim_ff),
            nn.Dropout(dropout, inplace=True),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_in)
        )

    def reset_parameters(self):
        self.ln.reset_parameters()
        self.attn.reset_parameters()
        for layer in self.ffn:
            if isinstance(layer, nn.LayerNorm):
                layer.reset_parameters()
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x, ehnn_cache) -> torch.Tensor:
        x_v, x_e = x
        h = (self.ln(x_v), self.ln(x_e))
        h = self.attn(h, ehnn_cache)
        x = x_v + h
        h = self.ffn(x)
        x = x + h
        return x