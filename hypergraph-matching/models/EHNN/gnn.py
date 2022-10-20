import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from collections import Iterable
import math

from src.lap_solvers.sinkhorn import Sinkhorn

from models.EHNN.mlp import MLP
from models.EHNN.hypernetwork import PositionalEncoding, PositionalMLP


class HyperGNNLayer(nn.Module):
    def __init__(self, in_node_features, in_edge_features, out_node_features, out_edge_features, orders=3, eps=1e-10,
                 sk_channel=False, sk_iter=20, sk_tau=0.05, transformer=False, n_heads=1,
                 ffn_dropout=0, att0_dropout=0, att1_dropout=0):
        super(HyperGNNLayer, self).__init__()
        self.in_nfeat = in_node_features
        self.in_efeat = in_edge_features
        self.out_efeat = out_edge_features
        self.eps = eps
        self.sk_channel = sk_channel
        assert out_node_features == out_edge_features + self.sk_channel
        if self.sk_channel > 0:
            self.out_nfeat = out_node_features - self.sk_channel
            self.sk = Sinkhorn(sk_iter, sk_tau)
            self.classifier = nn.Linear(self.out_nfeat, self.sk_channel)
        else:
            self.out_nfeat = out_node_features
            self.sk = self.classifier = None
        # ========================================================================
        self.transformer = transformer
        if self.transformer:
            print('USING TRANSFORMER %d HEADS' % n_heads)
        else:
            print('NOT USING TRANSFORMER')
        # mlp
        hyper_layers = 3
        hyper_dropout = 0
        self.ffn_dropout = ffn_dropout
        self.att0_dropout = att0_dropout
        self.att1_dropout = att1_dropout
        self.hidden_dim = self.out_nfeat
        self.n_heads = n_heads
        self.x_input = nn.Linear(self.in_nfeat, self.hidden_dim)
        self.W2_input = nn.Linear(1, self.hidden_dim)
        self.W3_input = nn.Linear(1, self.hidden_dim)
        self.mlp1 = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, hyper_layers, hyper_dropout)
        self.mlp2 = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, hyper_layers, hyper_dropout)
        self.mlp3 = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, hyper_layers, hyper_dropout)
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)
        self.norm4 = nn.LayerNorm(self.hidden_dim)
        self.pe1 = PositionalEncoding(self.hidden_dim, 4)
        self.pe2 = PositionalEncoding(self.hidden_dim, 2)
        self.out = nn.Linear(self.hidden_dim, self.out_nfeat)
        # bias (self.b) is absorbed to mlp3
        # attention
        self.dim_qk = self.hidden_dim * self.n_heads
        #self.dim_qk = self.hidden_dim
        self.dim_v = self.hidden_dim
        self.dim_qk_head = self.dim_qk // self.n_heads if self.dim_qk >= self.n_heads else 1
        self.dim_v_head = self.dim_v // self.n_heads if self.dim_v >= self.n_heads else 1
        self.ln = nn.LayerNorm(self.hidden_dim)
        self.q = PositionalMLP(self.n_heads * self.dim_qk_head, 2, self.hidden_dim, self.hidden_dim, hyper_layers, hyper_dropout)
        self.k = nn.Linear(self.hidden_dim, 2 * self.dim_qk)
        self.v = nn.Linear(self.hidden_dim, self.dim_v)
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.dim_v),
            nn.Linear(self.dim_v, self.hidden_dim),
            nn.Dropout(self.ffn_dropout, inplace=True),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.dim_v)
        )

        # ========================================================================

    def forward(self, A, W, x, n1=None, n2=None, weight=None, norm=True):
        """wrapper function of forward (support dense/sparse)"""
        if not isinstance(A, Iterable):
            A = [A]
            W = [W]

        W2 = W[0].coalesce()
        W3 = W[1].coalesce()
        x_k1 = self.x_input(x)  # [B, N, D]
        x_k2 = self.W2_input(W2.values())  # [|E2|, D]
        x_k3 = self.W3_input(W3.values())  # [|E3|, D]
        x_k2_indices = W2.indices()  # [3, |E2|] (b, i1, i2), symmetric (if (i1, i2) exists, so does (i2, i1))
        x_k3_indices = W3.indices()  # [4, |E3|] (b, i1, i2, i3), supersymmetric

        if self.transformer:
            h_k1, h_k2, h_k3 = self.ln(x_k1), self.ln(x_k2), self.ln(x_k3)
            x = x_k1 + self.forward_e2v_attn(h_k1, h_k2, h_k3, x_k2_indices, x_k3_indices, n1, n2)
            x = x + self.ffn(x)
        else:
            x = x_k1 + self.forward_e2v_linear(x_k1, x_k2, x_k3, x_k2_indices, x_k3_indices, n1, n2)
        x2 = x

        if self.classifier is not None:
            assert n1.max() * n2.max() == x.shape[1]
            x2 = self.out(self.norm4(x2))
            x3 = self.classifier(x2)
            n1_rep = torch.repeat_interleave(n1, self.sk_channel, dim=0)
            n2_rep = torch.repeat_interleave(n2, self.sk_channel, dim=0)
            x4 = x3.permute(0, 2, 1).reshape(x.shape[0] * self.sk_channel, n2.max(), n1.max()).transpose(1, 2)
            x5 = self.sk(x4, n1_rep, n2_rep, dummy_row=True).transpose(2, 1).contiguous()
            x6 = x5.reshape(x.shape[0], self.sk_channel, n1.max() * n2.max()).permute(0, 2, 1)
            x_new = torch.cat((x2, x6), dim=-1)

        else:
            x_new = self.out(x2)
        return W, x_new

    def forward_e2v_linear(self, x_k1, x_k2, x_k3, x_k2_indices, x_k3_indices, n1, n2):
        b, n, _ = x_k1.size()
        device = x_k1.device
        e2 = x_k2.size(0)
        e3 = x_k3.size(0)
        if n1 is not None and n2 is not None:
            n_vec = n1 * n2  # [B,]
            node_mask = (torch.arange(x_k1.size(1), device=device)[None, :] - n_vec[:, None] < 0)  # [B, N]
        else:
            n_vec = n * torch.ones(b, device=device)  # [B,]
            node_mask = torch.ones(x_k1.size()[:-1], device=device).bool()  # [B, N]
        assert (node_mask.float().sum(1) - n_vec == 0).all()

        # MLP1
        pe1_k1 = self.pe1(1 * torch.ones(1, dtype=torch.long, device=device)).view(1, 1, self.hidden_dim)
        pe1_k2 = self.pe1(2 * torch.ones(1, dtype=torch.long, device=device)).view(1, self.hidden_dim)
        pe1_k3 = self.pe1(3 * torch.ones(1, dtype=torch.long, device=device)).view(1, self.hidden_dim)
        x_k1 = x_k1 + self.mlp1(self.norm1(x_k1) + pe1_k1)  # [B, N, D]
        x_k2 = x_k2 + self.mlp1(self.norm1(x_k2) + pe1_k2)  # [|E2|, D]
        x_k3 = x_k3 + self.mlp1(self.norm1(x_k3) + pe1_k3)  # [|E3|, D]
        x_k1 = x_k1.clone().masked_fill(~node_mask[..., None], 0)

        # aggregation
        x0_k1 = x_k1.sum(1)
        x0_k1_normalizer = n_vec[:, None]
        x0_k2 = torch.sparse_coo_tensor(
            x_k2_indices[0:1, :],
            torch.cat((x_k2, torch.ones(e2, 1, device=device)), dim=-1),  # [|E2|, D + 1]
            size=(b, self.hidden_dim + 1)  # [B, D + 1]
        ).coalesce().to_dense()
        x0_k3 = torch.sparse_coo_tensor(
            x_k3_indices[0:1, :],
            torch.cat((x_k3, torch.ones(e3, 1, device=device)), dim=-1),  # [|E3|, D + 1]
            size=(b, self.hidden_dim + 1)  # [B, D + 1]
        ).coalesce().to_dense()
        x0_k2, x0_k2_normalizer = x0_k2[..., :-1], x0_k2[..., -1:]  # [B, D], [B, 1]
        x0_k3, x0_k3_normalizer = x0_k3[..., :-1], x0_k3[..., -1:]  # [B, D], [B, 1]
        x0 = (x0_k1 + x0_k2 / 2 + x0_k3 / 3) / (x0_k1_normalizer + x0_k2_normalizer + x0_k3_normalizer)  # [B, D]
        x0 = x0[:, None, :]  # [B, 1, D]
        x1_k1 = x_k1  # [B, N, D]
        x1_k2 = torch.sparse_coo_tensor(
            x_k2_indices[0:2, :],
            torch.cat((x_k2, torch.ones(e2, 1, device=device)), dim=-1),  # [|E2|, D + 1]
            size=(b, n, self.hidden_dim + 1)  # [B, N, D + 1]
        ).coalesce().to_dense()
        x1_k3 = torch.sparse_coo_tensor(
            x_k3_indices[0:2, :],
            torch.cat((x_k3, torch.ones(e3, 1, device=device)), dim=-1),  # [|E3|, D + 1]
            size=(b, n, self.hidden_dim + 1)  # [B, N, D + 1]
        ).coalesce().to_dense()
        x1_k2, x1_k2_normalizer = x1_k2[..., :-1], x1_k2[..., -1:]  # [B, N, D], [B, N, 1]
        x1_k3, x1_k3_normalizer = x1_k3[..., :-1], x1_k3[..., -1:]  # [B, N, D], [B, N, 1]
        x1 = (x1_k1 + x1_k2 + x1_k3 / 2) / (1 + x1_k2_normalizer + x1_k3_normalizer)  # [B, N, D]

        # MLP2
        pe2_s0 = self.pe2(torch.zeros(1, dtype=torch.long, device=device)).view(1, 1, self.hidden_dim)
        pe2_s1 = self.pe2(torch.ones(1, dtype=torch.long, device=device)).view(1, 1, self.hidden_dim)
        x0 = x0 + self.mlp2(self.norm2(x0) + pe2_s0)
        x1 = x1 + self.mlp2(self.norm2(x1) + pe2_s1)
        x = x0 + x1

        # MLP3
        x = x + self.mlp3(self.norm3(x))  # [B, N, D]
        x = x.clone().masked_fill(~node_mask[..., None], 0)
        return x

    def forward_e2v_attn(self, x_k1, x_k2, x_k3, x_k2_indices, x_k3_indices, n1, n2):
        b, n, _ = x_k1.size()
        device = x_k1.device
        if n1 is not None and n2 is not None:
            n_vec = n1 * n2  # [B,]
            node_mask = (torch.arange(x_k1.size(1), device=device)[None, :] - n_vec[:, None] < 0)  # [B, N]
        else:
            n_vec = n * torch.ones(b, device=device)  # [B,]
            node_mask = torch.ones(x_k1.size()[:-1], device=device).bool()  # [B, N]
        assert (node_mask.float().sum(1) - n_vec == 0).all()

        # MLP1
        pe1_k1 = self.pe1(1 * torch.ones(1, dtype=torch.long, device=device)).view(1, 1, self.hidden_dim)
        pe1_k2 = self.pe1(2 * torch.ones(1, dtype=torch.long, device=device)).view(1, self.hidden_dim)
        pe1_k3 = self.pe1(3 * torch.ones(1, dtype=torch.long, device=device)).view(1, self.hidden_dim)
        x_k1 = x_k1 + self.mlp1(self.norm1(x_k1) + pe1_k1)  # [B, N, D]
        x_k2 = x_k2 + self.mlp1(self.norm1(x_k2) + pe1_k2)  # [|E2|, D]
        x_k3 = x_k3 + self.mlp1(self.norm1(x_k3) + pe1_k3)  # [|E3|, D]
        x_k1 = x_k1.clone().masked_fill(~node_mask[..., None], 0)

        q_0 = self.q(torch.zeros(1, dtype=torch.long, device=device)).view(self.n_heads, self.dim_qk_head)
        q_1 = self.q(torch.ones(1, dtype=torch.long, device=device)).view(self.n_heads, self.dim_qk_head)
        k_k1_0, k_k1_1 = self.k(x_k1).split(self.dim_qk, -1)
        k_k2_0, k_k2_1 = self.k(x_k2).split(self.dim_qk, -1)
        k_k3_0, k_k3_1 = self.k(x_k3).split(self.dim_qk, -1)
        k_k1_0 = torch.stack(k_k1_0.split(self.dim_qk_head, -1), -2)  # [B, N, H, D/H]
        k_k1_1 = torch.stack(k_k1_1.split(self.dim_qk_head, -1), -2)  # [B, N, H, D/H]
        k_k2_0 = torch.stack(k_k2_0.split(self.dim_qk_head, -1), -2)  # [|E2|, H, D/H]
        k_k2_1 = torch.stack(k_k2_1.split(self.dim_qk_head, -1), -2)  # [|E2|, H, D/H]
        k_k3_0 = torch.stack(k_k3_0.split(self.dim_qk_head, -1), -2)  # [|E3|, H, D/H]
        k_k3_1 = torch.stack(k_k3_1.split(self.dim_qk_head, -1), -2)  # [|E3|, H, D/H]
        v_k1 = torch.stack(self.v(x_k1).split(self.dim_v_head, -1), -2)  # [B, N, H, Dv/H]
        v_k2 = torch.stack(self.v(x_k2).split(self.dim_v_head, -1), -2)  # [|E2|, H, Dv/H]
        v_k3 = torch.stack(self.v(x_k3).split(self.dim_v_head, -1), -2)  # [|E3|, H, Dv/H]
        k_k1_0 = k_k1_0.clone().masked_fill(~node_mask[..., None, None], 0)  # [B, N, H, D/H]
        k_k1_1 = k_k1_1.clone().masked_fill(~node_mask[..., None, None], 0)  # [B, N, H, D/H]
        v_k1 = v_k1.clone().masked_fill(~node_mask[..., None, None], 0)  # [B, N, H, Dv/H]

        # aggregation
        # either 0-overlap (global), or 1-overlap (local)
        # 0-overlap
        logit_k1 = torch.einsum('hd,bnhd->bnh', q_0, k_k1_0) / math.sqrt(self.dim_qk_head)  # [B, N, H]
        logit_k2 = torch.einsum('hd,ehd->eh', q_0, k_k2_0) / math.sqrt(self.dim_qk_head)  # [|E2|, H]
        logit_k3 = torch.einsum('hd,ehd->eh', q_0, k_k3_0) / math.sqrt(self.dim_qk_head)  # [|E3|, H]
        logit_k1 = logit_k1.clone().masked_fill(~node_mask[..., None], float('-inf'))  # [B, N, H]

        max_logit = max(logit_k1.max(), logit_k2.max(), logit_k3.max())
        exp_k1 = torch.exp(logit_k1 - max_logit)[..., None]  # [B, N, H, 1]
        exp_k2 = torch.exp(logit_k2 - max_logit)[..., None]  # [|E2|, H, 1]
        exp_k3 = torch.exp(logit_k3 - max_logit)[..., None]  # [|E3|, H, 1]
        exp_k1 = exp_k1.clone().masked_fill(~node_mask[..., None, None], 0)  # [B, N, H, 1]

        att_k1 = (v_k1 * exp_k1).sum(1)  # [B, H, Dv/H]
        exp_sum_k1 = exp_k1.sum(1)  # [B, H, 1]
        att_k2 = torch.sparse_coo_tensor(
            x_k2_indices[0:1, :],
            torch.cat((v_k2 * exp_k2, exp_k2), dim=-1),  # [|E2|, H, Dv/H + 1]
            size=(b, self.n_heads, self.dim_v_head + 1)  # [B, H, Dv/H + 1]
        ).coalesce().to_dense()
        att_k3 = torch.sparse_coo_tensor(
            x_k3_indices[0:1, :],
            torch.cat((v_k3 * exp_k3, exp_k3), dim=-1),  # [|E3|, H, Dv/H + 1]
            size=(b, self.n_heads, self.dim_v_head + 1)  # [B, H, Dv/H + 1]
        ).coalesce().to_dense()
        att_k2, exp_sum_k2 = att_k2[..., :-1], att_k2[..., -1:]  # [B, H, Dv/H], [B, H, 1]
        att_k3, exp_sum_k3 = att_k3[..., :-1], att_k3[..., -1:]  # [B, H, Dv/H], [B, H, 1]
        att0 = (att_k1 + att_k2 + att_k3) / (exp_sum_k1 + exp_sum_k2 + exp_sum_k3)  # [B, H, Dv/H]
        att0 = torch.cat(att0.unbind(-2), -1)  # [B, Dv]
        att0 = att0[:, None, :]  # [B, 1, Dv]

        # 1-overlap
        logit_k1 = torch.einsum('hd,bnhd->bnh', q_1, k_k1_1) / math.sqrt(self.dim_qk_head)  # [B, N, H]
        logit_k2 = torch.einsum('hd,ehd->eh', q_1, k_k2_1) / math.sqrt(self.dim_qk_head)  # [|E2|, H]
        logit_k3 = torch.einsum('hd,ehd->eh', q_1, k_k3_1) / math.sqrt(self.dim_qk_head)  # [|E3|, H]
        logit_k1 = logit_k1.clone().masked_fill(~node_mask[..., None], float('-inf'))  # [B, N, H]

        max_logit = max(logit_k1.max(), logit_k2.max(), logit_k3.max())

        exp_k1 = torch.exp(logit_k1 - max_logit)[..., None]  # [B, N, H, 1]
        exp_k2 = torch.exp(logit_k2 - max_logit)[..., None]  # [|E2|, H, 1]
        exp_k3 = torch.exp(logit_k3 - max_logit)[..., None]  # [|E3|, H, 1]
        exp_k1 = exp_k1.clone().masked_fill(~node_mask[..., None, None], 0)  # [B, N, H, 1]

        att_k1 = v_k1 * exp_k1  # [B, N, H, Dv/H]
        exp_sum_k1 = exp_k1  # [B, N, H, 1]
        att_k2 = torch.sparse_coo_tensor(
            x_k2_indices[0:2, :],
            torch.cat((v_k2 * exp_k2, exp_k2), dim=-1),  # [|E2|, H, Dv/H + 1]
            size=(b, n, self.n_heads, self.dim_v_head + 1)  # [B, N, H, Dv/H + 1]
        ).coalesce().to_dense()
        att_k3 = torch.sparse_coo_tensor(
            x_k3_indices[0:2, :],
            torch.cat((v_k3 * exp_k3, exp_k3), dim=-1),  # [|E3|, H, Dv/H + 1]
            size=(b, n, self.n_heads, self.dim_v_head + 1)  # [B, N, H, Dv/H + 1]
        ).coalesce().to_dense()
        att_k2, exp_sum_k2 = att_k2[..., :-1], att_k2[..., -1:]  # [B, N, H, Dv/H], [B, N, H, 1]
        att_k3, exp_sum_k3 = att_k3[..., :-1], att_k3[..., -1:]  # [B, N, H, Dv/H], [B, N, H, 1]

        att1 = (att_k1 + att_k2 + att_k3) / (exp_sum_k1 + exp_sum_k2 + exp_sum_k3 + 1e-12)  # [B, N, H, Dv/H]

        att1 = torch.cat(att1.unbind(-2), -1)  # [B, N, Dv]

        # MLP2
        pe2_s0 = self.pe2(torch.zeros(1, dtype=torch.long, device=device)).view(1, self.hidden_dim)  # [1, D]
        pe2_s1 = self.pe2(torch.ones(1, dtype=torch.long, device=device)).view(1, self.hidden_dim)  # [1, D]
        att0 = att0 + self.mlp2(self.norm2(att0) + pe2_s0)
        att1 = att1 + self.mlp2(self.norm2(att1) + pe2_s1)

        x = nn.Dropout(self.att0_dropout, inplace=True)(att0) + nn.Dropout(self.att1_dropout, inplace=True)(att1)

        # MLP3
        x = x + self.mlp3(self.norm3(x))  # [B, N, D]

        x = x.clone().masked_fill(~node_mask[..., None], 0)

        return x
