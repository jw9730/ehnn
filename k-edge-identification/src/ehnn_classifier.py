import torch
import torch.nn as nn

import ast

from ehnn import EHNNLinear, EHNNTransformer
from ehnn import EHNNNaive, EHNNLinear_wo_global, EHNNLinear_wo_order, EHNNLinear_wo_global_order


class EHNNClassifier(nn.Module):
    def __init__(self, args, max_edge_order, max_overlap):
        super().__init__()
        self.device = args.device
        hypernet_info = (max_edge_order, max_edge_order, max_overlap)
        self.input = nn.Embedding(args.num_features, args.ehnn_hidden_channel)
        print(args.ehnn_type)
        if args.ehnn_type == 'linear':
            self.model = EHNNLinear(
                args.ehnn_hidden_channel, args.num_classes, args.ehnn_hidden_channel, args.ehnn_inner_channel, args.ehnn_n_layers,
                args.dropout, hypernet_info, args.ehnn_pe_dim, args.ehnn_hyper_dim, args.ehnn_hyper_layers,
                args.ehnn_hyper_dropout, ast.literal_eval(args.ehnn_force_broadcast), args.ehnn_input_dropout,
                ast.literal_eval(args.ehnn_mlp_classifier), args.Classifier_hidden, args.Classifier_num_layers, args.normalization
            )
        elif args.ehnn_type == 'linear_wo_global':
            self.model = EHNNLinear_wo_global(
                args.ehnn_hidden_channel, args.num_classes, args.ehnn_hidden_channel, args.ehnn_inner_channel, args.ehnn_n_layers,
                args.dropout, hypernet_info, args.ehnn_pe_dim, args.ehnn_hyper_dim, args.ehnn_hyper_layers,
                args.ehnn_hyper_dropout, ast.literal_eval(args.ehnn_force_broadcast), args.ehnn_input_dropout,
                ast.literal_eval(args.ehnn_mlp_classifier), args.Classifier_hidden, args.Classifier_num_layers, args.normalization
            )
        elif args.ehnn_type == 'linear_wo_order':
            self.model = EHNNLinear_wo_order(
                args.ehnn_hidden_channel, args.num_classes, args.ehnn_hidden_channel, args.ehnn_inner_channel, args.ehnn_n_layers,
                args.dropout, hypernet_info, args.ehnn_pe_dim, args.ehnn_hyper_dim, args.ehnn_hyper_layers,
                args.ehnn_hyper_dropout, ast.literal_eval(args.ehnn_force_broadcast), args.ehnn_input_dropout,
                ast.literal_eval(args.ehnn_mlp_classifier), args.Classifier_hidden, args.Classifier_num_layers, args.normalization
            )
        elif args.ehnn_type == 'linear_wo_global_order':
            self.model = EHNNLinear_wo_global_order(
                args.ehnn_hidden_channel, args.num_classes, args.ehnn_hidden_channel, args.ehnn_inner_channel, args.ehnn_n_layers,
                args.dropout, hypernet_info, args.ehnn_pe_dim, args.ehnn_hyper_dim, args.ehnn_hyper_layers,
                args.ehnn_hyper_dropout, ast.literal_eval(args.ehnn_force_broadcast), args.ehnn_input_dropout,
                ast.literal_eval(args.ehnn_mlp_classifier), args.Classifier_hidden, args.Classifier_num_layers, args.normalization
            )
        elif args.ehnn_type == 'transformer':
            self.model = EHNNTransformer(
                args.ehnn_hidden_channel, args.num_classes, args.ehnn_hidden_channel, args.ehnn_n_layers,
                args.ehnn_qk_channel, args.ehnn_hidden_channel, args.ehnn_n_heads, args.dropout, hypernet_info,
                args.ehnn_inner_channel, args.ehnn_pe_dim, args.ehnn_hyper_dim, args.ehnn_hyper_layers,
                args.ehnn_hyper_dropout, ast.literal_eval(args.ehnn_force_broadcast), args.ehnn_input_dropout,
                ast.literal_eval(args.ehnn_mlp_classifier), args.Classifier_hidden, args.Classifier_num_layers, args.normalization
            )
        elif args.ehnn_type == 'naive':
            self.model = EHNNNaive(
                args.ehnn_hidden_channel, args.num_classes, args.ehnn_hidden_channel, args.ehnn_inner_channel, args.ehnn_n_layers,
                args.dropout, hypernet_info, args.ehnn_pe_dim, args.ehnn_hyper_dim, args.ehnn_hyper_layers,
                args.ehnn_hyper_dropout, ast.literal_eval(args.ehnn_force_broadcast), args.ehnn_input_dropout,
                ast.literal_eval(args.ehnn_mlp_classifier), args.Classifier_hidden, args.Classifier_num_layers, args.normalization,
                args.ehnn_naive_use_hypernet
            )
        else:
            raise NotImplementedError
        self.reset_parameters()

    def reset_parameters(self):
        self.model.reset_parameters()

    def forward(self, data):
        """forward method
        :param data:
            data[idx]: {query_labels: [N,],
                        target_labels: [N,],
                        incidence: [N, |E|] sparse,
                        edge_orders: [|E|,],
                        prefix_normalizer: [|E|,],
                        suffix_normalizer: [N,]}
        :return: [N, C] dense
        """
        b = len(data)
        max_n = max([d['query_labels'].size(0) for d in data])
        max_e = max([d['edge_orders'].size(0) for d in data])

        x = torch.zeros((b, max_n), device=self.device, dtype=torch.long)  # [B, N]
        incidence = torch.zeros((b, max_n, max_e), device=self.device)  # [B, N, |E|]
        edge_orders = torch.zeros((b, max_e), device=self.device, dtype=torch.long)  # [B, |E|]
        prefix_normalizer = torch.zeros((b, max_e), device=self.device)  # [B, |E|]
        suffix_normalizer = torch.zeros((b, max_n), device=self.device)  # [B, N]
        node_mask = torch.zeros((b, max_n), device=self.device, dtype=torch.bool)  # [B, N]
        edge_mask = torch.zeros((b, max_e), device=self.device, dtype=torch.bool)  # [B, |E|]

        for idx, d in enumerate(data):
            n, e = d['query_labels'].size(0), d['edge_orders'].size(0)
            x[idx, :n] = d['query_labels'].to(self.device)
            edge_orders[idx, :n] = d['edge_orders'].to(self.device)
            prefix_normalizer[idx, :e] = d['prefix_normalizer'].to(self.device)
            suffix_normalizer[idx, :n] = d['suffix_normalizer'].to(self.device)
            incidence[idx, :n, :e] = d['incidence'].coalesce().to_dense().to(self.device)
            node_mask[idx, :n] = True
            edge_mask[idx, :e] = True

        ehnn_cache = {
            'incidence': incidence,
            'edge_orders': edge_orders,
            'prefix_normalizer': prefix_normalizer,
            'suffix_normalizer': suffix_normalizer,
            'node_mask': node_mask,
            'edge_mask': edge_mask,
            'n_nodes': node_mask.float().sum(1),
            'n_edges': edge_mask.float().sum(1)
        }
        x = self.input(x)  # [B, N, D]
        x = x.masked_fill(~node_mask[..., None], 0)
        x = self.model(x, ehnn_cache)  # [N, C]
        return x
