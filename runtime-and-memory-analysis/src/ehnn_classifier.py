import ast
import torch.nn as nn
from ehnn import EHNNLinear, EHNNTransformer, EHNNNaive, EHNNTransformerOpt
import torch


class EHNNClassifier(nn.Module):
    def __init__(self, args, ehnn_cache):
        super().__init__()
        # edge_orders = ehnn_cache['edge_orders']  # [|E|,]
        # overlaps = ehnn_cache['overlaps']  # [|overlaps|,]
        # max_edge_order = int(edge_orders.max().item())
        # max_overlap = int(overlaps.max().item()) if overlaps is not None else 0
        max_edge_order = ehnn_cache['max_edge_order']
        max_overlap = ehnn_cache['max_overlap']
        hypernet_info = (max_edge_order, max_edge_order, max_overlap)
        self.input = nn.Embedding(args.num_features, args.ehnn_hidden_channel)
        self.device = args.device

        if args.ehnn_type == 'linear':
            self.model = EHNNLinear(
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
                ast.literal_eval(args.ehnn_mlp_classifier), args.Classifier_hidden, args.Classifier_num_layers, args.normalization,
                args.ehnn_att0_dropout, args.ehnn_att1_dropout
            )
        elif args.ehnn_type == 'transformer_optimized':
            self.model = EHNNTransformerOpt(
                args.ehnn_hidden_channel, args.num_classes, args.ehnn_hidden_channel, args.ehnn_n_layers,
                args.ehnn_qk_channel, args.ehnn_hidden_channel, args.ehnn_n_heads, args.dropout, hypernet_info,
                args.ehnn_inner_channel, args.ehnn_pe_dim, args.ehnn_hyper_dim, args.ehnn_hyper_layers,
                args.ehnn_hyper_dropout, ast.literal_eval(args.ehnn_force_broadcast), args.ehnn_input_dropout,
                ast.literal_eval(args.ehnn_mlp_classifier), args.Classifier_hidden, args.Classifier_num_layers, args.normalization,
                args.ehnn_att0_dropout, args.ehnn_att1_dropout
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

    def reset_parameters(self):
        self.model.reset_parameters()

    def forward(self, data, ehnn_cache):
        """forward method
        :param data:
        :param ehnn_cache:
        :return: [N, C] dense
        """

        max_n = data['query_labels'].size(0)
        max_e = data['edge_orders'].size(0)

        x = torch.zeros(max_n, device=self.device, dtype=torch.long)  # [B, N]
        n, e = data['query_labels'].size(0), data['edge_orders'].size(0)
        x = data['query_labels'].to(self.device)

        data['incidence'] = data['incidence'].to(self.device)
        data['edge_orders'] = data['edge_orders'].to(self.device)
        data['prefix_normalizer'] = data['prefix_normalizer'].to(self.device)
        data['suffix_normalizer'] = data['suffix_normalizer'].to(self.device)
        data['target_labels'] = data['target_labels'].to(self.device)
        data['node_mask'] = torch.ones(n, device=self.device, dtype=torch.bool)
        data['edge_mask'] = torch.ones(e, device=self.device, dtype=torch.bool)

        data['n_nodes'] = n
        data['n_edges'] = e


        x = self.input(x)
        x = self.model(x, data)
        return x
