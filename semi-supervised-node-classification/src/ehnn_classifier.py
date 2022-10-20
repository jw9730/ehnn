import ast

import torch.nn as nn

from ehnn import EHNNLinear, EHNNTransformer


class EHNNClassifier(nn.Module):
    def __init__(self, args, ehnn_cache):
        super().__init__()
        edge_orders = ehnn_cache["edge_orders"]  # [|E|,]
        overlaps = ehnn_cache["overlaps"]  # [|overlaps|,]
        max_edge_order = int(edge_orders.max().item())
        max_overlap = int(overlaps.max().item()) if overlaps is not None else 0
        hypernet_info = (max_edge_order, max_edge_order, max_overlap)

        if args.ehnn_type == "linear":
            self.model = EHNNLinear(
                args.num_features,
                args.num_classes,
                args.ehnn_hidden_channel,
                args.ehnn_inner_channel,
                args.ehnn_n_layers,
                args.dropout,
                hypernet_info,
                args.ehnn_pe_dim,
                args.ehnn_hyper_dim,
                args.ehnn_hyper_layers,
                args.ehnn_hyper_dropout,
                ast.literal_eval(args.ehnn_force_broadcast),
                args.ehnn_input_dropout,
                ast.literal_eval(args.ehnn_mlp_classifier),
                args.Classifier_hidden,
                args.Classifier_num_layers,
                args.normalization,
            )
        elif args.ehnn_type == "transformer":
            self.model = EHNNTransformer(
                args.num_features,
                args.num_classes,
                args.ehnn_hidden_channel,
                args.ehnn_n_layers,
                args.ehnn_qk_channel,
                args.ehnn_hidden_channel,
                args.ehnn_n_heads,
                args.dropout,
                hypernet_info,
                args.ehnn_inner_channel,
                args.ehnn_pe_dim,
                args.ehnn_hyper_dim,
                args.ehnn_hyper_layers,
                args.ehnn_hyper_dropout,
                ast.literal_eval(args.ehnn_force_broadcast),
                args.ehnn_input_dropout,
                ast.literal_eval(args.ehnn_mlp_classifier),
                args.Classifier_hidden,
                args.Classifier_num_layers,
                args.normalization,
                args.ehnn_att0_dropout,
                args.ehnn_att1_dropout,
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
        x = data.x
        x = self.model(x, ehnn_cache)
        return x
