import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear import LinearV2E, LinearE2V
from .attn import TransformerV2E, TransformerE2V
from .attn_optimized import TransformerV2EOpt, TransformerE2VOpt
from .naive import NaiveV2E, NaiveE2V


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='bn', InputNorm=False):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif Normalization == 'ln':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ is 'Identity'):
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i + 1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class EHNNLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, inner_dim, n_layers, dropout, hypernet_info,
                 pe_dim, hyper_dim, hyper_layers, hyper_dropout, force_broadcast, input_dropout,
                 mlp_classifier, Classifier_hidden, Classifier_num_layers, NormLayer):
        super().__init__()
        assert n_layers == 2
        assert mlp_classifier
        self.n_layers = n_layers
        self.input_dropout = nn.Dropout(input_dropout)
        self.input = nn.Linear(dim_in, dim_hidden)
        self.layer1 = LinearV2E(dim_hidden, dim_hidden, hypernet_info, inner_dim, pe_dim, hyper_dim, hyper_layers,
                                hyper_dropout, force_broadcast)
        self.layer2 = LinearE2V(dim_hidden, dim_hidden, hypernet_info, inner_dim, pe_dim, hyper_dim, hyper_layers,
                                hyper_dropout, force_broadcast)
        self.f = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.output = MLP(in_channels=dim_hidden,
                          hidden_channels=Classifier_hidden,
                          out_channels=dim_out,
                          num_layers=Classifier_num_layers,
                          dropout=dropout,
                          Normalization=NormLayer,
                          InputNorm=False)
        print(f'hypernet max input range: {hypernet_info}')

    def reset_parameters(self):
        self.input.reset_parameters()
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.output.reset_parameters()

    def forward(self, x, ehnn_cache) -> torch.Tensor:

        x = self.input(self.input_dropout(x))
        x_v, x_e = self.layer1(x, ehnn_cache)
        x_v, x_e = self.f(x_v), self.f(x_e)
        x_v, x_e = self.dropout(x_v), self.dropout(x_e)
        x = (x_v, x_e)
        x = self.layer2(x, ehnn_cache)
        x = self.output(x)
        return x


class EHNNTransformer(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, n_layers, dim_qk, dim_ff, n_heads, dropout, hypernet_info,
                 inner_dim, pe_dim, hyper_dim, hyper_layers, hyper_dropout, force_broadcast, input_dropout,
                 mlp_classifier, Classifier_hidden, Classifier_num_layers, NormLayer, att0_dropout, att1_dropout):
        super().__init__()
        assert n_layers == 2
        assert mlp_classifier
        self.input_dropout = nn.Dropout(input_dropout)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim_hidden)
        self.input = nn.Linear(dim_in, dim_hidden)
        self.layer1 = TransformerV2E(dim_hidden, dim_qk, dim_ff, n_heads, dropout, hypernet_info,
                                     inner_dim, pe_dim, hyper_dim, hyper_layers, hyper_dropout, force_broadcast, att0_dropout, att1_dropout)
        self.layer2 = TransformerE2V(dim_hidden, dim_qk, dim_ff, n_heads, dropout, hypernet_info,
                                     inner_dim, pe_dim, hyper_dim, hyper_layers, hyper_dropout, force_broadcast, att0_dropout, att1_dropout)
        self.output = MLP(in_channels=dim_hidden,
                          hidden_channels=Classifier_hidden,
                          out_channels=dim_out,
                          num_layers=Classifier_num_layers,
                          dropout=dropout,
                          Normalization=NormLayer,
                          InputNorm=False)
        print(f'hypernet max input range: {hypernet_info}')

    def reset_parameters(self):
        self.input.reset_parameters()
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.output.reset_parameters()

    def forward(self, x, ehnn_cache) -> torch.Tensor:
        with torch.no_grad():
            ehnn_cache['incidence'] = ehnn_cache['incidence'].coalesce()
            n, e = ehnn_cache['incidence'].size()
            node_indices = torch.arange(0, n, device=ehnn_cache['incidence'].device)[None, :].repeat(2, 1)
            node_indices[1] += e
            ehnn_cache['indices_with_nodes'] = torch.cat((ehnn_cache['incidence'].indices(), node_indices), dim=1)
        x = self.input_dropout(x)
        x = self.input(x)

        x_v, x_e = self.layer1(x, ehnn_cache)
        x_v, x_e = self.dropout(x_v), self.dropout(x_e)
        x = (x_v, x_e)
        x = self.layer2(x, ehnn_cache)
        x = self.dropout(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


class EHNNTransformerOpt(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, n_layers, dim_qk, dim_ff, n_heads, dropout, hypernet_info,
                 inner_dim, pe_dim, hyper_dim, hyper_layers, hyper_dropout, force_broadcast, input_dropout,
                 mlp_classifier, Classifier_hidden, Classifier_num_layers, NormLayer, att0_dropout, att1_dropout):
        super().__init__()
        assert n_layers == 2
        assert mlp_classifier
        self.input_dropout = nn.Dropout(input_dropout)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim_hidden)
        self.input = nn.Linear(dim_in, dim_hidden)
        self.layer1 = TransformerV2EOpt(dim_hidden, dim_qk, dim_ff, n_heads, dropout, hypernet_info,
                                        inner_dim, pe_dim, hyper_dim, hyper_layers, hyper_dropout, force_broadcast, att0_dropout, att1_dropout)
        self.layer2 = TransformerE2VOpt(dim_hidden, dim_qk, dim_ff, n_heads, dropout, hypernet_info,
                                        inner_dim, pe_dim, hyper_dim, hyper_layers, hyper_dropout, force_broadcast, att0_dropout, att1_dropout)
        self.output = MLP(in_channels=dim_hidden,
                          hidden_channels=Classifier_hidden,
                          out_channels=dim_out,
                          num_layers=Classifier_num_layers,
                          dropout=dropout,
                          Normalization=NormLayer,
                          InputNorm=False)
        print(f'hypernet max input range: {hypernet_info}')

    def reset_parameters(self):
        self.input.reset_parameters()
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.output.reset_parameters()

    def forward(self, x, ehnn_cache) -> torch.Tensor:
        with torch.no_grad():
            ehnn_cache['incidence'] = ehnn_cache['incidence'].coalesce()
            n, e = ehnn_cache['incidence'].size()
            node_indices = torch.arange(0, n, device=ehnn_cache['incidence'].device)[None, :].repeat(2, 1)
            node_indices[1] += e
            ehnn_cache['indices_with_nodes'] = torch.cat((ehnn_cache['incidence'].indices(), node_indices), dim=1)
        x = self.input_dropout(x)
        x = self.input(x)

        x_v, x_e = self.layer1(x, ehnn_cache)
        x_v, x_e = self.dropout(x_v), self.dropout(x_e)
        x = (x_v, x_e)
        x = self.layer2(x, ehnn_cache)
        x = self.dropout(x)

        x = self.output(x)
        return x


class EHNNNaive(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, inner_dim, n_layers, dropout, hypernet_info,
                 pe_dim, hyper_dim, hyper_layers, hyper_dropout, force_broadcast, input_dropout,
                 mlp_classifier, Classifier_hidden, Classifier_num_layers, NormLayer, use_hypernet):
        super().__init__()
        assert n_layers == 2
        assert mlp_classifier
        self.n_layers = n_layers
        self.input_dropout = nn.Dropout(input_dropout)
        self.input = nn.Linear(dim_in, dim_hidden)
        self.layer1 = NaiveV2E(dim_hidden, dim_hidden, hypernet_info, inner_dim, pe_dim, hyper_dim, hyper_layers,
                               hyper_dropout, force_broadcast, use_hypernet)
        self.layer2 = NaiveE2V(dim_hidden, dim_hidden, hypernet_info, inner_dim, pe_dim, hyper_dim, hyper_layers,
                               hyper_dropout, force_broadcast, use_hypernet)
        self.f = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.output = MLP(in_channels=dim_hidden,
                          hidden_channels=Classifier_hidden,
                          out_channels=dim_out,
                          num_layers=Classifier_num_layers,
                          dropout=dropout,
                          Normalization=NormLayer,
                          InputNorm=False)
        print(f'hypernet max input range: {hypernet_info}')

    def reset_parameters(self):
        self.input.reset_parameters()
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.output.reset_parameters()

    def forward(self, x, ehnn_cache) -> torch.Tensor:
        """forward method
        :param x: [B, N, D]
        :param ehnn_cache: {incidence: [B, N, |E|],
                            edge_orders: [B, |E|],
                            prefix_normalizer: [B, |E|],
                            suffix_normalizer: [B, N],
                            node_mask: [B, N],
                            edge_mask: [B, |E|],
                            incidence_mask: [B, N, |E|]}
        :return: [N, C] dense
        """
        x = self.input(self.input_dropout(x))
        x_v, x_e = self.layer1(x, ehnn_cache)
        x_v, x_e = self.f(x_v), self.f(x_e)
        x_v, x_e = self.dropout(x_v), self.dropout(x_e)
        x = (x_v, x_e)
        x = self.layer2(x, ehnn_cache)
        x = self.output(x)
        x = torch.cat([x_[m_] for x_, m_ in zip(x, ehnn_cache['node_mask'])])  # [N, C]
        return x
