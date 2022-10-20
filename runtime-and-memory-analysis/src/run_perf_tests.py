import argparse
import copy
import numpy as np
import time
import torch

from models import SetGNN
from ehnn_classifier import EHNNClassifier
from generate_hypergraph import generate_random_hypergraph, get_query_target


def parse_method(args, data, ehnn_cache=None):
    if args.method == 'AllSetTransformer':
        if args.LearnMask:
            model = SetGNN(args, data.norm)
        else:
            model = SetGNN(args)

    elif args.method == 'EHNN':
        model = EHNNClassifier(args, ehnn_cache)

    elif args.method == 'AllDeepSets':
        args.PMA = False
        args.aggregate = 'add'
        if args.LearnMask:
            model = SetGNN(args, data.norm)
        else:
            model = SetGNN(args)

    else:
        raise NotImplementedError
    return model


def numel(m: torch.nn.Module, only_trainable: bool = True):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def get_data(args, possible_edge_orders):
    hypergraph = generate_random_hypergraph(args.n_max_nodes, args.n_edges, possible_edge_orders)
    query_labels, target_labels, query_order = get_query_target(hypergraph)
    edge_orders = hypergraph['incidence'].sum(0).long()
    incidence = hypergraph['incidence'].to_sparse()
    prefix_normalizer = torch.sparse.sum(incidence, 0).to_dense()  # [|E|,]
    prefix_normalizer = prefix_normalizer.masked_fill_(prefix_normalizer == 0, 1e-5)
    suffix_normalizer = torch.sparse.sum(incidence, 1).to_dense()  # [|V|,]
    suffix_normalizer = suffix_normalizer.masked_fill_(suffix_normalizer == 0, 1e-5)
    return {'query_labels': query_labels,
            'incidence': incidence,
            'edge_orders': edge_orders,
            'prefix_normalizer': prefix_normalizer,
            'suffix_normalizer': suffix_normalizer,
            'target_labels': target_labels,
            'query_order': query_order}


def get_peak_mem_and_reset():
    stats = torch.cuda.memory_stats()
    peak_bytes_requirement = stats["allocated_bytes.all.peak"]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    return peak_bytes_requirement / 1024 ** 2  # unit: MB


def measure(model, data, ehnn_cache=None):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    if ehnn_cache is None:
        out = model(data)
    else:
        out = model(data, ehnn_cache)
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    forward_t = start.elapsed_time(end) # unit: milliseconds

    out = out.sum()

    start.record()
    out.backward()
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    backward_t = start.elapsed_time(end)  # unit: milliseconds

    peak_mem = get_peak_mem_and_reset()
    return forward_t, backward_t, peak_mem


def main_routine(args, repeat):
    # print(f'\n\nn = {n}')
    result = {}

    n_train = args.n_train
    train_orders = eval(args.train_orders)

    # train_dataset = SyntheticDataset([get_data(args, train_orders) for _ in range(n_train)])
    # train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=lambda x: x)
    data = get_data(args, train_orders)

    print("AllDeepSets")
    argsDS = copy.deepcopy(args)
    argsDS.method = 'AllDeepSets'
    try:
        forward_t, backward_t, peak_mem = [], [], []
        model = parse_method(argsDS, data)
        model.to(argsDS.device)
        print(f"number of parameters: {numel(model)}")
        for idx in range(repeat):
            ft, bt, pm = measure(model, data)
            forward_t.append(ft)
            backward_t.append(bt)
            peak_mem.append(pm)
        forward_t = forward_t[1:]
        backward_t = backward_t[1:]
        peak_mem = peak_mem[1:]
        result['DS_forward_t'] = (np.mean(np.array(forward_t)), np.std(np.array(forward_t)))
        result['DS_backward_t'] = (np.mean(np.array(backward_t)), np.std(np.array(backward_t)))
        result['DS_peak_mem'] = (np.mean(np.array(peak_mem)), np.std(np.array(peak_mem)))
    except RuntimeError as e:
        result['DS_forward_t'] = 'OOM'
        result['DS_backward_t'] = 'OOM'
        result['DS_peak_mem'] = 'OOM'
        print(e)

    print("AllSetTransformer")
    argsST = copy.deepcopy(args)
    argsST.method = 'AllSetTransformer'
    try:
        forward_t, backward_t, peak_mem = [], [], []
        model = parse_method(argsST, data)
        model.to(argsST.device)
        print(f"number of parameters: {numel(model)}")
        for idx in range(repeat):
            ft, bt, pm = measure(model, data)
            forward_t.append(ft)
            backward_t.append(bt)
            peak_mem.append(pm)
        forward_t = forward_t[1:]
        backward_t = backward_t[1:]
        peak_mem = peak_mem[1:]
        result['ST_forward_t'] = (np.mean(np.array(forward_t)), np.std(np.array(forward_t)))
        result['ST_backward_t'] = (np.mean(np.array(backward_t)), np.std(np.array(backward_t)))
        result['ST_peak_mem'] = (np.mean(np.array(peak_mem)), np.std(np.array(peak_mem)))
    except RuntimeError as e:
        result['ST_forward_t'] = 'OOM'
        result['ST_backward_t'] = 'OOM'
        result['ST_peak_mem'] = 'OOM'
        print(e)

    print("Naive Implementation (with lookup table)")
    argsEHNN_naive = copy.deepcopy(args)
    argsEHNN_naive.ehnn_type = 'naive'
    argsEHNN_naive.ehnn_naive_use_hypernet = False
    try:
        forward_t, backward_t, peak_mem = [], [], []
        ehnn_cache = {'max_edge_order': train_orders.max(), 'max_overlap': 1}
        model = parse_method(argsEHNN_naive, data, ehnn_cache)
        model.to(args.device)
        print(f"number of parameters: {numel(model)}")
        for idx in range(repeat):
            ft, bt, pm = measure(model, data, ehnn_cache)
            forward_t.append(ft)
            backward_t.append(bt)
            peak_mem.append(pm)
        forward_t = forward_t[1:]
        backward_t = backward_t[1:]
        peak_mem = peak_mem[1:]
        result['naive_lookup_forward_t'] = (np.mean(np.array(forward_t)), np.std(np.array(forward_t)))
        result['naive_lookup_backward_t'] = (np.mean(np.array(backward_t)), np.std(np.array(backward_t)))
        result['naive_lookup_peak_mem'] = (np.mean(np.array(peak_mem)), np.std(np.array(peak_mem)))
    except RuntimeError as e:
        result['naive_lookup_forward_t'] = 'OOM'
        result['naive_lookup_backward_t'] = 'OOM'
        result['naive_lookup_peak_mem'] = 'OOM'
        print(e)

    print("Naive Implementation (with hypernetwork)")
    argsEHNN_naive = copy.deepcopy(args)
    argsEHNN_naive.ehnn_type = 'naive'
    argsEHNN_naive.ehnn_naive_use_hypernet = True
    try:
        forward_t, backward_t, peak_mem = [], [], []
        ehnn_cache = {'max_edge_order': train_orders.max(), 'max_overlap': 1}
        model = parse_method(argsEHNN_naive, data, ehnn_cache)
        model.to(args.device)
        print(f"number of parameters: {numel(model)}")
        for idx in range(repeat):
            ft, bt, pm = measure(model, data, ehnn_cache)
            forward_t.append(ft)
            backward_t.append(bt)
            peak_mem.append(pm)
        forward_t = forward_t[1:]
        backward_t = backward_t[1:]
        peak_mem = peak_mem[1:]
        result['naive_hypernet_forward_t'] = (np.mean(np.array(forward_t)), np.std(np.array(forward_t)))
        result['naive_hypernet_backward_t'] = (np.mean(np.array(backward_t)), np.std(np.array(backward_t)))
        result['naive_hypernet_peak_mem'] = (np.mean(np.array(peak_mem)), np.std(np.array(peak_mem)))
    except RuntimeError as e:
        result['naive_hypernet_forward_t'] = 'OOM'
        result['naive_hypernet_backward_t'] = 'OOM'
        result['naive_hypernet_peak_mem'] = 'OOM'
        print(e)

    print("EHNN-MLP")
    argsEHNN_MLP = copy.deepcopy(args)
    try:
        forward_t, backward_t, peak_mem = [], [], []
        ehnn_cache = {'max_edge_order': train_orders.max(), 'max_overlap': 1}
        model = parse_method(argsEHNN_MLP, data, ehnn_cache)
        model.to(args.device)
        print(f"number of parameters: {numel(model)}")
        for idx in range(repeat):
            ft, bt, pm = measure(model, data, ehnn_cache)
            forward_t.append(ft)
            backward_t.append(bt)
            peak_mem.append(pm)
        forward_t = forward_t[1:]
        backward_t = backward_t[1:]
        peak_mem = peak_mem[1:]
        result['ehnn_mlp_forward_t'] = (np.mean(np.array(forward_t)), np.std(np.array(forward_t)))
        result['ehnn_mlp_backward_t'] = (np.mean(np.array(backward_t)), np.std(np.array(backward_t)))
        result['ehnn_mlp_peak_mem'] = (np.mean(np.array(peak_mem)), np.std(np.array(peak_mem)))
    except RuntimeError as e:
        result['ehnn_mlp_forward_t'] = 'OOM'
        result['ehnn_mlp_backward_t'] = 'OOM'
        result['ehnn_mlp_peak_mem'] = 'OOM'
        print(e)

    print("EHNN-Transformer")
    argsEHNN_Transformer = copy.deepcopy(args)
    argsEHNN_Transformer.ehnn_type = 'transformer'
    try:
        forward_t, backward_t, peak_mem = [], [], []
        ehnn_cache = {'max_edge_order': train_orders.max(), 'max_overlap': 1}
        model = parse_method(argsEHNN_Transformer, data, ehnn_cache)
        model.to(args.device)
        print(f"number of parameters: {numel(model)}")
        for idx in range(repeat):
            ft, bt, pm = measure(model, data, ehnn_cache)
            forward_t.append(ft)
            backward_t.append(bt)
            peak_mem.append(pm)
        forward_t = forward_t[1:]
        backward_t = backward_t[1:]
        peak_mem = peak_mem[1:]
        result['ehnn_transformer_forward_t'] = (np.mean(np.array(forward_t)), np.std(np.array(forward_t)))
        result['ehnn_transformer_backward_t'] = (np.mean(np.array(backward_t)), np.std(np.array(backward_t)))
        result['ehnn_transformer_peak_mem'] = (np.mean(np.array(peak_mem)), np.std(np.array(peak_mem)))
    except RuntimeError as e:
        result['ehnn_transformer_forward_t'] = 'OOM'
        result['ehnn_transformer_backward_t'] = 'OOM'
        result['ehnn_transformer_peak_mem'] = 'OOM'
        print(e)

    print("EHNN-Transformer Optimized")
    argsEHNN_Transformer = copy.deepcopy(args)
    argsEHNN_Transformer.ehnn_type = 'transformer_optimized'
    try:
        forward_t, backward_t, peak_mem = [], [], []
        ehnn_cache = {'max_edge_order': train_orders.max(), 'max_overlap': 1}
        model = parse_method(argsEHNN_Transformer, data, ehnn_cache)
        model.to(args.device)
        print(f"number of parameters: {numel(model)}")
        for idx in range(repeat):
            ft, bt, pm = measure(model, data, ehnn_cache)
            forward_t.append(ft)
            backward_t.append(bt)
            peak_mem.append(pm)
        forward_t = forward_t[1:]
        backward_t = backward_t[1:]
        peak_mem = peak_mem[1:]
        result['ehnn_transformer_optimized_forward_t'] = (np.mean(np.array(forward_t)), np.std(np.array(forward_t)))
        result['ehnn_transformer_optimized_backward_t'] = (np.mean(np.array(backward_t)), np.std(np.array(backward_t)))
        result['ehnn_transformer_optimized_peak_mem'] = (np.mean(np.array(peak_mem)), np.std(np.array(peak_mem)))
    except RuntimeError as e:
        result['ehnn_transformer_optimized_forward_t'] = 'OOM'
        result['ehnn_transformer_optimized_backward_t'] = 'OOM'
        result['ehnn_transformer_optimized_peak_mem'] = 'OOM'
        print(e)

    return result


def main(args):
    device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    args.device = device

    repeat = 21
    # n_list = list((2 ** np.linspace(5, 18, 27, endpoint=True)).astype(int) // 5)  # for log-scale plot
    start = time.time()
    result = main_routine(args, repeat)
    print(f"done after {(time.time() - start):.2f} sec")
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset options
    parser.add_argument('--n_max_nodes', type=int, default=50, help='number of initialized nodes')
    parser.add_argument('--n_edges', type=int, default=10, help='number of edges to be generated')
    parser.add_argument('--train_orders', type=str, default='torch.arange(2, 5)', help='train hyperedge order range')
    parser.add_argument('--test_orders', type=str, default='torch.arange(2, 5)', help='test hyperedge order range')
    parser.add_argument('--n_train', type=int, default=50, help='number of train graphs')
    parser.add_argument('--n_test', type=int, default=25, help='number of test graphs')

    # common options
    parser.add_argument('--cuda', default=0, choices=[-1, 0, 1], type=int)
    parser.add_argument('--num_features', type=int, default=2, help='input feature dimension')
    parser.add_argument('--num_classes', type=int, default=2, help='target classes')
    parser.add_argument('--dropout', type=float, default=0, help='dropout after first layer')
    parser.add_argument('--Classifier_num_layers', default=1, type=int)  # How many layers of decoder
    parser.add_argument('--Classifier_hidden', default=64, type=int)  # Decoder hidden units
    parser.add_argument('--normalization', default='ln')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--wd', default=0.0, type=float)
    parser.add_argument('--epochs', default=50, type=int)

    parser.add_argument('--All_num_layers', default=2, type=int)
    parser.add_argument('--MLP_num_layers', default=2, type=int)  # How many layers of encoder
    parser.add_argument('--MLP_hidden', default=64, type=int)  # Encoder hidden units

    parser.add_argument('--display_step', type=int, default=10)
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    # ['all_one','deg_half_sym']
    parser.add_argument('--normtype', default='all_one')
    parser.add_argument('--add_self_loop', action='store_false')
    # NormLayer for MLP. ['bn','ln','None']
    parser.add_argument('--deepset_input_norm', default=True)
    parser.add_argument('--GPR', action='store_false')  # skip all but last dec

    # whether the he contain self node or not
    parser.add_argument('--exclude_self', action='store_true')
    parser.add_argument('--PMA', action='store_true')
    parser.add_argument('--LearnMask', action='store_true')

    # model options
    parser.add_argument('--method', type=str, default='EHNN', help='method')
    parser.add_argument('--heads', default=4, type=int)  # Placeholder
    parser.add_argument('--output_heads', default=1, type=int)  # Placeholder

    # Args for EHNN
    parser.add_argument('--ehnn_n_layers', type=int, default=2, help='layer num')
    parser.add_argument('--ehnn_ff_layers', type=int, default=2, help='encoder ff layer num')
    parser.add_argument('--ehnn_mlp_channel', type=int, default=64, help='mlp channel')
    parser.add_argument('--ehnn_qk_channel', type=int, default=64, help='qk channel')
    parser.add_argument('--ehnn_n_heads', type=int, default=4, help='n_heads')
    parser.add_argument('--ehnn_inner_channel', type=int, default=64, help='inner channel')
    parser.add_argument('--ehnn_hidden_channel', type=int, default=64, help='hidden dim')
    parser.add_argument('--ehnn_type', type=str, default='transformer', help='ehnn type')
    parser.add_argument('--ehnn_pe_dim', type=int, default=64, help='pe dim')
    parser.add_argument('--ehnn_hyper_dim', type=int, default=64, help='hypernetwork dim')
    parser.add_argument('--ehnn_hyper_layers', type=int, default=3, help='hypernetwork layers')
    parser.add_argument('--ehnn_hyper_dropout', type=float, default=0, help='hypernetwork dropout rate')
    parser.add_argument('--ehnn_force_broadcast', type=str, default='False', help='force broadcast based pe')
    parser.add_argument('--ehnn_input_dropout', type=float, default=0., help='input dropout rate')
    parser.add_argument('--ehnn_mlp_classifier', type=str, default='True', help='mlp classifier head')
    parser.add_argument('--ehnn_slack_channel', type=str, default="None", help='manual slack channel assignment')
    parser.add_argument('--ehnn_att0_dropout', type=float, default=0., help='att0 dropout rate')
    parser.add_argument('--ehnn_att1_dropout', type=float, default=0., help='att1 dropout rate')
    parser.add_argument('--ehnn_naive_use_hypernet', action='store_true')

    parser.set_defaults(PMA=True)  # True: Use PMA. False: Use Deepsets.
    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(exclude_self=False)
    parser.set_defaults(GPR=False)
    parser.set_defaults(LearnMask=False)
    parser.set_defaults(HyperGCN_mediators=True)
    parser.set_defaults(HyperGCN_fast=True)
    parser.set_defaults(HCHA_symdegnorm=False)

    args = parser.parse_args()
    main(args)
    print('All done! Exit python code')
    quit()
