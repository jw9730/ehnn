import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from generate_hypergraph import generate_random_hypergraph, get_query_target

from models import SetGNN
from ehnn_classifier import EHNNClassifier


def parse_method(args, ehnn_cache=None):
    if args.method == 'EHNN':
        model = EHNNClassifier(args, ehnn_cache['max_edge_order'], ehnn_cache['max_overlap'])
    elif args.method == 'AllSetTransformer':
        args.LearnMask = False
        model = SetGNN(args)
    elif args.method == 'AllDeepSets':
        args.PMA = False
        args.aggregate = 'add'
        args.LearnMask = False
        model = SetGNN(args)
    else:
        raise NotImplementedError
    return model


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)


@torch.no_grad()
def evaluate(args, model, train_dataset, test_dataset, seen_orders, unseen_orders):
    model.eval()

    out = model(train_dataset)
    out = F.log_softmax(out, dim=1)
    gt = torch.cat([b['target_labels'] for b in train_dataset]).to(args.device)
    train_loss = F.nll_loss(out, gt)
    train_acc = eval_acc(gt, out)

    out = model(test_dataset)
    out = F.log_softmax(out, dim=1)
    gt = torch.cat([b['target_labels'] for b in test_dataset]).to(args.device)
    test_loss = F.nll_loss(out, gt)
    test_acc = eval_acc(gt, out)

    # for seen data
    test_dataset_seen = [d for d in test_dataset if d['query_order'] in seen_orders]
    test_seen_acc = -1
    if len(test_dataset_seen) > 0:
        out = model(test_dataset_seen)
        out = F.log_softmax(out, dim=1)
        gt = torch.cat([b['target_labels'] for b in test_dataset_seen]).to(args.device)
        test_seen_acc = eval_acc(gt, out)

    test_dataset_unseen = [d for d in test_dataset if d['query_order'] in unseen_orders]
    test_unseen_acc = -1
    if len(test_dataset_unseen) > 0:
        out = model(test_dataset_unseen)
        out = F.log_softmax(out, dim=1)
        gt = torch.cat([b['target_labels'] for b in test_dataset_unseen]).to(args.device)
        test_unseen_acc = eval_acc(gt, out)

    assert len(test_dataset_seen) + len(test_dataset_unseen) == len(test_dataset)

    return train_acc, test_acc, train_loss, test_loss, test_seen_acc, test_unseen_acc


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


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def main(args):
    n_train = args.n_train
    n_test = args.n_test
    train_orders = eval(args.train_orders)  # dangerous
    test_orders = eval(args.test_orders)  # dangerous
    seen_orders = set(test_orders.tolist()).intersection(set(train_orders.tolist()))
    unseen_orders = set(test_orders.tolist()).difference(set(train_orders.tolist()))
    print(f'test time: seen orders {seen_orders}, unseen orders {unseen_orders}')

    device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    args.device = device

    # Construct/Load synthetic data
    train_dataset = SyntheticDataset([get_data(args, train_orders) for _ in range(n_train)])
    test_dataset = SyntheticDataset([get_data(args, test_orders) for _ in range(n_test)])
    """
    train_path = './cache/train.pt'
    valid_path = './cache/valid.pt'
    test_path = './cache/test.pt'
    os.makedirs('./cache', exist_ok=True)
    if not os.path.isfile(train_path):
        train_dataset = SyntheticDataset([get_data(args, train_orders) for _ in range(n_train)])
        valid_dataset = SyntheticDataset([get_data(args, train_orders) for _ in range(n_valid)])
        test_dataset = SyntheticDataset([get_data(args, test_orders) for _ in range(n_test)])
        torch.save(train_dataset, train_path)
        torch.save(valid_dataset, valid_path)
        torch.save(test_dataset, test_path)
    else:
        train_dataset = torch.load(train_path)
        valid_dataset = torch.load(valid_path)
        test_dataset = torch.load(test_path)
    """
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=lambda x: x)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=8, collate_fn=lambda x: x)

    # Setup model
    if args.method == 'EHNN':
        model = parse_method(args, {'max_edge_order': max(train_orders.max(), test_orders.max()), 'max_overlap': 1})
    elif args.method == 'AllSetTransformer':
        model = SetGNN(args)
    elif args.method == 'AllDeepSets':
        args.PMA = False
        args.aggregate = 'add'
        model = SetGNN(args)
    else:
        raise NotImplementedError

    print(f'Model parameter number: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    model.to(device)

    # Start training
    criterion = nn.CrossEntropyLoss()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_epoch = 0
    best_test_acc = 0
    results = []
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            # batch[idx]: {query_labels: [N,],
            #              target_labels: [N,],
            #              incidence: [N, |E|] sparse,
            #              edge_orders: [|E|,],
            #              prefix_normalizer: [|E|,],
            #              suffix_normalizer: [N,],
            #              query_order: [1,]}

            optimizer.zero_grad()

            out = model(batch)
            out = F.log_softmax(out, dim=1)
            gt = torch.cat([b['target_labels'] for b in batch]).to(device)

            loss = criterion(out, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        result = evaluate(args, model, train_dataset, test_dataset, seen_orders, unseen_orders)
        results.append(result)
        if result[1] > best_test_acc:
            best_test_acc = result[1]
            best_epoch = epoch

        if (epoch % args.display_step == 0 or epoch == args.epochs - 1) and args.display_step > 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Train Loss: {loss:.4f}, '
                  f'Test  Loss: {result[3]:.4f}, '
                  f'Train Acc: {100 * result[0]:.2f}%, '
                  f'Test  Acc: {100 * result[1]:.2f}%, '
                  f'Test  Acc (seen): {100 * result[4]:.2f}%, '
                  f'Test  Acc (unseen): {100 * result[5]:.2f}%')

    print(f'Final result\n'
          f'Epoch: {best_epoch:02d}\n'
          f'Train Acc: {100 * results[best_epoch][0]:.2f}%\n'
          f'Test  Acc: {100 * results[best_epoch][1]:.2f}%\n'
          f'Test  Acc (seen): {100 * results[best_epoch][4]:.2f}%\n'
          f'Test  Acc (unseen): {100 * results[best_epoch][5]:.2f}%')


if __name__ == '__main__':
    # --- Main part of the training ---
    # Part 0: Parse arguments
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
    parser.add_argument('--heads', default=1, type=int)  # Placeholder
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
    parser.add_argument('--ehnn_naive_use_hypernet', action='store_true')

    args = parser.parse_args()
    main(args)
    print('All done! Exit python code')
    quit()
