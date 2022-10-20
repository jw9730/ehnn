#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from convert_datasets_to_pygDataset import dataset_Hypergraph
from ehnn import build_mask, build_mask_chunk
from ehnn_classifier import EHNNClassifier
from preprocessing import ConstructH, ConstructHSparse, ExtractV2E, rand_train_test_idx


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n))


def parse_method(args, ehnn_cache=None):
    if args.method == "EHNN":
        model = EHNNClassifier(args, ehnn_cache)
    else:
        # Below we can add different model, such as HyperGCN and so on
        raise NotImplementedError
    return model


class Logger(object):
    """Adapted from https://github.com/snap-stanford/ogb/"""

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert 0 <= run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f"Run {run + 1:02d}:")
            print(f"Highest Train: {result[:, 0].max():.2f}")
            print(f"Highest Valid: {result[:, 1].max():.2f}")
            print(f"  Final Train: {result[argmax, 0]:.2f}")
            print(f"   Final Test: {result[argmax, 2]:.2f}")
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f"All runs:")
            msg = ""
            r = best_result[:, 0]
            print(f"Highest Train: {r.mean():.2f} ± {r.std():.2f}")
            msg += f"Highest Train: {r.mean():.2f} ± {r.std():.2f}\n"
            r = best_result[:, 1]
            print(f"Highest Valid: {r.mean():.2f} ± {r.std():.2f}")
            msg += f"Highest Valid: {r.mean():.2f} ± {r.std():.2f}\n"
            r = best_result[:, 2]
            print(f"  Final Train: {r.mean():.2f} ± {r.std():.2f}")
            msg += f"  Final Train: {r.mean():.2f} ± {r.std():.2f}\n"
            r = best_result[:, 3]
            print(f"   Final Test: {r.mean():.2f} ± {r.std():.2f}")
            msg += f"   Final Test: {r.mean():.2f} ± {r.std():.2f}\n"

            return best_result[:, 1], best_result[:, 3], msg


@torch.no_grad()
def evaluate(args, model, data, split_idx, eval_func, result=None, ehnn_cache=None):
    if result is not None:
        out = result
    else:
        model.eval()
        if args.method in ["EHNN"]:
            out = model(data, ehnn_cache)
        else:
            out = model(data)
        out = F.log_softmax(out, dim=1)

    train_acc = eval_func(data.y[split_idx["train"]], out[split_idx["train"]])
    valid_acc = eval_func(data.y[split_idx["valid"]], out[split_idx["valid"]])
    test_acc = eval_func(data.y[split_idx["test"]], out[split_idx["test"]])

    # Also keep track of losses
    train_loss = F.nll_loss(out[split_idx["train"]], data.y[split_idx["train"]])
    valid_loss = F.nll_loss(out[split_idx["valid"]], data.y[split_idx["valid"]])
    test_loss = F.nll_loss(out[split_idx["test"]], data.y[split_idx["test"]])
    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()
    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # --- Main part of the training ---
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_prop", type=float, default=0.5)
    parser.add_argument("--valid_prop", type=float, default=0.25)
    parser.add_argument("--dname", default="walmart-trips-100")
    parser.add_argument("--method", default="EHNN")
    parser.add_argument("--epochs", default=500, type=int)
    # Number of runs for each split (test fix, only shuffle train/val)
    parser.add_argument("--runs", default=20, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--wd", default=0.0, type=float)

    parser.add_argument("--Classifier_num_layers", default=2, type=int)  # How many layers of decoder
    parser.add_argument("--Classifier_hidden", default=64, type=int)  # Decoder hidden units
    parser.add_argument("--display_step", type=int, default=-1)
    # NormLayer for MLP. ['bn','ln','None']
    parser.add_argument("--normalization", default="ln")
    parser.add_argument("--num_features", default=0, type=int)  # Placeholder
    parser.add_argument("--num_classes", default=0, type=int)  # Placeholder
    # Choose std for synthetic feature noise
    parser.add_argument("--feature_noise", default="1", type=str)

    # Args for EHNN
    parser.add_argument("--ehnn_n_layers", type=int, default=2, help="layer num")
    parser.add_argument("--ehnn_ff_layers", type=int, default=2, help="encoder ff layer num")
    parser.add_argument("--ehnn_qk_channel", type=int, default=64, help="qk channel")
    parser.add_argument("--ehnn_n_heads", type=int, default=4, help="n_heads")
    parser.add_argument("--ehnn_inner_channel", type=int, default=64, help="inner channel")
    parser.add_argument("--ehnn_hidden_channel", type=int, default=64, help="hidden dim")
    parser.add_argument("--ehnn_type", type=str, default="encoder", help="ehnn type")
    parser.add_argument("--ehnn_pe_dim", type=int, default=64, help="pe dim")
    parser.add_argument("--ehnn_hyper_dim", type=int, default=64, help="hypernetwork dim")
    parser.add_argument("--ehnn_hyper_layers", type=int, default=2, help="hypernetwork layers")
    parser.add_argument("--ehnn_hyper_dropout", type=float, default=0.2, help="hypernetwork dropout rate",)
    parser.add_argument("--ehnn_force_broadcast", type=str, help="force broadcast based pe")
    parser.add_argument("--ehnn_input_dropout", type=float, default=0.0, help="input dropout rate")
    parser.add_argument("--ehnn_mlp_classifier", type=str, help="mlp classifier head")
    parser.add_argument("--ehnn_att0_dropout", type=float, default=0.0, help="att0 dropout rate")
    parser.add_argument("--ehnn_att1_dropout", type=float, default=0.0, help="att1 dropout rate")

    args = parser.parse_args()

    # Load and preprocess data
    existing_dataset = [
        "zoo",
        "20newsW100",
        "Mushroom",
        "NTU2012",
        "ModelNet40",
        "yelp",
        "house-committees-100",
        "walmart-trips-100",
    ]

    synthetic_list = ["walmart-trips-100", "house-committees-100"]

    if args.dname in existing_dataset:
        dname = args.dname
        f_noise = args.feature_noise
        if (f_noise is not None) and dname in synthetic_list:
            p2raw = "../data/raw_data/"
            dataset = dataset_Hypergraph(name=dname, feature_noise=f_noise, p2raw=p2raw)
        else:
            if dname in ["yelp"]:
                p2raw = "../data/raw_data/yelp/"
            else:
                p2raw = "../data/raw_data/"
            dataset = dataset_Hypergraph(
                name=dname,
                root="../data/pyg_data/hypergraph_dataset_updated/",
                p2raw=p2raw,
            )
        data = dataset.data
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        if args.dname in ["yelp", "walmart-trips-100", "house-committees-100"]:
            # Shift the y label to start with 0
            args.num_classes = len(data.y.unique())
            data.y = data.y - data.y.min()
        if not hasattr(data, "n_x"):
            data.n_x = torch.tensor([data.x.shape[0]])
        if not hasattr(data, "num_hyperedges"):
            # note that we assume the he_id is consecutive.
            data.num_hyperedges = torch.tensor(
                [data.edge_index[0].max() - data.n_x[0] + 1]
            )
    else:
        raise NotImplementedError

    print(args.method)
    print(f"feature noise: {args.feature_noise}")
    print(f"force_broadcast: {args.ehnn_force_broadcast}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ehnn_cache = None

    if args.method in ["EHNN"]:
        data = ExtractV2E(data)  # [2, |E|] (caution: [V; E])
        x = data.x  # [N, D]
        y = data.y  # [N,]
        num_nodes = data.n_x[0] if isinstance(data.n_x, list) else data.n_x
        num_hyperedges = (data.num_hyperedges[0] if isinstance(data.num_hyperedges, list) else data.num_hyperedges)

        assert (num_nodes + num_hyperedges - 1) == data.edge_index[1].max().item(), "num_hyperedges does not match!"
        assert num_nodes == data.x.size(0), f"num_nodes does not match!"

        # build memory-efficiently for yelp and walmart-trips-100
        if args.dname not in ["yelp", "walmart-trips-100"]:
            original_edge_index = data.edge_index
            data = ConstructH(data)  # [|V|, |E|]
            incidence_d = (
                torch.tensor(data.edge_index, dtype=torch.float32)
                .to_sparse(2)
                .coalesce()
                .to(device)
            )
            edge_orders_d = (
                torch.sparse.sum(incidence_d, 0).to_dense().long().to(device)
            )  # [|E|,]

            data.edge_index = original_edge_index
            data = ConstructHSparse(data)
            num_nodes = data.x.shape[0]
            num_hyperedges = np.max(data.edge_index[1]) + 1
            incidence_s = torch.sparse_coo_tensor(
                data.edge_index,
                torch.ones(len(data.edge_index[0])),
                (num_nodes, num_hyperedges),
                device=device,
            ).coalesce()
            edge_orders_s = (
                torch.sparse.sum(incidence_s, 0).to_dense().long().to(device)
            )  # [|E|,]

            assert (incidence_d.indices() - incidence_s.indices() == 0).all()
            assert (incidence_d.values() - incidence_s.values() == 0).all()

            incidence = incidence_d
            edge_orders = edge_orders_d

        else:
            data = ConstructHSparse(data)
            num_nodes = data.x.shape[0]
            num_hyperedges = np.max(data.edge_index[1]) + 1
            incidence = (
                torch.sparse_coo_tensor(
                    data.edge_index,
                    torch.ones(len(data.edge_index[0])),
                    (num_nodes, num_hyperedges),
                )
                .coalesce()
                .to(device)
            )
            edge_orders = torch.sparse.sum(incidence, 0).to_dense().long().to(device)

        os.makedirs("./cache", exist_ok=True)
        if not osp.isfile(f"./cache/{args.dname}.pt"):
            print(f"preprocessing {args.dname}")
            overlaps = None
            masks = None
            n_overlaps = None
            prefix_normalizer = (torch.sparse.sum(incidence, 0).to_dense().to(device))  # [|E|,]
            prefix_normalizer = prefix_normalizer.masked_fill_(prefix_normalizer == 0, 1e-5)
            normalizer = None
            suffix_normalizer = (torch.sparse.sum(incidence, 1).to_dense().to(device))  # [|V|,]
            suffix_normalizer = suffix_normalizer.masked_fill_(suffix_normalizer == 0, 1e-5)

            if args.dname not in ["yelp", "walmart-trips-100"]:
                # chunked mask computation
                mask_dict_chunk = build_mask_chunk(incidence, device)  # Dict(overlap: [|E|, |E|] sparse)
                overlaps_chunk, masks_chunk = list(mask_dict_chunk.keys()), list(mask_dict_chunk.values())
                overlaps_chunk = torch.tensor(overlaps_chunk, dtype=torch.long, device=device)  # [|overlaps|,]
                masks_chunk = torch.stack(masks_chunk, dim=0).coalesce()  # [|overlaps|, |E|, |E|]]

                # correctness check with non-chunked masks
                mask_dict = build_mask(incidence, device)  # Dict(overlap: [|E|, |E|] sparse)
                overlaps, masks = list(mask_dict.keys()), list(mask_dict.values())
                overlaps = torch.tensor(overlaps, dtype=torch.long, device=device)  # [|overlaps|,]
                masks = torch.stack(masks, dim=0).coalesce()  # [|overlaps|, |E|, |E|]]
                assert (masks.indices() - masks_chunk.indices() == 0).all()
                assert (masks.values() - masks_chunk.values() == 0).all()
                assert (overlaps - overlaps_chunk == 0).all()

                masks = masks_chunk
                overlaps = overlaps_chunk
                n_overlaps = len(overlaps)

                normalizer = (torch.sparse.sum(masks, 2).to_dense().unsqueeze(-1))  # [|overlaps|, |E|, 1]
                normalizer = normalizer.masked_fill_(normalizer == 0, 1e-5)

            ehnn_cache = dict(
                incidence=incidence,
                edge_orders=edge_orders,
                overlaps=overlaps,
                n_overlaps=n_overlaps,
                prefix_normalizer=prefix_normalizer,
                suffix_normalizer=suffix_normalizer,
            )

            torch.save(ehnn_cache, f"./cache/{args.dname}.pt")
            print(f"saved ehnn_cache for {args.dname}")
        else:
            ehnn_cache = torch.load(f"./cache/{args.dname}.pt")
        print(f"number of mask channels: {ehnn_cache['n_overlaps']}")
    else:
        raise NotImplementedError

    # Get splits
    split_idx_lst = []
    for run in range(args.runs):
        split_idx = rand_train_test_idx(data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
        split_idx_lst.append(split_idx)

    # Load model
    model = parse_method(args, ehnn_cache)
    model, data = model.to(device), data.to(device)
    num_params = count_parameters(model)
    print(f"num params: {num_params}")

    # Main training and evaluation
    logger = Logger(args.runs, args)
    criterion = nn.NLLLoss()
    eval_func = eval_acc
    model.train()

    # Training loop
    runtime_list = []
    for run in range(args.runs):
        start_time = time.time()
        split_idx = split_idx_lst[run]
        train_idx = split_idx["train"].to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        best_val = float("-inf")

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data, ehnn_cache)  # [N, D]
            out = F.log_softmax(out, dim=1)
            loss = criterion(out[train_idx], data.y[train_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            result = evaluate(args, model, data, split_idx, eval_func, None, ehnn_cache)
            logger.add_result(run, result[:3])

            if epoch % args.display_step == 0 and args.display_step > 0:
                print(
                    f"Epoch: {epoch:02d}, "
                    f"Train Loss: {loss:.4f}, "
                    f"Valid Loss: {result[4]:.4f}, "
                    f"Test  Loss: {result[5]:.4f}, "
                    f"Train Acc: {100 * result[0]:.2f}%, "
                    f"Valid Acc: {100 * result[1]:.2f}%, "
                    f"Test  Acc: {100 * result[2]:.2f}%"
                )

        end_time = time.time()
        runtime_list.append(end_time - start_time)

    # Save results
    avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)
    best_val, best_test, msg = logger.print_statistics()
    arguments = [(str(key) + ":" + str(getattr(args, key))) for key in vars(args)]
    msg += "_".join(arguments) + "\n"
    res_root = "exp_result"
    if not osp.isdir(res_root):
        os.makedirs(res_root)

    filename = f"{res_root}/{args.dname}_noise_{args.feature_noise}.csv"
    print(f"Saving results to {filename}")
    with open(filename, "a+") as write_obj:
        cur_line = f"{args.method}_{args.lr}_{args.wd}_{args.ehnn_n_heads}"
        cur_line += f",{best_val.mean():.3f} ± {best_val.std():.3f}"
        cur_line += f",{best_test.mean():.3f} ± {best_test.std():.3f}"
        cur_line += f",{num_params}, {avg_time:.2f}s, {std_time:.2f}s"
        cur_line += f",{avg_time // 60}min{(avg_time % 60):.2f}s"
        cur_line += f"\n"
        write_obj.write(cur_line)

    all_args_file = f"{res_root}/all_args_{args.dname}_noise_{args.feature_noise}.csv"
    with open(all_args_file, "a+") as f:
        f.write(str(args))
        f.write("\n")

    print("All done! Exit python code")
    quit()
