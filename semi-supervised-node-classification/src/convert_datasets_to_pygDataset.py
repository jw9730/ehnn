import os
import os.path as osp
import pickle

import torch
from torch_geometric.data import Data, InMemoryDataset

from load_other_datasets import load_cornell_dataset, load_yelp_dataset, load_LE_dataset


def save_data_to_pickle(data, p2root="../data/", file_name=None):
    """
    if file name not specified, use time stamp.
    """
    tmp_data_name = file_name
    path_name = osp.join(p2root, tmp_data_name)
    if not osp.isdir(p2root):
        os.makedirs(p2root)
    with open(path_name, "bw") as f:
        pickle.dump(data, f)
    return path_name


class dataset_Hypergraph(InMemoryDataset):
    def __init__(
        self,
        root="../data/pyg_data/hypergraph_dataset_updated/",
        name=None,
        p2raw=None,
        train_percent=0.01,
        feature_noise=None,
    ):

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
        if name not in existing_dataset:
            raise ValueError(
                f"name of hypergraph dataset must be one of: {existing_dataset}"
            )
        else:
            self.name = name

        self.feature_noise = feature_noise

        self._train_percent = train_percent

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{p2raw}" does not exist!'
            )

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root
        self.myraw_dir = osp.join(root, self.name, "raw")
        self.myprocessed_dir = osp.join(root, self.name, "processed")

        super(dataset_Hypergraph, self).__init__(osp.join(root, name))

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_percent = self.data.train_percent

    @property
    def raw_file_names(self):
        if self.feature_noise is not None:
            file_names = [f"{self.name}_noise_{self.feature_noise}"]
        else:
            file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        if self.feature_noise is not None:
            file_names = [f"data_noise_{self.feature_noise}.pt"]
        else:
            file_names = ["data.pt"]
        return file_names

    @property
    def num_features(self):
        return self.data.num_node_features

    def download(self):
        for name in self.raw_file_names:
            p2f = osp.join(self.myraw_dir, name)
            if not osp.isfile(p2f):
                # file not exist, so we create it and save it there.

                if self.name in ["walmart-trips-100", "house-committees-100"]:
                    if self.feature_noise is None:
                        raise ValueError(
                            f"for cornell datasets, feature noise cannot be {self.feature_noise}"
                        )
                    feature_dim = int(self.name.split("-")[-1])
                    tmp_name = "-".join(self.name.split("-")[:-1])
                    tmp_data = load_cornell_dataset(
                        path=self.p2raw,
                        dataset=tmp_name,
                        feature_dim=feature_dim,
                        feature_noise=self.feature_noise,
                        train_percent=self._train_percent,
                    )
                    print(f"num_node: {tmp_data.n_x}")
                    print(f"num_edge: {tmp_data.num_hyperedges}")

                elif self.name == "yelp":
                    tmp_data = load_yelp_dataset(
                        path=self.p2raw,
                        dataset=self.name,
                        train_percent=self._train_percent,
                    )
                    print(f"num_node: {tmp_data.n_x}")
                    print(f"num_edge: {tmp_data.num_hyperedges}")

                else:
                    tmp_data = load_LE_dataset(
                        path=self.p2raw,
                        dataset=self.name,
                        train_percent=self._train_percent,
                    )
                    print(f"num_node: {tmp_data.n_x}")
                    print(f"num_edge: {tmp_data.num_hyperedges}")

                _ = save_data_to_pickle(
                    tmp_data, p2root=self.myraw_dir, file_name=self.raw_file_names[0]
                )
            else:
                pass

    def process(self):
        p2f = osp.join(self.myraw_dir, self.raw_file_names[0])
        with open(p2f, "rb") as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return "{}()".format(self.name)
