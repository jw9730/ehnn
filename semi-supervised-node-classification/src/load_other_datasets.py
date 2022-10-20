import os
import os.path as osp

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from sklearn.feature_extraction.text import CountVectorizer
from torch_geometric.data import Data
from torch_sparse import coalesce


def load_LE_dataset(path=None, dataset="ModelNet40", train_percent=0.025):
    # load edges, features, and labels.
    print("Loading {} dataset...".format(dataset))
    file_name = f"{dataset}.content"
    p2idx_features_labels = osp.join(path, dataset, file_name)
    idx_features_labels = np.genfromtxt(p2idx_features_labels, dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = torch.LongTensor(idx_features_labels[:, -1].astype(float))
    print("load features")
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    file_name = f"{dataset}.edges"
    p2edges_unordered = osp.join(path, dataset, file_name)
    edges_unordered = np.genfromtxt(p2edges_unordered, dtype=np.int32)

    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
    ).reshape(edges_unordered.shape)
    print("load edges")
    edge_index = edges.T
    assert edge_index[0].max() == edge_index[1].min() - 1
    assert len(np.unique(edge_index)) == edge_index.max() + 1

    num_nodes = edge_index[0].max() + 1
    num_he = edge_index[1].max() - num_nodes + 1

    edge_index = np.hstack((edge_index, edge_index[::-1, :]))
    data = Data(
        x=torch.FloatTensor(np.array(features[:num_nodes].todense())),
        edge_index=torch.LongTensor(edge_index),
        y=labels[:num_nodes],
    )

    total_num_node_id_he_id = len(np.unique(edge_index))
    data.edge_index, data.edge_attr = coalesce(
        data.edge_index, None, total_num_node_id_he_id, total_num_node_id_he_id
    )

    n_x = num_nodes
    data.n_x = n_x
    data.train_percent = train_percent
    data.num_hyperedges = num_he

    return data


def load_yelp_dataset(
    path="../data/raw_data/yelp_raw_datasets/",
    dataset="yelp",
    name_dictionary_size=1000,
    train_percent=0.025,
):
    """
    this will read the yelp dataset from source files, and convert it edge_list to
    [[ -V- | -E- ]
     [ -E- | -V- ]]
    each node is a restaurant, a hyperedge represent a set of restaurants one user had been to.
    node features:
        - latitude, longitude
        - state, in one-hot coding.
        - city, in one-hot coding.
        - name, in bag-of-words
    node label:
        - average stars from 2-10, converted from original stars which is binned in x.5, min stars = 1
    """
    print(f"Loading hypergraph dataset from {dataset}")

    # first load node features:
    # load longtitude and latitude of restaurant.
    latlong = pd.read_csv(osp.join(path, "yelp_restaurant_latlong.csv")).values

    # city - zipcode - state integer indicator dataframe.
    loc = pd.read_csv(osp.join(path, "yelp_restaurant_locations.csv"))
    state_int = loc.state_int.values
    city_int = loc.city_int.values

    num_nodes = loc.shape[0]
    state_1hot = np.zeros((num_nodes, state_int.max()))
    state_1hot[np.arange(num_nodes), state_int - 1] = 1

    city_1hot = np.zeros((num_nodes, city_int.max()))
    city_1hot[np.arange(num_nodes), city_int - 1] = 1

    # convert restaurant name into bag-of-words feature.
    vectorizer = CountVectorizer(
        max_features=name_dictionary_size, stop_words="english", strip_accents="ascii"
    )
    res_name = pd.read_csv(osp.join(path, "yelp_restaurant_name.csv")).values.flatten()
    name_bow = vectorizer.fit_transform(res_name).todense()
    features = np.hstack([latlong, state_1hot, city_1hot, name_bow])

    # then load node labels:
    df_labels = pd.read_csv(osp.join(path, "yelp_restaurant_business_stars.csv"))
    labels = df_labels.values.flatten()

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f"number of nodes:{num_nodes}, feature dimension: {feature_dim}")

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    # Yelp restaurant review hypergraph is store in a incidence matrix.
    H = pd.read_csv(osp.join(path, "yelp_restaurant_incidence_H.csv"))
    node_list = H.node.values - 1
    edge_list = H.he.values - 1 + num_nodes
    edge_index = np.vstack([node_list, edge_list])
    edge_index = np.hstack([edge_index, edge_index[::-1, :]])
    edge_index = torch.LongTensor(edge_index)
    data = Data(x=features, edge_index=edge_index, y=labels)

    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(
        data.edge_index, None, total_num_node_id_he_id, total_num_node_id_he_id
    )

    n_x = num_nodes
    data.n_x = n_x
    data.train_percent = train_percent
    data.num_hyperedges = H.he.values.max()

    return data


def load_cornell_dataset(
    path="../data/raw_data/",
    dataset="walmart-trips-100",
    feature_noise=0.1,
    feature_dim=None,
    train_percent=0.025,
):
    """
    this will read the yelp dataset from source files, and convert it edge_list to
    [[ -V- | -E- ]
     [ -E- | -V- ]]
    each node is a restaurant, a hyperedge represent a set of restaurants one user had been to.
    node features:
        - add gaussian noise with sigma = nosie, mean = one hot coded label.
    node label:
        - average stars from 2-10, converted from original stars which is binned in x.5, min stars = 1
    """
    print(f"Loading hypergraph dataset from cornell: {dataset}")

    # first load node labels
    df_labels = pd.read_csv(
        osp.join(path, dataset, f"node-labels-{dataset}.txt"), names=["node_label"]
    )
    num_nodes = df_labels.shape[0]
    labels = df_labels.values.flatten()

    # then create node features.
    num_classes = df_labels.values.max()
    features = np.zeros((num_nodes, num_classes))

    features[np.arange(num_nodes), labels - 1] = 1
    if feature_dim is not None:
        num_row, num_col = features.shape
        zero_col = np.zeros((num_row, feature_dim - num_col), dtype=features.dtype)
        features = np.hstack((features, zero_col))

    features = np.random.normal(features, feature_noise, features.shape)
    print(f"number of nodes:{num_nodes}, feature dimension: {features.shape[1]}")

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    # Corenll datasets are stored in lines of hyperedges. Each line is the set of nodes for that edge.
    p2hyperedge_list = osp.join(path, dataset, f"hyperedges-{dataset}.txt")
    node_list = []
    he_list = []
    he_id = num_nodes

    with open(p2hyperedge_list, "r") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            cur_set = line.split(",")
            cur_set = [int(x) for x in cur_set]

            node_list += cur_set
            he_list += [he_id] * len(cur_set)
            he_id += 1
    # shift node_idx to start with 0.
    node_idx_min = np.min(node_list)
    node_list = [x - node_idx_min for x in node_list]
    edge_index = [node_list + he_list, he_list + node_list]
    edge_index = torch.LongTensor(edge_index)
    data = Data(x=features, edge_index=edge_index, y=labels)

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(
        data.edge_index, None, total_num_node_id_he_id, total_num_node_id_he_id
    )

    n_x = num_nodes
    data.n_x = n_x
    data.train_percent = train_percent
    data.num_hyperedges = he_id - num_nodes

    return data
