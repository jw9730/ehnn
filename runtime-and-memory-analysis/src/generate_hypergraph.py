import torch

def generate_random_hypergraph(n_nodes, n_edges, orders):
    k_indices = torch.randint(len(orders), size=(n_edges,))
    rand_int = torch.randint(n_edges, size=())
    query_k_index = k_indices[rand_int:rand_int + 1]
    k = orders[torch.cat((query_k_index, k_indices))]

    incidence = torch.arange(n_nodes)[:, None].expand(n_nodes, 1 + n_edges) - k < 0
    permutation = torch.argsort(torch.rand(n_nodes, 1 + n_edges), dim=0)
    incidence = torch.gather(incidence, dim=0, index=permutation).float()
    incidence = incidence[incidence.sum(dim=1) != 0]

    nodes = torch.arange(incidence.size(0)).tolist()
    edges = []
    for binary_vec in incidence.unbind(1):
        edges.append(torch.nonzero(binary_vec).squeeze().tolist())

    return {'nodes': nodes, 'edges': edges, 'incidence': incidence}


def get_query_target(hypergraph):
    nodes = hypergraph['nodes']
    edges = hypergraph['edges']
    n = len(nodes)
    query_edge = edges[0]
    k = len(query_edge)
    target_edges = [e for e in edges if len(e) == len(query_edge)]
    query_labels = torch.zeros(n, ).long()
    target_labels = torch.zeros(n, ).long()
    query_labels[query_edge] = 1
    for e in target_edges:
        target_labels[e] = 1
    return query_labels, target_labels, k
