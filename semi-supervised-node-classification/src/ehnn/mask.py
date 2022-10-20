import time
from collections import defaultdict

import torch


@torch.no_grad()
def build_mask_chunk(incidence, device):
    tic = time.time()
    stat_overlap_num = {}  # key: overlap, value: number of edges
    incidence = incidence.to(device)
    N = incidence.size(0)
    E = incidence.size(1)
    chunk_size = 1000
    chunk_begin_index = 0
    output = torch.sparse_coo_tensor(
        torch.empty([2, 0]), [], [E, E], device=device
    ).coalesce()
    target_indices = incidence.indices()[
        [1, 0]
    ]  # caution: indices of transposed incidence matrix
    row_indices = target_indices[0]  # [|E|,]
    target_values = incidence.values()  # [|E|,]

    while chunk_begin_index < E:
        chunk_end_index = min(chunk_begin_index + chunk_size, incidence.size(1))
        chunk_mask = torch.logical_and(
            row_indices >= chunk_begin_index, row_indices < chunk_end_index
        )  # [|chunk|]
        chunk_matrix = torch.sparse_coo_tensor(
            target_indices[:, chunk_mask],
            target_values[chunk_mask],
            [E, N],
            device=device,
        ).coalesce()  # [2, |chunk|]
        del chunk_mask
        print(
            f"chunk_begin_index: {chunk_begin_index}/{E}, "
            f"output: {output.indices().size(1)} nnz, "
            f"chunk_matrix: {chunk_matrix.indices().size(1)} nnz"
        )
        chunk_output = torch.sparse.mm(chunk_matrix, incidence)
        output = (output + chunk_output).coalesce()
        del chunk_matrix
        chunk_begin_index += chunk_size

    overlap_mask_indices = defaultdict(lambda: [[], []])
    for overlap, row_idx, col_idx in zip(
        output.values(), output.indices()[0], output.indices()[1]
    ):
        overlap = int(overlap.item())
        overlap_mask_indices[overlap][0].append(row_idx.item())
        overlap_mask_indices[overlap][1].append(col_idx.item())

    overlap_masks = {}
    for overlap, mask_indices in overlap_mask_indices.items():
        overlap_masks[overlap] = torch.sparse_coo_tensor(
            mask_indices, torch.ones(len(mask_indices[0])), (E, E), device=device
        )
    print(f"Took {time.time() - tic} seconds")
    return overlap_masks


@torch.no_grad()
def build_mask(incidence, device):
    tic = time.time()
    overlap_masks = {}
    # stat_overlap_num = {}  # Dict(overlap: number of edges)

    incidence = incidence.to(device)

    E = incidence.size(1)
    pre_overlap = torch.sparse.mm(incidence.t(), incidence).coalesce()  # [|E|, |E|]
    # stat_num_entry = len(pre_overlap.indices()[0])
    overlap_mask_indices = defaultdict(lambda: [[], []])

    # Generate Overlap masks - Not made for overlap mask of overlap 0
    for overlap, row_idx, col_idx in zip(
        pre_overlap.values(), pre_overlap.indices()[0], pre_overlap.indices()[1]
    ):
        overlap = int(overlap.item())
        overlap_mask_indices[overlap][0].append(row_idx.item())
        overlap_mask_indices[overlap][1].append(col_idx.item())

    for overlap, mask_indices in overlap_mask_indices.items():
        overlap_masks[overlap] = torch.sparse_coo_tensor(
            mask_indices, torch.ones(len(mask_indices[0])), (E, E), device=device
        )
    print(f"Took {time.time() - tic} seconds")

    return overlap_masks
