"""
Drop-in replacements for the torch_cluster functions used in this repository, implemented with native torch
(cdist + topk) so they run on CPU, CUDA and MPS (torch_cluster only ships CPU/CUDA kernels).
Edges follow torch_cluster's source_to_target flow: edge_index[0] are neighbors, edge_index[1] are the centers.
The dense pairwise distance matrix is built per call, which is fine for single-complex inputs (N in the thousands).
"""
import torch

from typing import Optional


def _neighbor_search(x: torch.Tensor, k: int, loop: bool, batch: Optional[torch.Tensor], r: Optional[float]) -> torch.Tensor:
    num_nodes = x.shape[0]
    dist = torch.cdist(x, x, compute_mode='donot_use_mm_for_euclid_dist')

    invalid = torch.zeros_like(dist, dtype=torch.bool)
    if batch is not None:
        invalid |= batch[:, None] != batch[None, :]
    if not loop:
        invalid |= torch.eye(num_nodes, dtype=torch.bool, device=x.device)
    if r is not None:
        invalid |= dist > r
    dist = dist.masked_fill(invalid, float('inf'))

    vals, neighbors = dist.topk(min(k, num_nodes), dim=1, largest=False)
    valid = torch.isfinite(vals)
    centers = torch.arange(num_nodes, device=x.device)[:, None].expand_as(neighbors)
    return torch.stack([neighbors[valid], centers[valid]])


def knn_graph(x: torch.Tensor, k: int, batch: Optional[torch.Tensor] = None, loop: bool = False) -> torch.Tensor:
    return _neighbor_search(x, k, loop, batch, None)


def radius_graph(x: torch.Tensor, r: float, batch: Optional[torch.Tensor] = None, loop: bool = False, max_num_neighbors: int = 32) -> torch.Tensor:
    return _neighbor_search(x, max_num_neighbors, loop, batch, r)
