"""
Drop-in replacements for the torch_scatter functions used in this repository, implemented with native
torch.scatter_reduce so they run on CPU, CUDA and MPS (torch_scatter only ships CPU/CUDA kernels).
Semantics follow torch_scatter: empty groups are filled with 0, and index is broadcast against src along dim.
"""
import torch

from typing import Optional, Tuple

_REDUCE_TO_TORCH = {'sum': 'sum', 'add': 'sum', 'mean': 'mean', 'max': 'amax', 'min': 'amin', 'mul': 'prod'}


def _broadcast_index(index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    if index.dim() == src.dim() and index.shape == src.shape:
        return index
    shape = [1] * src.dim()
    shape[dim] = -1
    return index.reshape(shape).expand_as(src)


def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, reduce: str = 'sum') -> torch.Tensor:
    dim = dim % src.dim()
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

    # MPS/CPU scatter_reduce amax/amin are not implemented for bool.
    was_bool = src.dtype == torch.bool
    if was_bool:
        src = src.to(torch.uint8)

    index = _broadcast_index(index, src, dim)
    if out is None:
        shape = list(src.shape)
        shape[dim] = dim_size
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)

    result = out.scatter_reduce(dim, index, src, reduce=_REDUCE_TO_TORCH[reduce], include_self=False)
    return result.bool() if was_bool else result


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1, dim_size: Optional[int] = None) -> torch.Tensor:
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='mean')


def scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim: int = -1, dim_size: Optional[int] = None) -> torch.Tensor:
    dim = dim % src.dim()
    index = _broadcast_index(index, src, dim)
    group_max = scatter(src, index, dim=dim, dim_size=dim_size, reduce='max')
    exp = (src - group_max.gather(dim, index)).exp()
    group_sum = scatter(exp, index, dim=dim, dim_size=dim_size, reduce='sum')
    return exp / group_sum.gather(dim, index)


def scatter_min(src: torch.Tensor, index: torch.Tensor, dim: int = -1, dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (min values, argmin positions along dim). Empty groups give value 0 and argmin src.size(dim), like torch_scatter."""
    dim = dim % src.dim()
    index = _broadcast_index(index, src, dim)
    min_vals = scatter(src, index, dim=dim, dim_size=dim_size, reduce='min')

    # Argmin is the first position along dim that attains the group minimum.
    positions = torch.arange(src.shape[dim], device=src.device).reshape([-1 if d == dim else 1 for d in range(src.dim())]).expand_as(src)
    is_min = src == min_vals.gather(dim, index)
    candidates = torch.where(is_min, positions, torch.full_like(positions, src.shape[dim]))
    argmin = torch.full_like(min_vals, src.shape[dim], dtype=torch.long).scatter_reduce(dim, index, candidates, reduce='amin', include_self=True)
    return min_vals, argmin
