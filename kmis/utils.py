import torch

from torch_geometric.typing import Tensor, SparseTensor


def get_ranking(value: Tensor, descending: bool = True) -> Tensor:
    perm = torch.argsort(value, 0, descending)
    rank = torch.zeros_like(perm)
    rank[perm] = torch.arange(rank.size(0), dtype=torch.long, device=rank.device)
    
    return rank


def normalize_dim(mat: SparseTensor, dim: int = -1) -> SparseTensor:
    norm = mat.sum(dim).unsqueeze(dim)
    norm = torch.where(norm == 0, torch.ones_like(norm), 1. / norm)
    return mat * norm
