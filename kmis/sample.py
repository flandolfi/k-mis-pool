import torch
from torch_geometric.typing import OptTensor, Tensor
from torch_sparse import SparseTensor
from torch_scatter import scatter_min, scatter_max


@torch.no_grad()
@torch.jit.script
def maximal_k_independent_set(adj: SparseTensor, k: int = 1, rank: OptTensor = None) -> Tensor:
    n, device = adj.size(0), adj.device()
    row, col, val = adj.coo()
    
    if rank is None:
        rank = torch.arange(n, dtype=torch.long, device=device)
    
    mis = torch.zeros(n, dtype=torch.bool, device=device)
    mask = mis.clone()
    min_rank = rank.clone()
    
    while not mask.all():
        for _ in range(k):
            min_neigh = torch.full_like(min_rank, fill_value=n)
            scatter_min(min_rank[row], col, out=min_neigh)
            torch.minimum(min_neigh, min_rank, out=min_rank)  # self-loops
        
        mis = mis | torch.eq(rank, min_rank)
        mask = mis.clone().byte()
        
        for _ in range(k):
            max_neigh = torch.full_like(mask, fill_value=0)
            scatter_max(mask[row], col, out=max_neigh)
            torch.maximum(max_neigh, mask, out=mask)  # self-loops
        
        mask = mask.to(dtype=torch.bool)
        min_rank = rank.clone()
        min_rank[mask] = n
    
    return mis


def maximal_independent_set(adj: SparseTensor, rank: OptTensor = None) -> Tensor:
    return maximal_k_independent_set(adj, 1, rank)
