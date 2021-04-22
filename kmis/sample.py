import torch
from torch_geometric.typing import OptTensor, Tensor
from torch_sparse import SparseTensor
from torch_scatter import scatter_min, scatter_max


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
            scatter_min(min_rank[row], col, out=min_rank)
        
        mis = mis | torch.eq(rank, min_rank)
        mask = mis.byte()
        
        for _ in range(k):
            scatter_max(mask[row], col, out=mask)
        
        mask = mask.to(dtype=torch.bool)
        min_rank = rank.clone()
        min_rank[mask] = n
    
    return mis


@torch.jit.script
def maximal_independent_set(adj: SparseTensor, rank: OptTensor = None) -> Tensor:
    return maximal_k_independent_set(adj, 1, rank)


@torch.jit.script
def sample_multinomial(prob: SparseTensor) -> Tensor:
    n, device = prob.size(0), prob.device()
    row, col, p = prob.coo()
    assert p is not None

    p_cumsum = torch.cumsum(p, dim=0) - row.float()
    sample = torch.rand(n, dtype=torch.float, device=device)

    mask = sample[row] < p_cumsum
    return scatter_min(col[mask], row[mask], dim_size=n)[0]
