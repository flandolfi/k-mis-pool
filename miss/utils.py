import torch
from torch_geometric.typing import OptTensor, Tensor, Tuple
from torch_sparse import SparseTensor, diag
from torch_scatter import scatter_min, scatter_max


def get_ranking(value: Tensor, descending: bool = True) -> Tensor:
    perm = torch.argsort(value, 0, descending)
    rank = torch.zeros_like(perm)
    rank[perm] = torch.arange(rank.size(0), dtype=torch.long, device=rank.device)
    
    return rank


def sparse_min_sum_mm(lhs: SparseTensor, rhs: SparseTensor) -> SparseTensor:
    assert lhs.size(1) == rhs.size(0)

    l_col, l_row, l_val = lhs.t().coo()
    r_row, r_col, r_val = rhs.coo()

    l_col_count = l_col.bincount(minlength=lhs.size(1))
    r_row_count = r_row.bincount(minlength=rhs.size(0))

    rep = r_row_count[l_col]

    perm = torch.arange(r_row.size(0)).split(r_row_count.tolist())
    perm = torch.cat([t.repeat(r) for t, r in zip(perm, l_col_count)])

    o_row = l_row.repeat_interleave(rep, dim=0)
    o_col = r_col[perm]
    o_val = l_val.repeat_interleave(rep, dim=0) + r_val[perm]

    return SparseTensor(
            row=o_row, col=o_col, value=o_val,
            sparse_sizes=(lhs.size(0), rhs.size(1))
        ).coalesce("min")


def sparse_matrix_power(matrix: SparseTensor, p: int = 2, min_sum: bool = False) -> SparseTensor:
    if p == 0:
        return SparseTensor.eye(matrix.size(0), matrix.size(1)).fill_diag(int(not min_sum))

    _mat_mul = sparse_min_sum_mm if min_sum else SparseTensor.matmul

    def _mat_pow(m: SparseTensor, k: int) -> SparseTensor:
        if k == 1:
            return m.clone()

        m_pow = _mat_pow(m, k//2)
        m_pow = _mat_mul(m_pow, m_pow)

        if k % 2 == 1:
            m_pow = _mat_mul(m_pow, m)

        return m_pow

    return _mat_pow(matrix, p)


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
def maximal_independent_sample(adj: SparseTensor, max_nodes: int) -> Tuple[Tensor, int]:
    n, device = adj.size(0), adj.device()
    
    if max_nodes >= n:
        return torch.ones(n, dtype=torch.bool, device=device), 0
    
    r_bound = 1
    mis_size = n
    mis = torch.empty(n, dtype=torch.bool, device=device)
    rank = torch.randperm(n, device=device)
    
    while mis_size > max_nodes:
        mis = maximal_k_independent_set(adj, r_bound, rank)
        mis_size = torch.count_nonzero(mis)
        
        if mis_size > max_nodes:
            r_bound *= 2
    
    l_bound = r_bound // 2
    
    while l_bound <= r_bound - 1:
        k = (r_bound + l_bound)//2
        
        if k == l_bound:
            break
        
        new_mis = maximal_k_independent_set(adj, k, rank)
        mis_size = torch.count_nonzero(new_mis)
        
        if mis_size > max_nodes:
            l_bound = k
        else:
            mis = new_mis.clone()
            r_bound = k
            
    return mis, r_bound
