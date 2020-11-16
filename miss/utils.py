import math

import torch
from torch_geometric.typing import OptTensor, Tensor, Tuple
from torch_sparse import SparseTensor


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


def maximal_k_independent_set(adj: SparseTensor, k: int = 1,
                              rank: OptTensor = None) -> Tensor:
    n, device = adj.size(0), adj.device()
    adj = adj.set_value(None, layout=None).set_diag()

    if rank is None:
        rank = torch.arange(n, dtype=torch.long, device=device)

    mis = torch.zeros(n, dtype=torch.bool, device=device)
    mask = mis.clone()
    rank = rank.unsqueeze(-1)
    min_rank = rank.clone()

    while not mask.all():
        for _ in range(k):
            min_rank = adj.matmul(min_rank, reduce='min')

        mis = mis | torch.eq(rank, min_rank).squeeze(-1)
        mask = mis.long().unsqueeze(-1)

        for _ in range(k):
            mask = adj.matmul(mask, reduce='max')

        mask = mask.bool().squeeze(-1)
        min_rank = rank.clone()
        min_rank[mask] = n

    return mis


def maximal_independent_set(adj: SparseTensor, rank: OptTensor = None) -> Tensor:
    return maximal_k_independent_set(adj, 1, rank)


def maximal_independent_sample(adj: SparseTensor, max_nodes: int,
                               runs: int = 16, max_iterations: int = 1) -> Tuple[Tensor, int]:
    n, device = adj.size(0), adj.device()
    
    if max_nodes >= n:
        return torch.ones(n, dtype=torch.bool, device=device), 0
    
    adj = adj.set_value(None, layout=None).set_diag()
    
    for _ in range(max(1, max_iterations)):
        sample_val = torch.randint(n, size=(n, runs), dtype=torch.long, device=device)
        _, sample_idx = torch.topk(sample_val, max_nodes, largest=False, dim=0)
        
        samples = torch.zeros(n, runs, dtype=torch.bool)
        samples = torch.scatter(samples, 0, sample_idx, True)
        covered = samples.long()
        covered_count = torch.full((runs,), fill_value=max_nodes,
                                   dtype=torch.long, device=device)
        is_dominating = torch.zeros(runs, dtype=torch.bool, device=device)
        stride = 0
    
        while not is_dominating.any():
            covered = adj.matmul(covered, reduce='max')
            count = covered.sum(0)
            
            if torch.equal(count, covered_count):
                break
            
            covered_count = count
            is_dominating = torch.eq(covered_count, n)
            stride += 1
        
        stride = math.ceil(stride/2)
        
        if stride == 0:
            mis = torch.ones(n, dtype=torch.bool, device=device)
            break
        
        perm = torch.argsort(sample_val[:, is_dominating][:, 0], descending=False)
        mis = maximal_k_independent_set(adj, stride, perm)
        
        if torch.count_nonzero(mis) <= max_nodes:
            return mis, stride
        
    return mis, stride  # noqa
