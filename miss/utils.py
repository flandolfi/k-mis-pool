import torch
from torch_geometric.typing import OptTensor, Tensor, Tuple
from torch_sparse import SparseTensor


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

        mis = mis | (torch.le(rank, min_rank).squeeze(-1) & ~mask)
        mask = mis.long().unsqueeze(-1)

        for _ in range(k):
            mask = adj.matmul(mask, reduce='max')

        mask = mask.bool().squeeze(-1)
        min_rank = rank.clone()
        min_rank[mask] = rank.max()

    return mis


def maximal_independent_set(adj: SparseTensor, rank: OptTensor = None) -> Tensor:
    return maximal_k_independent_set(adj, 1, rank)


def geodesic_fps(adj: SparseTensor, max_nodes: int, rank: OptTensor = None) -> Tuple[Tensor, int]:
    n, device = adj.size(0), adj.device()
    adj = adj.set_value(None, layout=None).set_diag()

    if rank is None:
        rank = torch.arange(n, dtype=torch.long, device=device)

    rank = rank.unsqueeze(-1)
    min_rank = rank.clone()
    stride = 0

    while n > max_nodes:
        min_rank = adj.matmul(min_rank, reduce='min')
        n = min_rank.unique(dtype=torch.long).size(0)
        stride += 1

    return torch.eq(rank, min_rank).view(-1), stride
