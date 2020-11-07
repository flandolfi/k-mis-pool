import torch
from torch_geometric.typing import OptTensor, Optional, Tensor
from torch_sparse import SparseTensor
from torch_scatter import scatter


def scatter_and(src: Tensor, index: Tensor,
                dim: int = -1, out: OptTensor = None,
                dim_size: Optional[int] = None) -> Tensor:
    if out is not None:
        out = out.to(torch.uint8)

    return scatter(src.to(torch.uint8), index, dim, out, dim_size, "min").to(torch.bool)


def scatter_or(src: Tensor, index: Tensor,
               dim: int = -1, out: OptTensor = None,
               dim_size: Optional[int] = None) -> Tensor:
    if out is not None:
        out = out.to(torch.uint8)

    return scatter(src.to(torch.uint8), index, dim, out, dim_size, "max").to(torch.bool)


def sparse_min_sum_mm(lhs: SparseTensor, rhs: SparseTensor) -> SparseTensor:
    assert lhs.size(1) == rhs.size(0)

    l_col, l_row, l_val = lhs.t().coo()
    r_row, r_col, r_val = rhs.coo()

    l_col_count = l_col.bincount(minlength=lhs.size(1))
    r_row_count = r_row.bincount(minlength=rhs.size(0))

    rep = r_row_count[l_col]

    perm = torch.arange(r_row.size(0)).split(r_row_count.tolist())
    perm = torch.cat([t.repeat(r) for t, r in zip(perm, l_col_count)])

    o_row = l_row.repeat_interleave(rep)
    o_col = r_col[perm]
    o_val = l_val.repeat_interleave(rep) + r_val[perm]

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


def maximal_independent_set(adj: SparseTensor, rank: OptTensor = None) -> Tensor:
    row, col, _ = adj.coo()
    n, device = adj.size(0), adj.device()

    if rank is None:
        rank = torch.arange(n, dtype=torch.long, device=device)

    # Remove self-loops (should not be a problem if there are no ties)
    edge_mask = ~torch.eq(row, col)
    row, col = row[edge_mask], col[edge_mask]

    mis = torch.zeros(n, dtype=torch.bool, device=device)
    excl = torch.zeros_like(mis)
    edge_mask = torch.lt(rank[row], rank[col])
    mask = scatter_and(edge_mask, row, out=torch.ones_like(mis))

    while mask.any():
        mis |= mask
        excl = mis | scatter_or(mis[row], col, out=excl)
        edge_mask = torch.lt(rank[row], rank[col]) | excl[col]
        mask = scatter_and(edge_mask, row, out=~excl)

    return mis
