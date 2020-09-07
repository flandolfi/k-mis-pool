import torch
from torch_sparse import SparseTensor


def sparse_min_sum_mm(lhs: SparseTensor, rhs: SparseTensor):
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


def pairwise_distances(adj: SparseTensor, max_value=None):
    if max_value is None:
        max_value = float("inf")

    dists = adj.fill_diag(0.)
    old_v = adj.storage.value().clone()

    for _ in range(1, adj.size(0)):
        step = sparse_min_sum_mm(dists, adj)
        r, c, v = [torch.cat(ts) for ts in zip(dists.coo(), step.coo())]
        mask = v <= max_value
        r, c, v = r[mask], c[mask], v[mask]
        dists = SparseTensor(row=r, col=c, value=v).coalesce("min")

        if dists.storage.value().equal(old_v):
            break

        old_v = dists.storage.value().clone()

    return dists


def sparse_matrix_power(matrix: SparseTensor, p=2):
    if p == 0:
        return SparseTensor.eye(matrix.size(0), matrix.size(1))

    def _mat_pow(m: SparseTensor, k):
        if k == 1:
            return m.clone()

        m_pow = _mat_pow(m, k//2)
        m_pow = m_pow @ m_pow

        if k % 2 == 1:
            m_pow = m_pow @ m

        return m_pow

    return _mat_pow(matrix, p)


def maximal_independent_set(adj: SparseTensor, rank=None):
    row, col, _ = adj.clone().coo()
    n, device = adj.size(0), adj.device()

    if rank is None:
        rank = torch.arange(n, dtype=torch.long, device=device)

    # Remove self-loops (should not be a problem if there are no ties)
    edge_mask = row != col
    row, col = row[edge_mask], col[edge_mask]

    mis = torch.zeros(n, dtype=torch.bool, device=device)
    excl = mis.clone()
    edge_mask = rank[row] >= rank[col]
    mask = ~torch.scatter_add(mis, 0, row, edge_mask)

    while mask.any():
        mis |= mask
        excl = mis | torch.scatter_add(excl, 0, col, mis[row])
        edge_mask = (rank[row] >= rank[col]) & ~excl[col]
        mask = ~torch.scatter_add(excl, 0, row, edge_mask)

    return mis
