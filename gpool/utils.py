import torch
from torch_geometric.data import Data
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

    dists = adj.clone().fill_diag(0.)
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


def maximal_independent_set(data: Data, perm=None):
    n, (row, col) = data.num_nodes, data.edge_index.clone()
    device = row.device

    if perm is None:
        perm = torch.arange(n, dtype=torch.long, device=device)

    mis = torch.zeros(n, dtype=torch.bool, device=device)
    mask = mis.clone()
    edge_mask = perm[row] > perm[col]
    mask = torch.scatter_add(mask, 0, row, edge_mask)

    while not mask.all():
        mis |= ~mask
        edge_mask = mask[row] | mask[col]
        row, col = row[edge_mask], col[edge_mask]
        edge_mask = perm[row] > perm[col]
        mask = torch.scatter_add(torch.zeros_like(mask), 0, row, edge_mask)

    return mis
