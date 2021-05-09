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


def get_diagonal_matrix(diag: Tensor):
    diag = diag.view(-1)
    idx = torch.arange(diag.size(0), dtype=torch.long, device=diag.device)
    return SparseTensor(row=idx, col=idx.clone(), value=diag)


def get_laplacian_matrix(adj: SparseTensor, normalization: str = None):
    lap = adj.set_value(-adj.storage.value(), layout="coo")
    deg = adj.sum(-1)
    lap = lap.set_diag(lap.get_diag() + deg)

    if normalization == 'sym':
        norm = torch.where(deg == 0, torch.ones_like(deg), 1./torch.sqrt(deg))
        lap = (lap * norm.unsqueeze(0)) * norm.unsqueeze(-1)
    elif normalization == 'rw':
        norm = torch.where(deg == 0, torch.ones_like(deg), 1./deg)
        lap = lap * norm.unsqueeze(-1)

    return lap


def get_incidence_matrix(adj: SparseTensor):
    row, col, val = adj.coo()
    n = adj.size(0)
    m = row.size(0)

    if val is None:
        val = torch.ones_like(row, dtype=torch.float)

    val = torch.sqrt(val)
    idx = torch.arange(m, dtype=torch.long, device=adj.device())
    return SparseTensor(row=torch.cat([row, col], dim=0),
                        col=torch.cat([idx, idx], dim=0),
                        value=torch.cat([val, -val], dim=0),
                        sparse_sizes=(n, m))


def sparse_eig(mat: SparseTensor, k: int = None, largest=True):
    n, m = mat.sparse_sizes()
    assert n == m, "`mat` must be symmetric."

    if k is None or n < 3*k:
        mat = mat.to_dense()
        l, U = torch.eig(mat, eigenvectors=True)
        l = l.T[0]

        if k is None:
            l, perm = torch.sort(l, descending=largest)
        else:
            l, perm = torch.topk(l, k=k, largest=largest, sorted=True)
        
        U = U[:, perm]
    else:
        mat = mat.to_torch_sparse_coo_tensor()
        l, U = torch.lobpcg(mat, k=k, largest=largest, tol=1e-3)

    return l, U
