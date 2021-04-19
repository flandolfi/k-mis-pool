import torch

from torch_geometric.typing import Tensor, SparseTensor
from torch_scatter import scatter


def get_ranking(value: Tensor, descending: bool = True) -> Tensor:
    perm = torch.argsort(value, 0, descending)
    rank = torch.zeros_like(perm)
    rank[perm] = torch.arange(rank.size(0), dtype=torch.long, device=rank.device)
    
    return rank


def normalize_dim(mat: SparseTensor, dim: int = -1) -> SparseTensor:
    norm = mat.sum(dim).unsqueeze(dim)
    norm = torch.where(norm == 0, torch.ones_like(norm), 1. / norm)
    return mat * norm


def sample_multinomial(prob: SparseTensor, normalize: bool = False) -> Tensor:
    if normalize:
        prob = normalize_dim(prob, -1)
    
    n, device = prob.size(0), prob.device()
    row, col, p = prob.coo()
    p_cumsum = torch.cumsum(p, dim=0) - row.float()
    sample = torch.rand(n, dtype=torch.float, device=device)
    
    mask = sample[row] < p_cumsum
    return scatter(col[mask], row[mask], dim_size=n, reduce='min')


if __name__ == "__main__":
    eye = SparseTensor.eye(10)
    print(sample_multinomial(eye))
    x = torch.rand(10, 3)
    x = SparseTensor.from_dense(x)
    x = normalize_dim(x, -1)
    print(x.to_dense())
    eye = torch.eye(3)
    samples = [eye[sample_multinomial(x)] for _ in range(10000)]
    print(torch.stack(samples).mean(0))

