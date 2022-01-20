from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_scatter import scatter_add


def get_ranking(value: Tensor, descending: bool = True) -> Tensor:
    perm = torch.argsort(value, 0, descending)
    rank = torch.zeros_like(perm)
    rank[perm] = torch.arange(rank.size(0), dtype=torch.long, device=rank.device)
    
    return rank


class Ordering(ABC):
    def __call__(self, x: Tensor, adj: SparseTensor) -> Tensor:
        return get_ranking(self._compute(x, adj), descending=True)

    @abstractmethod
    def _compute(self, x: Tensor, adj: SparseTensor) -> Tensor:
        raise NotImplementedError


class Greedy(Ordering):
    def _compute(self, x: Tensor, adj: SparseTensor) -> Tensor:
        return x


class DivKSum(Ordering):
    def __init__(self, k: int = 1):
        self.k = k

    def _compute(self, x: Tensor, adj: SparseTensor) -> Tensor:
        assert x.dim() == 1 or x.size(1) == 1
        
        row, col, _ = adj.coo()
        x = x.view(-1)
        k_sums = x.clone()

        for _ in range(self.k):
            scatter_add(k_sums[row], col, out=k_sums)

        return x / k_sums


class DivKDegree(Ordering):
    def __init__(self, k: int = 1):
        self.k = k

    def _compute(self, x: Tensor, adj: SparseTensor) -> Tensor:
        assert x.dim() == 1 or x.size(1) == 1
        
        row, col, _ = adj.coo()
        x = x.view(-1)
        k_deg = torch.ones_like(x)

        for _ in range(self.k):
            scatter_add(k_deg[row], col, out=k_deg)

        return x / k_deg


class InvKDegree(DivKDegree):
    def _compute(self, x: Tensor, adj: SparseTensor) -> Tensor:
        x = torch.ones((adj.size(0),), dtype=torch.float, device=adj.device())
        return super(InvKDegree, self)._compute(x, adj)

