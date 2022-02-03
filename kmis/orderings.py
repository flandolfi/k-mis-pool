from abc import ABC, abstractmethod

import torch
from torch_sparse import SparseTensor
from torch_scatter import scatter_add
from torch_geometric.typing import Tensor, OptTensor


def get_ranking(value: Tensor, descending: bool = True) -> Tensor:
    perm = torch.argsort(value.view(-1), 0, descending)
    rank = torch.zeros_like(perm)
    rank[perm] = torch.arange(rank.size(0), dtype=torch.long, device=rank.device)
    
    return rank


def get_approx_k_neighbors(adj: SparseTensor, k: int = 1, weight: OptTensor = None) -> Tensor:
    if weight is None:
        weight = torch.ones((adj.size(0),), dtype=torch.float, device=adj.device())
    else:
        weight = weight.view(-1)
    
    if k == 0:
        return weight
    
    row, col, _ = adj.coo()
    last_k_paths = torch.ones_like(weight)
    k_sums = weight.clone()
    scatter_add(k_sums[row], col, out=k_sums)
    
    for _ in range(1, k):
        scatter_add(last_k_paths[row], col, out=last_k_paths)
        scatter_add(k_sums[row], col, out=k_sums)
    
    return k_sums / last_k_paths


class Ordering(ABC):
    def __call__(self, x: Tensor, adj: SparseTensor) -> Tensor:
        assert x.dim() == 1 or x.size(1) == 1
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
        return x/get_approx_k_neighbors(adj, self.k, x)


class DivKDegree(Ordering):
    def __init__(self, k: int = 1):
        self.k = k

    def _compute(self, x: Tensor, adj: SparseTensor) -> Tensor:
        return x/get_approx_k_neighbors(adj, self.k)


class InvKDegree(DivKDegree):
    def _compute(self, x: Tensor, adj: SparseTensor) -> Tensor:
        x = torch.ones((adj.size(0),), dtype=torch.float, device=adj.device())
        return super(InvKDegree, self)._compute(x, adj)


class DenseDivKSum(Ordering):
    def __init__(self, k: int = 1):
        self.k = k

    def _compute(self, x: Tensor, adj: SparseTensor) -> Tensor:
        adj = adj.set_value(None, layout=None).fill_diag(None)
        adj_pow = adj.clone()
    
        for _ in range(1, self.k):
            adj_pow @= adj
            adj_pow.set_value_(None, layout=None)
    
        k_sums = adj_pow @ x.view(-1, 1)
    
        return x.view(-1) / k_sums.view(-1)


class DenseDivKDegree(Ordering):
    def __init__(self, k: int = 1):
        self.k = k

    def _compute(self, x: Tensor, adj: SparseTensor) -> Tensor:
        adj = adj.set_value(None, layout=None).fill_diag(None)
        adj_pow = adj.clone()
    
        for _ in range(1, self.k):
            adj_pow @= adj
            adj_pow.set_value_(None, layout=None)

        return x.view(-1) / adj_pow.sum(-1).view(-1)


class DenseInvKDegree(DivKDegree):
    def _compute(self, x: Tensor, adj: SparseTensor) -> Tensor:
        x = torch.ones((adj.size(0),), dtype=torch.float, device=adj.device())
        return super(DenseInvKDegree, self)._compute(x, adj)
