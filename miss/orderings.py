from abc import ABC, abstractmethod

import torch
from torch_geometric.utils import get_laplacian
from torch_sparse import SparseTensor


class Ordering(ABC):
    cacheable = False

    def __init__(self, descending=True):
        self.descending = descending

    def __call__(self, x: torch.FloatTensor, adj: SparseTensor):
        perm = torch.argsort(self._compute(x, adj), 0, self.descending)
        rank = torch.zeros_like(perm)
        rank[perm] = torch.arange(rank.size(0), dtype=torch.long, device=rank.device)
        return rank

    @abstractmethod
    def _compute(self, x: torch.FloatTensor, adj: SparseTensor):
        raise NotImplementedError


class Random(Ordering):
    def __call__(self, x: torch.FloatTensor, adj: SparseTensor):
        return self._compute(x, adj)

    def _compute(self, x: torch.FloatTensor, adj: SparseTensor):
        return torch.randperm(adj.size(0), device=adj.device())


class Degree(Ordering):
    cacheable = True

    def _compute(self, x: torch.FloatTensor, adj: SparseTensor):
        return adj.sum(1).view(-1)


class KPath(Ordering):
    cacheable = True

    def __init__(self, k=1, descending=True):
        super(KPath, self).__init__(descending)
        self.k = k

    def _compute(self, x: torch.FloatTensor, adj: SparseTensor):
        k_paths = torch.ones((adj.size(1), 1), dtype=torch.float, device=adj.device())

        for _ in range(self.k):
            k_paths = adj @ k_paths

        return k_paths.view(-1)


class Curvature(Ordering):
    def __init__(self, descending=True, normalization=None):
        super(Curvature, self).__init__(descending)
        self.normalization = normalization

    def _compute(self, x: torch.FloatTensor, adj: SparseTensor):
        row, col, val = adj.coo()
        lap_idx, lap_val = get_laplacian(torch.stack((row, col)), val,
                                         self.normalization, num_nodes=adj.size(0))
        lap = SparseTensor.from_edge_index(lap_idx, lap_val, adj.sparse_sizes())
        return 0.5*torch.norm(lap @ x, p=2, dim=-1)