from abc import ABC, abstractmethod

import torch
from torch_geometric.utils import get_laplacian
from torch_sparse import SparseTensor


class Ordering(ABC):
    cacheable = False

    def __init__(self, descending=True):
        self.descending = descending

    def __call__(self, x: torch.FloatTensor, adj: SparseTensor):
        out = torch.argsort(self._compute(x, adj), 0, self.descending)
        return out

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


class Curvature(Ordering):
    def __init__(self, descending=True, normalization=None):
        super(Curvature, self).__init__(descending)
        self.normalization = normalization

    def _compute(self, x: torch.FloatTensor, adj: SparseTensor):
        row, col, val = adj.coo()
        lap_idx, lap_val = get_laplacian(torch.stack((row, col)), val,
                                         self.normalization, torch.float, adj.size(0))
        lap = SparseTensor.from_edge_index(lap_idx, lap_val, adj.sparse_sizes(), True)
        return 0.5*torch.norm(lap @ x, p=2, dim=-1)
