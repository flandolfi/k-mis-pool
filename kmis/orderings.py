from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch_geometric.utils import get_laplacian
from torch_sparse import SparseTensor

from kmis.utils import get_ranking


class Ordering(ABC):
    def __init__(self, descending=True):
        self.descending = descending

    def __call__(self, x: Tensor, adj: SparseTensor) -> Tensor:
        return get_ranking(self._compute(x, adj), self.descending)

    @abstractmethod
    def _compute(self, x: Tensor, adj: SparseTensor) -> Tensor:
        raise NotImplementedError


class Random(Ordering):
    def __call__(self, x: Tensor, adj: SparseTensor) -> Tensor:
        return self._compute(x, adj)

    def _compute(self, x: Tensor, adj: SparseTensor) -> Tensor:
        return torch.randperm(adj.size(0), device=adj.device())


class Degree(Ordering):
    def _compute(self, x: Tensor, adj: SparseTensor) -> Tensor:
        return adj.sum(1).view(-1)


class KPaths(Ordering):
    def __init__(self, k=1, descending=True):
        super(KPaths, self).__init__(descending)
        self.k = k

    def _compute(self, x: Tensor, adj: SparseTensor) -> Tensor:
        k_paths = torch.ones((adj.size(1), 1), dtype=torch.float, device=adj.device())

        for _ in range(self.k):
            k_paths = adj @ k_paths

        return k_paths.view(-1)


class Curvature(Ordering):
    def __init__(self, descending=True, normalization=None, k=1):
        super(Curvature, self).__init__(descending)
        self.normalization = normalization
        self.k = k

    def _compute(self, x: Tensor, adj: SparseTensor) -> Tensor:
        row, col, val = adj.coo()
        lap_idx, lap_val = get_laplacian(torch.stack((row, col)), val,
                                         self.normalization, num_nodes=adj.size(0))
        lap = SparseTensor.from_edge_index(lap_idx, lap_val, adj.sparse_sizes())
        H = x

        for _ in range(self.k):
            H = lap @ H

        return 0.5*torch.norm(H, p=2, dim=-1)


class Lambda(Ordering):
    def __init__(self, function, dim=-1, descending=True, **kwargs):
        super(Lambda, self).__init__(descending)
        self.function = function
        self.dim = dim
        self.kwargs = kwargs

    def _compute(self, x: Tensor, adj: SparseTensor) -> Tensor:
        out = self.function(x, dim=self.dim, **self.kwargs)

        if isinstance(out, tuple):
            return out[0]

        return out
