import inspect
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_scatter import scatter_mean

from kmis import utils


class Ordering(ABC):
    def __init__(self, descending=True):
        self.descending = descending

    def __call__(self, x: Tensor, adj: SparseTensor) -> Tensor:
        return utils.get_ranking(self._compute(x, adj), self.descending)

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


class RandomWalk(Ordering):
    def __init__(self, k=1, descending=True):
        super(RandomWalk, self).__init__(descending)
        self.k = k

    def _compute(self, x: Tensor, adj: SparseTensor) -> Tensor:
        n = adj.size(1)
        p = torch.ones((n, 1), dtype=torch.float, device=adj.device())/n
        rw = utils.normalize_dim(adj).t()

        for _ in range(self.k):
            p = rw @ p 

        return p.view(-1)


class Curvature(Ordering):
    def __init__(self, descending=True, normalization=None, k=1):
        super(Curvature, self).__init__(descending)
        self.normalization = normalization
        self.k = k

    def _compute(self, x: Tensor, adj: SparseTensor) -> Tensor:
        lap = utils.get_laplacian_matrix(adj, self.normalization)
        H = x

        for _ in range(self.k):
            H = lap @ H

        return 0.5*torch.norm(H, p=2, dim=-1)


class LocalVariation(Ordering):
    def __init__(self, descending=True, k=1):
        super(LocalVariation, self).__init__(descending)
        self.k = k

    def _compute(self, x: Tensor, adj: SparseTensor) -> Tensor:
        rw = utils.normalize_dim(adj, -1)
        rw.storage._value *= 0.5
        rw = rw.set_diag(rw.get_diag() + 0.5)
        inc = utils.get_incidence_matrix(adj)
        n_id, e_id, _ = inc.coo() 
        var = x

        for _ in range(self.k):
            var = rw @ var
        
        var = torch.square(inc.t() @ (x - var))
        var = scatter_mean(var[e_id], n_id, dim=0, dim_size=rw.size(0))
        norm = torch.norm(x, p=2, dim=0, keepdim=True)
        return torch.mean(var/norm, dim=-1)


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


def get_ordering(name: str, **kwargs) -> Ordering:
    tokens = name.split('-')
    opts = {'descending': True}

    if len(tokens) > 1 and tokens[0] in {'min', 'max'}:
        opts['descending'] = tokens[0] == 'max'
        tokens = tokens[1:]

    cls_name = ''.join(t.title() for t in tokens)
    glob = globals()

    if cls_name not in glob:
        return Lambda(getattr(torch, tokens[-1]), **opts)

    cls = glob[cls_name]
    keys = kwargs.keys() & dict(inspect.signature(cls.__init__).parameters).keys()

    for k in keys:
        opts[k] = kwargs[k]

    return cls(**opts)
