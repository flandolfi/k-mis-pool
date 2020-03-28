from abc import ABC

import torch
import torch_sparse
from torch_geometric.data import Data


class Ordering(ABC):
    def __init__(self, descending=False):
        self.descending = descending

    def __call__(self, data: Data):
        out = self._compute(data)

        if not self.descending:
            out *= -1

        if 'adj' in data and out.dim() != 2:
            out = out.view((-1, data.num_nodes)).expand_as(data.mask)
            out[~data.mask] = float('inf')

        return out

    def _compute(self, data: Data):
        raise NotImplementedError


class Random(Ordering):
    def _compute(self, data: Data):
        return torch.randperm(data.num_nodes)


class Canonical(Ordering):
    def _compute(self, data: Data):
        return torch.arange(data.num_nodes)


class InverseCanonical(Canonical):
    def __init__(self):
        super(InverseCanonical, self).__init__(True)


class KHopDegree(Ordering):
    def __init__(self, k=1, descending=False):
        super(KHopDegree, self).__init__(descending)
        self.k = k

    def _compute(self, data: Data):
        if 'adj' in data:
            out = torch.ones_like(data.mask, dtype=torch.float)
            out.unsqueeze_(-1)

            for _ in range(self.k):
                out = data.adj @ out

            return out.squeeze(-1)

        out = torch.ones([data.num_nodes, 1], dtype=torch.float)
        ind, val, n = data.edge_index, data.edge_attr, data.num_nodes

        if val is None:
            val = torch.ones_like(ind[0], dtype=torch.float)

        for _ in range(self.k):
            out = torch_sparse.spmm(ind, val, n, n, out)

        return out.view(-1)


class MinKHopDegree(KHopDegree):
    def __init__(self, k=1):
        super(MinKHopDegree, self).__init__(k, False)


class MaxKHopDegree(KHopDegree):
    def __init__(self, k=1):
        super(MaxKHopDegree, self).__init__(k, True)


class Degree(KHopDegree):
    def __init__(self, descending=False):
        super(Degree, self).__init__(1, descending)


class MinDegree(Degree):
    def __init__(self):
        super(MinDegree, self).__init__(False)


class MaxDegree(Degree):
    def __init__(self):
        super(MaxDegree, self).__init__(True)
