import torch
import torch_sparse
from torch_geometric.data import Data


class Ordering:
    def __init__(self, descending=False):
        self.descending = descending

    def __call__(self, data: Data):
        raise NotImplementedError


class Random(Ordering):
    def __call__(self, data: Data):
        return torch.randperm(data.num_nodes)


class Canonical(Ordering):
    def __call__(self, data: Data):
        return torch.arange(data.num_nodes)[::(-1)**int(self.descending)]


class InverseCanonical(Canonical):
    def __init__(self):
        super(InverseCanonical, self).__init__(True)


class KHopDegree(Ordering):
    def __init__(self, k=1, descending=False):
        super(KHopDegree, self).__init__(descending)
        self.k = k

    def __call__(self, data: Data):
        out = torch.ones([data.num_nodes, 1], dtype=torch.float)
        ind, val, n = data.edge_index, data.edge_attr, data.num_nodes

        if val is None:
            val = torch.ones_like(ind[0], dtype=torch.float)

        for _ in range(self.k):
            out = torch_sparse.spmm(ind, val, n, out)

        return out * (-1)**int(self.descending)


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
