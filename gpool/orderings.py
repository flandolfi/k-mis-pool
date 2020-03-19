import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree


class Ordering:
    def __call__(self, data: Data):
        return data


class Random(Ordering):
    def __call__(self, data: Data):
        return torch.randperm(data.num_nodes)


class Canonical(Ordering):
    def __call__(self, data: Data):
        return torch.arange(data.num_nodes)


class Degree(Ordering):
    def __init__(self, descending=False):
        self.descending = (-1)**int(descending)

    def __call__(self, data: Data):
        return self.descending * degree(data.edge_index[0], data.num_nodes)


class MinDegree(Degree):
    def __init__(self):
        super(MinDegree, self).__init__(False)


class MaxDegree(Degree):
    def __init__(self):
        super(MaxDegree, self).__init__(True)
