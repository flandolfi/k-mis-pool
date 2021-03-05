import math

import torch

from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from torch_geometric.typing import Adj, Tuple, OptTensor, Size, Union, Tensor
from torch_sparse import SparseTensor

from networkx.algorithms import centrality as nxc

from miss.pool import MISSPool
from miss.utils import maximal_independent_sample, maximal_k_independent_set


class Permute(object):
    def __init__(self, centrality='betweenness_centrality', descending=True, *args, **kwargs):
        if isinstance(centrality, str):
            centrality = getattr(nxc, centrality)

        self.centrality = centrality
        self.descending = descending
        self.args = args
        self.kwargs = kwargs

    def __call__(self, data: Data):
        G = to_networkx(data, edge_attrs=None if data.edge_attr is None else ["edge_attr"],
                        to_undirected=True)
        c_dict = self.centrality(G, *self.args, **self.kwargs)  # noqa
        perm = torch.argsort(torch.FloatTensor(list(c_dict.values())),
                             dim=0, descending=self.descending)
        adj = SparseTensor.from_edge_index(data.edge_index, data.edge_attr,
                                           sparse_sizes=(data.num_nodes, data.num_nodes),
                                           is_sorted=True)
        row, col, val = adj.permute(perm).coo()
        kwargs = {
            "edge_index": torch.stack([row, col]),
            "edge_attr": val
        }

        for key, val in data("x", "y", "pos"):
            if val is not None and val.size(0) == data.num_nodes:
                kwargs[key] = val[perm]
            else:
                kwargs[key] = val

        return Data.from_dict(kwargs)


class MISSampling(MISSPool):
    def __init__(self, max_nodes=4096, pool_ratio=1., **kwargs):
        super(MISSampling, self).__init__(pool_size=1, stride=1, ordering=None, **kwargs)
        
        self.max_nodes = max_nodes
        self.pool_ratio = pool_ratio

    def _get_mis(self, adj: Adj, *xs: OptTensor) -> Tensor:
        mis, self.stride = maximal_independent_sample(adj, self.max_nodes)
        self.pool_size = int(self.stride*self.pool_ratio)

        return mis

    def __call__(self, data: Data) -> Data:
        data.edge_index, data.x, data.pos, _ = self.forward(data.edge_index, data.edge_attr, data.x, data.pos)
        return data
