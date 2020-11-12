import torch

from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from torch_geometric.typing import Adj, Tensor, Tuple, OptTensor, Size
from torch_sparse import SparseTensor

from networkx.algorithms import centrality as nxc

from miss.pool import MISSPool
from miss.utils import geodesic_fps


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
        c_dict = self.centrality(G, *self.args, **self.kwargs)
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
    def __init__(self, max_nodes=4096, pool_ratio=1., aggr='mean', ordering="random",
                 add_self_loops=True, normalize=False, distances=False, kernel=None):
        super(MISSampling, self).__init__(pool_size=1, aggr=aggr, ordering=ordering,
                                          add_self_loops=add_self_loops, normalize=normalize,
                                          distances=distances, kernel=kernel)
        self.max_nodes = max_nodes
        self.pool_ratio = pool_ratio

    def forward(self, x: OptTensor, edge_index: Adj,
                edge_attr: OptTensor = None,
                pos: OptTensor = None,
                batch: OptTensor = None,
                size: Size = None) -> Tuple[OptTensor, Adj, OptTensor, OptTensor]:
        adj = self._get_adj(x, edge_index, edge_attr, pos, batch, size)
        perm = None if self.ordering is None else self.ordering(x, adj)
        mis, self.stride = geodesic_fps(adj, self.max_nodes, perm)
        self.pool_size = int(self.stride*self.pool_ratio)

        x, pos = self.pool(adj, mis, x, pos)
        adj = self.coarsen(adj, mis)

        if batch is not None:
            batch = batch[mis]

        return x, adj, pos, batch

    def __call__(self, data: Data):
        data.x, data.edge_index, data.pos, _ = self.forward(data.x, data.edge_index, data.edge_attr, data.pos)
        return data
