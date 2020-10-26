import torch
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from networkx.algorithms import centrality as nxc


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
