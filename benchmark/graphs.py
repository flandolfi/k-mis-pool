import torch

from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix

from graph_coarsening import graph_lib
from pygsp import graphs

GRAPH_NAMES = {"airfoil", "bunny", "minnesota", "yeast"}


def get_graph(name: str, **kwargs):
    if name not in GRAPH_NAMES:
        name = ''.join(s.title() for s in name.split('-'))
        return getattr(graphs, name)(**kwargs)

    return graph_lib.real(-1, name, **kwargs)


def gsp2pyg(G: graphs.Graph):
    index, val = from_scipy_sparse_matrix(G.W)
    data = Data(num_nodes=G.N, edge_index=index, edge_attr=val.float())

    if G.coords is not None:
        data.pos = torch.FloatTensor(G.coords)

    return data
