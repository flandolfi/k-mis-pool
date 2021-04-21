import math

import torch

from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, grid

import networkx as nx
import matplotlib.pyplot as plt

from kmis import reduce


def draw_data(data, **kwargs):
    fig, ax = plt.subplots(**kwargs)
    pos = dict(enumerate(data.pos.cpu().numpy()))

    nx.draw(to_networkx(data, to_undirected=True),
            vmin=0, vmax=1,
            node_shape='s',
            node_color=data.x.clone().cpu().numpy(),
            cmap='viridis',
            edge_color='black',
            node_size=math.sqrt(28*28/data.num_nodes)*500,
            pos=pos,
            ax=ax)

    return ax


def img2grid(img: torch.Tensor):
    shape = img.shape
    img = img.reshape(-1, 1)
    idx, pos = grid(*shape)

    return Data(edge_index=idx, pos=pos, x=img)


def draw_mnist(**kwargs):
    mnist = MNIST("data/", download=True)
    img = ToTensor()(mnist[0][0])[0]
    data = img2grid(img)
    k_mis = reduce.KMISCoarsening(**kwargs)

    (x, pos), adj, _, _, _, _ = k_mis((data.x, data.pos), data.edge_index, data.edge_attr)
    row, col, val = adj.coo()
    redux = Data(x=x, pos=pos, edge_index=torch.stack([row, col]), edge_attr=val)
    draw_data(redux, figsize=(12, 12))
    plt.show()
