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


def draw_mnist(show=True, save_fig: str = None, index: int = 0, size: int = 12, pool=False, k=1, **kwargs):
    mnist = MNIST("dataset/torchvision/", download=True)
    img = ToTensor()(mnist[0][0])[index]
    
    if pool:
        pool = torch.nn.AvgPool2d(kernel_size=2*k+1, stride=k+1, padding=k, ceil_mode=False)
        img = pool(img.unsqueeze(0)).squeeze(0).cpu().numpy()
        _, ax = plt.subplots(figsize=(size, size))
        ax.imshow(img, cmap='viridis', vmin=0, vmax=1)
        ax.set_axis_off()
    else:
        data = img2grid(img)
        k_mis = reduce.KMISCoarsening(k=k, **kwargs)
    
        (x, pos), adj, _, _, _ = k_mis((data.x, data.pos), data.edge_index, data.edge_attr)
        row, col, val = adj.coo()
        redux = Data(x=x, pos=pos, edge_index=torch.stack([row, col]), edge_attr=val)
        draw_data(redux, figsize=(size, size))
    
    if show:
        plt.show()
    
    if save_fig:
        plt.savefig(save_fig, format=save_fig.split('.')[-1], bbox_inches='tight')
