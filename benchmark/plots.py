import math
from typing import List, Union

import torch
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

from torch_geometric.nn.pool import avg_pool_x
from torch_geometric.utils import grid
from torch_sparse import SparseTensor

from benchmark.datasets import load_graph
from kmis import KMISPooling

import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def plot_reductions(name: str = 'minnesota',
                    group: str = 'Gleich',
                    root: str = './datasets',
                    k: Union[int, List[int]] = 1,
                    scorer: str = 'constant',
                    ordering: str = 'div-k-sum',
                    aggregate: bool = False,
                    edge_width: float = 1,
                    fig_size: float = 2,
                    aspect: float = 1,
                    out_path: str = None):
    if out_path is not None:
        mpl.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })

    ks = k if isinstance(k, list) else [k]
    adj, coords = load_graph(name, group, root=root, device='cpu', return_coords=True)
    pos = coords.numpy()
    p_max, p_min = pos.max(0), pos.min(0)
    margin = 0.01*(p_max - p_min)
    x_lim, y_lim = tuple(zip(p_min - margin, p_max + margin))

    num_axes = len(ks) + 1
    cmap_width = 0.0666
    fig_size = (fig_size*aspect*num_axes + fig_size*aspect*cmap_width, fig_size)
    fig, axes = plt.subplots(1, num_axes + 1, figsize=fig_size,
                             gridspec_kw=dict(width_ratios=[1.]*num_axes + [cmap_width],
                                              left=0, right=0.95, wspace=0.05))

    G = nx.from_scipy_sparse_matrix(adj.to_scipy())
    apsp = dict(nx.all_pairs_shortest_path_length(G))

    cmap = sns.diverging_palette(250, 15, center="dark", s=100, as_cmap=True)
    sm = plt.cm.ScalarMappable(mpl.colors.Normalize(vmin=0, vmax=1), cmap)

    pos = dict(enumerate(pos))
    nx.draw_networkx(G, pos=pos, node_size=0, width=edge_width, ax=axes[0],
                     edge_color=sm.to_rgba(0.5*np.ones(G.number_of_edges())),
                     with_labels=False)
    axes[0].axis('off')
    axes[0].set_title('$k = 0$')
    axes[0].set_xlim(x_lim)
    axes[0].set_ylim(y_lim)

    for k, ax in zip(ks, axes[1:-1]):
        pool = KMISPooling(k=k, scorer=scorer, ordering=ordering)
        x = torch.ones((adj.size(0), 1))
        _, adj_redux, _, _, cluster, mis, _ = pool.forward(x, adj)
        H = nx.from_scipy_sparse_matrix(adj_redux.to_scipy())

        if aggregate:
            pos_redux, _ = avg_pool_x(cluster, coords, batch=torch.zeros_like(cluster).long())
            pos_redux = dict(enumerate(pos_redux.numpy()))
        else:
            pos_redux = dict(enumerate(coords[mis].numpy()))

        ids = torch.nonzero(mis).view(-1)
        row, col, _ = adj_redux.coo()

        edge_color = []

        for u, v in zip(ids[row].tolist(), ids[col].tolist()):
            ratio = (apsp[u][v] - k - 1)/k
            edge_color.append(ratio)

        nx.draw_networkx(H, pos=pos_redux, node_size=0, width=edge_width,
                         ax=ax, edge_color=sm.to_rgba(edge_color), with_labels=False)
        ax.axis('off')
        ax.set_title(f'$k = {k}$')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    cbar = plt.colorbar(sm, cax=axes[-1], ticks=[0, 1])
    cbar.ax.set_yticklabels(['$k + 1$', '$2k + 1$'])

    if out_path is not None:
        plt.tight_layout()
        fig.savefig(out_path, format='pgf', bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()


def plot_mnist(root: str = './datasets/',
               index: int = 0,
               k: Union[int, List[int]] = 1,
               scorer: str = 'lightness',
               ordering: str = 'div-k-degree',
               average: bool = True,
               fig_size: float = 12,
               node_size: float = 1000,
               save_fig: str = None):
    mnist = MNIST(root, download=True)
    img = ToTensor()(mnist[0][0])[index]
    img = img.reshape(-1, 1)
    side = 28
    (row, col), pos = grid(side, side)
    pos[:, 1] = pos[:, 1] - side + 1
    n = side*side

    l_min, l_max = img.min().item(), img.max().item()

    adj = SparseTensor(row=row, col=col, sparse_sizes=(n, n))
    x = torch.cat([img, pos], dim=-1)

    if scorer == 'lightness':
        def scorer(x, edge_index, edge_attr):
            return x, x[:, 0]

    pool = KMISPooling(k=k, scorer=scorer, ordering=ordering, reduce_x='mean')
    x, adj, _, _, _, mis, _ = pool.forward(x, adj)
    pos = x[:, 1:]

    if average:
        img = x[:, :1]
    else:
        img = img[mis]

    node_size = (k + 1)*node_size
    width = 2

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.axis('off')
    nx.draw_networkx(nx.from_scipy_sparse_matrix(adj.to_scipy()),
                     vmin=l_min, vmax=l_max,
                     node_shape='s',
                     node_color=img.cpu().numpy(),
                     cmap='viridis',
                     edge_color='black',
                     width=width,
                     node_size=node_size,
                     pos=dict(enumerate(pos.clone().cpu().numpy())),
                     with_labels=False,
                     ax=ax)

    margin = 1.5
    xlim, ylim = tuple(zip(pos.min(dim=0)[0] - margin, pos.max(dim=0)[0] + margin))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
    else:
        plt.show()

