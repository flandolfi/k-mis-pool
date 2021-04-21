import logging

import torch
import numpy as np

from scipy import sparse
from tqdm import tqdm
from tabulate import tabulate

from torch_geometric.typing import SparseTensor
from graph_coarsening import coarsening_quality

from benchmark import graphs

from kmis.utils import normalize_dim
from kmis import reduce

logging.disable(logging.WARNING)


def approx(num_eigenvalues: int = 10, max_k: int = 3, **kwargs):
    results = []

    for graph in tqdm(sorted(graphs.GRAPH_NAMES), leave=False):
        G = graphs.get_graph(graph)
        G.compute_fourier_basis()

        data = graphs.gsp2pyg(G)
        idx, val, n = data.edge_index, data.edge_attr, data.num_nodes
        adj = SparseTensor.from_edge_index(idx, val, sparse_sizes=(n, n))

        for k in tqdm(list(range(1, max_k + 1)), leave=False):
            k_mis = reduce.KMISCoarsening(k=k, **kwargs)
            c_mat, _, _ = k_mis.get_coarsening_matrices(adj, data.pos)

            c_mat = normalize_dim(c_mat, 0)
            c_mat = c_mat.set_value(torch.sqrt(c_mat.storage.value()),
                                    layout="coo")

            row, col, val = c_mat.coo()
            C = sparse.coo_matrix((val.numpy(), (row.numpy(), col.numpy())),
                                  shape=c_mat.sparse_sizes())

            metrics = {'graph': graph, 'k': k}
            metrics.update(coarsening_quality(G, C.T, kmax=num_eigenvalues))
            metrics.pop('angle_matrix')
            metrics['graph'] = graph

            for key, val in metrics.items():
                if isinstance(val, np.ndarray):
                    metrics[key] = val.mean()

            results.append(metrics)

    print(tabulate(results, headers='keys'))
