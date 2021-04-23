import logging

import torch
import numpy as np

from scipy import sparse
from tqdm import tqdm
from tabulate import tabulate

from torch_geometric.typing import SparseTensor
from graph_coarsening import coarsening_quality

from benchmark import graphs

from kmis import utils
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

            c_mat = utils.normalize_dim(c_mat, 0)
            c_mat = c_mat.set_value(torch.sqrt(c_mat.storage.value()),
                                    layout="coo")

            row, col, val = c_mat.coo()
            C = sparse.coo_matrix((val.numpy(), (row.numpy(), col.numpy())),
                                  shape=c_mat.sparse_sizes())

            metrics = {'graph': graph, 'k': k}
            metrics.update(coarsening_quality(G, C.T, kmax=num_eigenvalues))
            metrics.pop('angle_matrix')

            for key, val in metrics.items():
                if isinstance(val, np.ndarray):
                    metrics[key] = val.mean()

            results.append(metrics)

    print(tabulate(results, headers='keys'))


def spectrum_approximation(num_eigenvalues: int = 10, max_k: int = 3,
                           matrix: str = 'adj', device: str = None, **kwargs):
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    results = []

    for graph in tqdm(sorted(graphs.GRAPH_NAMES), leave=False):
        G = graphs.get_graph(graph)
        data = graphs.gsp2pyg(G).to(device)
        idx, val, n = data.edge_index, data.edge_attr, data.num_nodes
        adj = SparseTensor.from_edge_index(idx, val, sparse_sizes=(n, n))
        largest = matrix == 'adj'

        if largest:
            mat = adj
        else:
            mat = utils.get_laplacian_matrix(adj, normalization=matrix)

        for k in tqdm(list(range(1, max_k + 1)), leave=False):
            k_mis = reduce.KMISCoarsening(k=k, **kwargs)
            c_mat, p_inv, mis = k_mis.get_coarsening_matrices(adj, data.pos)

            c_mat = utils.normalize_dim(c_mat, 0)
            c_mat = c_mat.set_value(torch.sqrt(c_mat.storage.value()), layout="coo")

            mat_redux = k_mis.coarsen(c_mat, mat)
            nc = mat_redux.size(0)
            mc = (mat_redux.nnz() - nc)//2

            l, U = torch.lobpcg(mat.to_torch_sparse_coo_tensor(),
                                k=num_eigenvalues, largest=largest, tol=1e-3)

            if nc < 3 * num_eigenvalues:
                dense_mat_redux = mat_redux.to_dense()
                lc, _ = torch.eig(dense_mat_redux)
                lc, _ = torch.sort(lc.T[0], descending=largest)
                lc = lc[:num_eigenvalues]
            else:
                lc, _ = torch.lobpcg(mat_redux.to_torch_sparse_coo_tensor(),
                                     k=num_eigenvalues, largest=largest, tol=1e-3)

            eig_err = torch.where(l == 0, torch.zeros_like(l), torch.abs(l - lc)/l)
            eig_err = eig_err.mean()

            metrics = {
                'graph': graph,
                'k': k,
                'n': nc,
                'm': mc,
                'r': 1 - nc/n,
                'eig_err': eig_err,
            }

            if matrix == 'lap':
                S = utils.get_incidence_matrix(adj).t()
                l_inv = l ** -0.5
                l_inv[0] = 0.
                M = S @ c_mat.fill_value(1.) @ p_inv @ U @ torch.diag(l_inv)
                ss_err = [torch.abs(torch.linalg.norm(M[:, :i], ord=2) - 1) for i in range(2, num_eigenvalues + 1)]
                ss_err = sum(ss_err)/(len(ss_err) + 1)
                metrics['ss_err'] = ss_err

            results.append(metrics)

    print(tabulate(results, headers='keys'))
