import logging

import torch
import pandas as pd
from tqdm import tqdm

from torch_geometric.typing import SparseTensor

from benchmark import graphs

from kmis import utils
from kmis import reduce

logging.disable(logging.WARNING)


def spectrum_approximation(num_eigenvalues: int = 10, max_k: int = 3,
                           matrix: str = 'adj', device: str = None,
                           store_results: str = None, **kwargs):
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    results = []

    for graph in tqdm(sorted(graphs.GRAPH_NAMES), leave=False):
        G = graphs.get_graph(graph)
        data = graphs.gsp2pyg(G).to(device)
        idx, val, n, m = data.edge_index, data.edge_attr, data.num_nodes, data.num_edges
        adj = SparseTensor.from_edge_index(idx, val, sparse_sizes=(n, n))
        largest = matrix == 'adj'

        if largest:
            mat = adj
        else:
            mat = utils.get_laplacian_matrix(adj, normalization=matrix)

        for k in tqdm(list(range(1, max_k + 1)), leave=False):
            k_mis = reduce.KMISCoarsening(k=k, sample_partition=False, **kwargs)
            c_mat, p_inv, mis = k_mis.get_coarsening_matrices(adj, data.pos)

            for partition in [False, True]:
                if partition:
                    c_mat = reduce.sample_partition_matrix(c_mat)

                c_norm = c_mat * (c_mat.sum(0).unsqueeze(0) ** -0.5)

                mat_redux = k_mis.coarsen(c_norm, mat)
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
                    'n': n,
                    'm': m,
                    'sample': partition,
                    'k': k,
                    'n_c': nc,
                    'm_c': mc,
                    'node_r': 1 - nc/n,
                    'edge_r': 1 - mc/m,
                    'eig_err': eig_err.item(),
                }

                if matrix == 'lap':
                    S = utils.get_incidence_matrix(adj).t()
                    l_inv = l ** -0.5
                    l_inv[0] = 0.
                    M = S @ c_norm @ c_norm.t() @ U @ torch.diag(l_inv)
                    ss_err = [torch.abs(torch.linalg.norm(M[:, :i], ord=2) - 1) for i in range(2, num_eigenvalues + 1)]
                    ss_err = sum(ss_err) / (len(ss_err) + 1)
                    metrics['subspace_err'] = ss_err.item()

                results.append(metrics)

    df = pd.DataFrame.from_records(results).set_index(['graph', 'n', 'm', 'k', 'sample'])

    if store_results is not None:
        df.to_json(store_results)

    return df
