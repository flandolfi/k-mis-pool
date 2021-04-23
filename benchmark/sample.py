import logging

import torch

from tqdm import tqdm

from torch_geometric.typing import SparseTensor

from benchmark import graphs

from kmis.utils import normalize_dim
from kmis import reduce

logging.disable(logging.WARNING)


def sample(graph: str = "airfoil", matrix: str = "pi", iterations: int = 10000, device: str = None, **kwargs):
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    k_mis = reduce.KMISCoarsening(sample_partition=False, **kwargs)

    G = graphs.get_graph(graph)
    data = graphs.gsp2pyg(G).to(device)
    idx, val, n = data.edge_index, data.edge_attr, data.num_nodes
    adj = SparseTensor.from_edge_index(idx, val, sparse_sizes=(n, n))
    c_mat, p_inv, _ = k_mis.get_coarsening_matrices(adj, data.pos)

    if matrix == 'inv':
        target = p_inv
    elif matrix == 'adj':
        target = k_mis.coarsen(c_mat, adj).to_dense()
    elif matrix == 'pi':
        target = c_mat.to_dense() @ p_inv
    elif matrix == 'identity':
        target = p_inv @ c_mat.to_dense()
    else:
        target = c_mat.to_dense()

    m_approx = torch.zeros_like(target)
    p_bar = tqdm(list(range(1, iterations + 1)), leave=True)
    mask = target != 0
    i = 1

    try:
        for i in p_bar:
            p_mat = reduce.sample_partition_matrix(c_mat)

            if matrix == 'inv':
                m_approx += normalize_dim(p_mat.t()).to_dense()
            elif matrix == 'adj':
                m_approx += k_mis.coarsen(p_mat, adj).to_dense()
            elif matrix == 'pi':
                m_approx += p_mat.to_dense() @ normalize_dim(p_mat.t()).to_dense()
            elif matrix == 'identity':
                m_approx += normalize_dim(p_mat.t()).to_dense() @ p_mat.to_dense()
            else:
                m_approx += p_mat.to_dense()

            p_bar.set_postfix(mae=torch.abs(target - m_approx/i)[mask].mean().item())
    except KeyboardInterrupt:
        pass

    print('Target:')
    print(target[mask].cpu().numpy())
    print('Approx:')
    print((m_approx/i)[mask].cpu().numpy())
    print('Error:')
    print((target - m_approx/i)[mask].cpu().numpy())

    if matrix == 'inv':
        print('Identity:')
        print((m_approx @ c_mat.to_dense()).cpu().numpy())
