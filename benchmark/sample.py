import logging

import torch

from tqdm import tqdm

from torch_geometric.typing import SparseTensor

from benchmark import graphs

from kmis.utils import normalize_dim, get_laplacian_matrix
from kmis import reduce

logging.disable(logging.WARNING)


def sample(graph: str = "airfoil", matrix: str = "lift", iterations: int = 10000, device: str = None, **kwargs):
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
    c_mat, _ = k_mis.get_coarsening_matrix(adj, data.pos)
    l_mat = normalize_dim(c_mat.t(), -1)

    if matrix == 'c':
        target = c_mat.to_dense()
    elif matrix == 'lift':
        target = l_mat.to_dense()
    elif matrix == 'adj':
        target = k_mis.coarsen(c_mat, adj).to_dense()
    elif matrix == 'pi':
        target = c_mat.to_dense() @ l_mat.to_dense()
    else:
        lap = get_laplacian_matrix(adj, normalization=matrix)
        target = k_mis.coarsen(c_mat, lap).to_dense()

    m_approx = torch.zeros_like(target)
    p_bar = tqdm(list(range(1, iterations + 1)), leave=True)
    k_mis.sample_partition = 'on_train'
    k_mis.training = False
    i = 1

    try:
        for i in p_bar:
            p_mat = reduce.sample_partition_matrix(c_mat)

            if matrix == 'c':
                m_approx += p_mat.to_dense()
            elif matrix == 'lift':
                m_approx += normalize_dim(p_mat.t()).to_dense()
            elif matrix == 'adj':
                m_approx += k_mis.coarsen(p_mat, adj).to_dense()
            elif matrix == 'pi':
                m_approx += p_mat.to_dense() @ normalize_dim(p_mat.t()).to_dense()
            else:
                m_approx += k_mis.coarsen(p_mat, lap).to_dense()  # noqa

            p_bar.set_postfix(mae=torch.abs(target - m_approx/i).mean().item())
    except KeyboardInterrupt:
        pass

    print('Target: \n%s' % target.cpu().numpy())
    print('Approx: \n%s' % (m_approx/i).cpu().numpy())
    print('Error: \n%s' % (target - m_approx/i).cpu().numpy())
