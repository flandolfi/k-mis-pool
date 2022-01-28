import logging
import os.path as osp

import torch
import h5py
import scipy.io as sio
from scipy.sparse import csr_matrix
import numpy as np

from torch_geometric.data import InMemoryDataset, Data, download_url
from torch_geometric.transforms import Compose, Distance, ToSparseTensor
from torch_geometric.typing import SparseTensor, Tensor, Tuple, Union


# Adapted from PyTorch Geometric (see https://pytorch-geometric.readthedocs.io/)
class SuiteSparseMatrixCollection(InMemoryDataset):
    r"""A suite of sparse matrix benchmarks known as the `Suite Sparse Matrix
    Collection <https://sparse.tamu.edu>`_ collected from a wide range of
    applications.

    Args:
        root (string): Root directory where the dataset should be saved.
        group (string): The group of the sparse matrix.
        name (string): The name of the sparse matrix.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://sparse.tamu.edu/mat/{}/{}.mat'

    def __init__(self, root, group, name, transform=None, pre_transform=None):
        self.group = group
        self.name = name
        super(SuiteSparseMatrixCollection,
              self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.group, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.group, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.name}.mat'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        url = self.url.format(self.group, self.name)
        download_url(url, self.raw_dir, log=True)

    def process(self):
        try:
            data = sio.loadmat(self.raw_paths[0])['Problem'][0][0]
            mat = data[2].tocsr().tocoo()
            pos = None
            
            for aux, aux_dtype in zip(data, data.dtype.names):
                if aux_dtype == 'aux':
                    for pos, pos_dtype in zip(aux, aux.dtype.names):
                        if pos_dtype == 'coord':
                            pos = torch.from_numpy(pos[0][0]).float()
                            break
                    break
        except NotImplementedError:
            with h5py.File(self.raw_paths[0], 'r') as file:
                data = np.array(file['Problem/A/data'])
                indices = np.array(file['Problem/A/ir'])
                indptr = np.array(file['Problem/A/jc'])
                mat = csr_matrix((data, indices, indptr)).tocoo()
                pos = None
                
                if 'Problem/aux/coord' in file:
                    pos = np.array(file['Problem/aux/coord'])
                    pos = torch.from_numpy(pos).float()

        row = torch.from_numpy(mat.row).to(torch.long)
        col = torch.from_numpy(mat.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = torch.from_numpy(mat.data).to(torch.float)
        if torch.all(edge_attr == 1.):
            edge_attr = None

        num_nodes = mat.shape[0]

        if pos is not None and pos.size(0) != num_nodes:
            assert pos.size(1) == num_nodes
            pos = pos.T

        data = Data(edge_index=edge_index, edge_attr=edge_attr,
                    num_nodes=num_nodes, pos=pos)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}(group={}, name={})'.format(self.__class__.__name__,
                                              self.group, self.name)


def load_graph(name: str = 'luxembourg_osm',
               group: str = 'DIMACS10',
               root: str = 'datasets/',
               device: str = 'cpu',
               return_coords: bool = False,
               add_distances: bool = False,
               logging_level: int = logging.INFO) -> Union[SparseTensor, Tuple[SparseTensor, Tensor]]:
    logging.basicConfig(level=logging_level)
    logging.info(f"Loading SuiteSparseMatrixCollection(name='{name}', group='{group}')...")
    transform = ToSparseTensor(attr='edge_attr')

    if name.endswith('_osm'):
        transform = Compose([Distance(False, cat=False), transform])

    data = SuiteSparseMatrixCollection(root=root, group=group, name=name,
                                       pre_transform=transform)[0]

    adj, pos, n, m = data.adj_t, data.pos, data.num_nodes, data.num_edges

    if not add_distances or not name.endswith('_osm'):
        adj.fill_value_(1.)

    adj.storage._value.squeeze_(-1)

    logging.info(f"Loaded graph of {n} nodes and {m} edges "
                 f"(density of {m/(n*n):.3g}).")

    if device != adj.device().type:
        logging.info(f"Moving adjacency matrix to {device}...")
        adj = adj.to(device)
        logging.info("Done.")

        if return_coords and pos is not None:
            logging.info(f"Moving coordinates matrix to {device}...")
            pos = pos.to(device)
            logging.info("Done.")

    if return_coords:
        return adj, pos

    return adj


def info(*args, **kwargs):
    graphs = {
        'SNAP': ['com-Youtube', 'com-LiveJournal', 'as-Skitter'],
        'DIMACS10': ['europe_osm', 'asia_osm', 'italy_osm',
                     'coPapersCiteseer', 'coPapersDBLP']
    }
    print('name,group,n,m,density,min_weight,max_weight')

    for group, names in graphs.items():
        for name in names:
            adj = load_graph(name, group, *args, **kwargs)
            val = adj.storage.value()
            min_weight = max_weight = 1.

            if val is not None:
                val = val[val > 0]
                min_weight = val.min().item()
                max_weight = val.max().item()

            print(f'{name},{group},{adj.size(0)},'
                  f'{adj.nnz()},{adj.density()},'
                  f'{min_weight},{max_weight}')
