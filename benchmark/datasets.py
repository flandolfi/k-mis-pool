import logging
import os
import os.path as osp
from typing import Callable, List, Optional, Tuple, Union

import torch
import h5py
import scipy.io as sio
from scipy.sparse import csr_matrix
import numpy as np

from torch_geometric.transforms import Compose, Distance, ToSparseTensor
from torch_geometric.typing import SparseTensor, Tensor

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_tar,
    extract_zip,
)


class MalNetTiny(InMemoryDataset):
    r"""The MalNet Tiny dataset from the
    `"A Large-Scale Database for Graph Representation Learning"
    <https://openreview.net/pdf?id=1xDTDk3XPW>`_ paper.
    :class:`MalNetTiny` contains 5,000 malicious and benign software function
    call graphs across 5 different types. Each graph contains at most 5k nodes.

    Args:
        root (string): Root directory where the dataset should be saved.
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"trainval"`, loads the training and validation dataset.
            If :obj:`"test"`, loads the test dataset.
            If :obj:`None`, loads the entire dataset.
            (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    data_url = 'http://malnet.cc.gatech.edu/' \
               'graph-data/malnet-graphs-tiny.tar.gz'
    split_url = 'http://malnet.cc.gatech.edu/split-info/split_info_tiny.zip'
    splits = ['train', 'val', 'test']

    def __init__(self, root: str, split: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        assert split in {*self.splits, 'trainval', None},   \
            f'Split "{split}" found, but expected either '  \
            f'"train", "val", "trainval", "test" or None'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices, split_slices = torch.load(
            self.processed_paths[0])

        if split is not None:
            indices = self.indices()

            if split == 'trainval':
                self._indices = indices[:split_slices['val'][1]]
            else:
                self._indices = indices[slice(*split_slices[split])]

    @property
    def raw_file_names(self) -> List[str]:
        return [
            osp.join('split_info_tiny', 'type', f'{split}.txt')
            for split in self.splits
        ] + ['malnet-graphs-tiny']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        path = download_url(self.data_url, self.raw_dir)
        extract_tar(path, self.raw_dir)
        os.unlink(path)

        path = download_url(self.split_url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        data_list = []
        split_idxs = [0]
        base_path = self.raw_paths[3]
        y_map = {}

        for path in self.raw_paths[:3]:
            with open(path, 'r') as f:
                filenames = f.readlines()

            split_idxs.append(split_idxs[-1] + len(filenames))

            for filename in filenames:
                malware_type = filename.split('/')[0]
                y = y_map.setdefault(malware_type, len(y_map))
                path = osp.join(base_path, filename[:-1] + '.edgelist')

                with open(path, 'r') as f:
                    edges = f.read().split('\n')[5:-1]

                edge_index = [[int(s) for s in edge.split()] for edge in edges]
                edge_index = torch.tensor(edge_index).t().contiguous()
                num_nodes = int(edge_index.max()) + 1
                data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes)
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        slices = dict(zip(self.splits, zip(split_idxs[:-1], split_idxs[1:])))
        torch.save((*self.collate(data_list), slices), self.processed_paths[0])


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
                        if pos_dtype in {'coord', 'nodename'}:
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
