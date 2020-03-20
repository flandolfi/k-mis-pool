import numpy as np
from os import path

import torch
from torch.utils.data._utils.collate import default_collate

from torch_geometric.data import Data, Batch, InMemoryDataset, Dataset, download_url
from torch_geometric.transforms import ToDense


class CustomDataset(InMemoryDataset):
    """Create a dataset from a `torch_geometric.Data` list.
    
    Args:
        data_list (list): List of graphs.
    """
    def __init__(self, data_list):
        super(CustomDataset, self).__init__("")
        self.data, self.slices = self.collate(data_list)
    
    def _download(self):
        pass

    def _process(self):
        pass


class DenseDataset(Dataset):
    """Dense Graphs Dataset.
    
    Args:
        data_list (list): list of graphs.
    """
    def __init__(self, data_list):
        super(DenseDataset, self).__init__("")

        self.data = Batch()
        self.max_nodes = max([data.num_nodes for data in data_list])
        to_dense = ToDense(self.max_nodes)
        dense_list = [to_dense(data) for data in data_list]

        for key in dense_list[0].keys:
            self.data[key] = default_collate([d[key] for d in dense_list])

    def __len__(self):
        if self.data.x is not None:
            return self.data.x.size(0)

        if 'adj' in self.data:
            return self.data.adj.size(0)

        return 0

    def get(self, idx):
        mask = self.data.mask[idx]
        max_nodes = mask.type(torch.uint8).argmax(-1).max().item() + 1
        out = Batch()

        for key, item in self.data('x', 'pos', 'mask'):
            out[key] = item[idx, :max_nodes]

        out.adj = self.data.adj[idx, :max_nodes, :max_nodes]
        
        if 'y' in self.data:
            out.y = self.data.y[idx]

        return out

    def _download(self):
        pass

    def _process(self):
        pass


class NDPDataset(InMemoryDataset):
    """The synthetic dataset from `"Hierarchical Representation Learning in 
    Graph Neural Networks with Node Decimation Pooling"
    <https://arxiv.org/abs/1910.11436>`_ paper.

    Args:
        root (string): Root directory where the dataset should be saved.
        split (string): If `"train"`, loads the training dataset.
            If `"val"`, loads the validation dataset.
            If `"test"`, loads the test dataset. Defaults to `"train"`.
        easy (bool, optional): If `True`, use the easy version of the dataset.
            Defaults to `True`.
        small (bool, optional): If `True`, use the small version of the
            dataset. Defaults to `True`.
        transform (callable, optional): A function/transform that takes in an
            `torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            Defaults to `None`.
        pre_transform (callable, optional): A function/transform that takes in
            an `torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. Defaults to `None`.
        pre_filter (callable, optional): A function that takes in an
            `torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. Defaults to `None`.
    """
    base_url = ('http://github.com/FilippoMB/'
                'Benchmark_dataset_for_graph_classification/'
                'raw/master/datasets/')
    
    def __init__(self, root, split='train', easy=True, small=True, transform=None, pre_transform=None, pre_filter=None):
        self.file_name = ('easy' if easy else 'hard') + ('_small' if small else '')
        self.split = split.lower()

        assert self.split in {'train', 'val', 'test'}

        if self.split != 'val':
            self.split = self.split[:2]
        
        super(NDPDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return '{}.npz'.format(self.file_name)
    
    @property
    def processed_file_names(self):
        return '{}.pt'.format(self.file_name)

    def download(self):
        download_url('{}{}.npz'.format(self.base_url, self.file_name), self.raw_dir)

    def process(self):
        npz = np.load(path.join(self.raw_dir, self.raw_file_names), allow_pickle=True)
        raw_data = (npz['{}_{}'.format(self.split, key)] for key in ['feat', 'adj', 'class']) 
        data_list = [Data(x=torch.from_numpy(x).float(),
                          edge_index=torch.from_numpy(np.stack(adj.nonzero())).long(),
                          y=torch.from_numpy(y.nonzero()[0]).long()) for x, adj, y in zip(*raw_data)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])
