import torch
import numpy as np

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import GNNBenchmarkDataset, ZINC, ModelNet
from torch_geometric.transforms import Compose, FaceToEdge

from sklearn.model_selection import StratifiedShuffleSplit
from skorch.dataset import Dataset


class OneHotEncoding:
    def __init__(self, num_classes, on_key='x'):
        self.num_classes = num_classes
        self.on_key = on_key

    def __call__(self, data: Data):
        val = data[self.on_key]
        eye = torch.eye(self.num_classes, dtype=torch.float, device=val.device)
        data[self.on_key] = eye[val.long()]

        return data


class CustomDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(CustomDataset, self).__init__("")
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass


def get_dataset(name='MNIST', root='data/'):
    if name == 'ZINC':
        transform = Compose([
            OneHotEncoding(28, 'x'),
            OneHotEncoding(4, 'edge_attr')
        ])
        return (ZINC(root, subset=True, split=split, pre_transform=transform)
                for split in ['train', 'val', 'test'])

    if name.startswith('ModelNet'):
        train = ModelNet(root, name=name[8:], train=True, pre_transform=FaceToEdge())
        test = ModelNet(root, name=name[8:], train=False, pre_transform=FaceToEdge())

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        y = train.data.y.numpy()
        x = np.zeros_like(y)
        tr_idx, val_idx = next(sss.split(x, y))

        val = train[val_idx]
        train = train[tr_idx]

        return train, val, test

    return (GNNBenchmarkDataset(root, name, split)
            for split in ['train', 'val', 'test'])


def merge_datasets(*datasets):
    Xs, Ys = [], []
    n = 0

    for ds in datasets:
        Xs.append(np.arange(n, n + len(ds)))
        Ys.append(ds.data.y.numpy())
        n += len(ds)

    return (CustomDataset(sum(map(list, datasets))),
            *(Dataset(X, y) for X, y in zip(Xs, Ys)))

