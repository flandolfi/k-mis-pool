import os.path as osp

import torch
import numpy as np

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import GNNBenchmarkDataset, ModelNet
from torch_geometric.transforms import Compose, FaceToEdge, NormalizeScale

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, make_scorer

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator


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


class SkorchDataset(torch.utils.data.Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

        self._len = len(X)

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def _evaluator_wrapper(evaluator):
    def _wrapper(y_true, y_pred):
        return evaluator({'y_true': y_true, 'y_pred': y_pred})[evaluator.eval_metric]

    return _wrapper


def get_dataset(name='MNIST', root='./dataset/'):
    if name.startswith('ModelNet'):
        pre_tr = Compose([NormalizeScale(), FaceToEdge(remove_faces=True)])
        train = ModelNet(osp.join(root, name), name=name[8:], train=True, pre_transform=pre_tr)
        test = ModelNet(osp.join(root, name), name=name[8:], train=False, pre_transform=pre_tr)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        y = train.data.y.numpy()
        x = np.zeros_like(y)
        tr_idx, val_idx = next(sss.split(x, y))

        val = CustomDataset([train[int(i)] for i in val_idx])
        train = CustomDataset([train[int(i)] for i in tr_idx])

        return (train, val, test), make_scorer(accuracy_score)

    if name in {'MNIST', 'CIFAR10'}:
        return ((GNNBenchmarkDataset(root, name, split)
                 for split in ['train', 'val', 'test']),
                make_scorer(accuracy_score))

    dataset = PygGraphPropPredDataset(name, root=root)
    split_idx = dataset.get_idx_split()

    return ((dataset[split_idx[split]]
             for split in ['train', 'valid', 'test']),
            make_scorer(_evaluator_wrapper(Evaluator(name))))


def merge_datasets(*datasets):
    splits = []
    n = 0

    for ds in datasets:
        splits.append(list(range(n, n + len(ds))))
        n += len(ds)

    return CustomDataset(sum(map(list, datasets), start=[])), splits
