import warnings

import fire
import numpy as np

import torch

from torch_geometric.data import DataLoader, Data
from torch_geometric.datasets import ModelNet
from torch_geometric import transforms as T

import skorch
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import ProgressBar, Checkpoint
from skorch.dataset import Dataset

from sklearn.model_selection import StratifiedShuffleSplit

from benchmark.models import GNN


class SkorchDataset(Dataset):
    def __init__(self, X, y=None, transform=None):
        super(SkorchDataset, self).__init__(X, y)
        self.X = list(X)
        self.y = y or X.data.y
        self.transform = transform
        
        self._len = len(X)

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        return self.transform(self.X[i]), self.y[i]


def _to_tensor_wrapper(func):
    to_tensor = func

    def wrapper(X, device, allow_sparse=False):
        if isinstance(X, Data):
            return X.to(device)

        return to_tensor(X, device, allow_sparse)

    return wrapper


torch.manual_seed(42)
np.random.seed(42)
device = 'cpu'

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    device = 'cuda'

skorch.net.to_tensor = _to_tensor_wrapper(skorch.net.to_tensor)
warnings.filterwarnings("ignore", category=UserWarning)


DEFAULT_NET_PARAMS = {
    'module__pool_size': 1,
    'module__ordering': 'random',
    'optimizer': torch.optim.Adam,
    'device': device,
    'max_epochs': 999999999,
    'verbose': 1,
    'lr': 0.001,
    'batch_size': -1,
    'iterator_train': DataLoader,
    'iterator_valid': DataLoader,
    'iterator_train__shuffle': True,
    'iterator_valid__shuffle': False,
    'iterator_train__drop_last': True,
    'iterator_valid__drop_last': False
}


def modelnet(root: str = './dataset/ModelNet40/',
             num_nodes: int = 4096,
             model_path: str = None,
             history_path: str = None,
             **net_kwargs):
    def pre_transform(data):
        return Data(x=data.x, pos=data.pos)
    
    ds = ModelNet(root, '40', train=True, pre_transform=pre_transform)
    transform = T.FixedPoints(num=num_nodes, replace=False, allow_duplicates=False)
    
    sss = StratifiedShuffleSplit(1, test_size=0.1, random_state=42)
    y = ds.data.y.numpy()
    idx_tr, idx_val = next(sss.split(y, y))
    ds_train, ds_val = ds[list(idx_tr)], ds[list(idx_val)]

    opts = dict(DEFAULT_NET_PARAMS)
    opts.update({
        'train_split': predefined_split(SkorchDataset(ds_val, transform=transform)),
        'callbacks': [
            ('progress_bar', ProgressBar),
            ('checkpoint', Checkpoint),
        ],
        'callbacks__checkpoint__monitor': 'valid_acc_best',
        'callbacks__checkpoint__f_params': model_path,
        'callbacks__checkpoint__f_optimizer': None,
        'callbacks__checkpoint__f_criterion': None,
        'callbacks__checkpoint__f_history': history_path,
        'callbacks__checkpoint__f_pickle': None,
    })
    opts.update(net_kwargs)
    net = NeuralNetClassifier(
        module=GNN,
        module__dataset=ds,
        **opts
    )

    net.fit(SkorchDataset(ds_train, transform=transform), None)


if __name__ == "__main__":
    fire.Fire(modelnet)
