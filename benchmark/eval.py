import warnings

import fire
import numpy as np

import torch

from torch_geometric.data import DataLoader, Data
from torch_geometric.datasets import ModelNet
from torch_geometric import transforms as T

import skorch
from skorch import NeuralNetClassifier
from skorch.callbacks import ProgressBar, Checkpoint, EpochScoring
from skorch.dataset import CVSplit

from benchmark.models import GNN


class SkorchDataset(torch.utils.data.Dataset):
    def __init__(self, X, y=None, transform=None):
        self.X = list(X)
        self.y = y
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
    'dataset': SkorchDataset,
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
    ds = ModelNet(root, '40', train=True)
    ds.data.face = None
    ds.slices.pop('face')
    
    opts = dict(DEFAULT_NET_PARAMS)
    opts.update({
        'callbacks': [
            ('progress_bar', ProgressBar),
            ('train_acc', EpochScoring(
                'accuracy',
                name='valid_acc',
                on_train=True,
                lower_is_better=False)),
            ('checkpoint', Checkpoint),
        ],
        'dataset__transform': T.FixedPoints(num=num_nodes, replace=False, allow_duplicates=False),
        'train_split': CVSplit(cv=0.1, stratified=True, random_state=42),
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

    net.fit(list(ds), ds.data.y)


if __name__ == "__main__":
    fire.Fire(modelnet)
