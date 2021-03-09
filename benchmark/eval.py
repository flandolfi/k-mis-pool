import warnings

import fire
import numpy as np

import torch

from torch_geometric.data import DataLoader, Data, Dataset
from torch_geometric.datasets import ModelNet
from torch_geometric import transforms as T

import skorch
from skorch import NeuralNetClassifier
from skorch.callbacks import ProgressBar, Checkpoint, EpochScoring, LRScheduler
from skorch.dataset import CVSplit

from benchmark.models import GNN


def _to_tensor_wrapper(func):
    to_tensor = func

    def wrapper(X, device, allow_sparse=False):
        if isinstance(X, Data):
            return X.to(device)

        return to_tensor(X, device, allow_sparse)

    return wrapper


def _get_item_wrapper(func):
    wrapped = func

    def wrapper(dataset, idx):
        if isinstance(idx, np.int64):
            idx = int(idx)

        return wrapped(dataset, idx)

    return wrapper


def _unpack_data_wrapper(func):
    wrapped = func

    def wrapper(data):
        if isinstance(data, Data):
            return data, data.y

        return wrapped(data)

    return wrapper


torch.manual_seed(42)
np.random.seed(42)
device = 'cpu'

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    device = 'cuda'

skorch.net.unpack_data = _unpack_data_wrapper(skorch.net.unpack_data)
skorch.net.to_tensor = _to_tensor_wrapper(skorch.net.to_tensor)
Dataset.__getitem__ = _get_item_wrapper(Dataset.__getitem__)

warnings.filterwarnings("ignore", category=UserWarning)


DEFAULT_NET_PARAMS = {
    'module__pool_size': 1,
    'module__ordering': 'random',
    'device': device,
    'verbose': 1,
    'lr': 0.001,
    'batch_size': 32,
    'max_epochs': 9999999999,
    'criterion': torch.nn.CrossEntropyLoss,
    'iterator_train': DataLoader,
    'iterator_valid': DataLoader,
    'iterator_train__shuffle': True,
    'iterator_valid__shuffle': False,
    'iterator_train__drop_last': True,
    'iterator_valid__drop_last': False
}


def modelnet(num_points: int = 1024,
             train_split: float = 0.1,
             optimizer: str = 'Adam',
             dataset_path: str = './dataset/ModelNet40/',
             **net_kwargs):
    ds = ModelNet(dataset_path, '40', train=True,
                  pre_transform=T.NormalizeScale(),
                  transform=T.SamplePoints(num=num_points))
    
    opts = dict(DEFAULT_NET_PARAMS)
    opts.update({
        'optimizer': getattr(torch.optim, optimizer),
        'optimizer__weight_decay': 0.0001,
        'train_split': CVSplit(cv=train_split, stratified=True, random_state=42),
        'callbacks': [
            ('progress_bar', ProgressBar),
            ('valid_bal', EpochScoring),
            ('lr_scheduler', LRScheduler),
            ('checkpoint', Checkpoint),
        ],
        'callbacks__checkpoint__monitor': 'valid_acc_best',
        'callbacks__checkpoint__f_params': 'params.pt',
        'callbacks__checkpoint__f_optimizer': None,
        'callbacks__checkpoint__f_criterion': None,
        'callbacks__checkpoint__f_history': 'history.json',
        'callbacks__checkpoint__f_pickle': None,
        'callbacks__lr_scheduler__policy': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        'callbacks__lr_scheduler__T_0': 10,
        'callbacks__lr_scheduler__T_mult': 2,
        'callbacks__valid_bal__scoring': 'balanced_accuracy',
        'callbacks__valid_bal__name': 'valid_bal',
        'callbacks__valid_bal__on_train': False,
        'callbacks__valid_bal__lower_is_better': False
    })
    opts.update(net_kwargs)
    
    NeuralNetClassifier(
        module=GNN,
        module__dataset=ds,
        **opts
    ).fit(ds, ds.data.y)


if __name__ == "__main__":
    fire.Fire(modelnet)
