import warnings

import fire
import numpy as np

import torch

from torch_geometric.data import DataLoader, Data, Dataset
from torch_geometric.datasets import ModelNet
from torch_geometric import transforms as T

import skorch
from skorch import NeuralNetClassifier
from skorch.callbacks import ProgressBar, Checkpoint, EpochScoring
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
    'optimizer': torch.optim.Adam,
    'device': device,
    'max_epochs': 999999999,
    'verbose': 1,
    'lr': 0.001,
    'batch_size': -1,
    'criterion': torch.nn.CrossEntropyLoss,
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
             train_split: float = 0.1,
             **net_kwargs):
    ds = ModelNet(root, '40', train=True,
                  transform=T.Compose([
                      T.NormalizeScale(),
                      T.SamplePoints(num=num_nodes)
                  ]))
    
    opts = dict(DEFAULT_NET_PARAMS)
    opts.update({
        'callbacks': [
            ('progress_bar', ProgressBar),
            ('train_acc', EpochScoring(
                'accuracy',
                name='train_acc',
                on_train=True,
                lower_is_better=False)),
            ('checkpoint', Checkpoint),
        ],
        'train_split': CVSplit(cv=train_split, stratified=True, random_state=42),
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

    net.fit(ds, ds.data.y)


if __name__ == "__main__":
    fire.Fire(modelnet)
