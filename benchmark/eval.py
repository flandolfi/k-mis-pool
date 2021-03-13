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

from sklearn.utils import compute_class_weight

from benchmark import models


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
    'verbose': 1,
    'batch_size': 8,
    'module__pool_size': 1,
    'module__ordering': 'random',
    'device': device,
    'iterator_train': DataLoader,
    'iterator_valid': DataLoader,
    'iterator_train__shuffle': True,
    'iterator_valid__shuffle': False,
    'iterator_train__drop_last': True,
    'iterator_valid__drop_last': False,
    'iterator_train__num_workers': 8,
    'iterator_valid__num_workers': 8,
    'iterator_train__persistent_workers': True,
    'iterator_valid__persistent_workers': True,
}


def train(num_points: int = 1024,
          train_split: float = 0.2,
          model: str = 'PointNet',
          optimizer: str = 'Adam',
          weighted_loss: bool = False,
          cosine_annealing: bool = False,
          dataset_path: str = './dataset/ModelNet40/',
          params_path: str = 'params.pt',
          history_path: str = None,
          **net_kwargs):
    ds = ModelNet(dataset_path, '40', train=True,
                  pre_transform=T.NormalizeScale(),
                  transform=T.Compose([
                      T.SamplePoints(num=num_points),
                      T.RandomTranslate(0.01),
                      T.RandomRotate(180, axis=1),
                      T.RandomRotate(15, axis=0),
                      T.RandomRotate(15, axis=2)
                  ]))

    weight = None

    if weighted_loss:
        y = ds.data.y.numpy()
        weight = compute_class_weight('balanced', classes=np.unique(y), y=y)
        weight = torch.from_numpy(weight).float().to(net_kwargs.get('device', device))
    
    opts = dict(DEFAULT_NET_PARAMS)
    opts.update({
        'lr': 0.001,
        'max_epochs': 9999999999,
        'optimizer': getattr(torch.optim, optimizer),
        'train_split': None if train_split <= 0 else CVSplit(cv=train_split, stratified=True, random_state=42),
        'criterion': torch.nn.CrossEntropyLoss,
        'criterion__weight': weight,
        'callbacks': [
            ('progress_bar', ProgressBar),
            ('train_acc', EpochScoring),
            ('train_bal', EpochScoring),
            ('checkpoint', Checkpoint),
        ],
        'callbacks__train_acc__scoring': 'accuracy',
        'callbacks__train_acc__name': 'train_acc',
        'callbacks__train_acc__on_train': True,
        'callbacks__train_acc__lower_is_better': False,
        'callbacks__train_bal__scoring': 'balanced_accuracy',
        'callbacks__train_bal__name': 'train_bal',
        'callbacks__train_bal__on_train': True,
        'callbacks__train_bal__lower_is_better': False,
        'callbacks__checkpoint__monitor': 'train_acc_best',
        'callbacks__checkpoint__f_params': params_path,
        'callbacks__checkpoint__f_optimizer': None,
        'callbacks__checkpoint__f_criterion': None,
        'callbacks__checkpoint__f_history': history_path,
        'callbacks__checkpoint__f_pickle': None,
    })
    
    if cosine_annealing:
        opts['callbacks'].append(('lr_scheduler', LRScheduler))  # noqa
        opts.update({
            'callbacks__lr_scheduler__policy': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            'callbacks__lr_scheduler__T_0': 10,
            'callbacks__lr_scheduler__T_mult': 2,
        })
    
    if train_split > 0:
        opts['callbacks'].append(('valid_bal', EpochScoring))
        opts.update({
            'callbacks__checkpoint__monitor': 'valid_acc_best',
            'callbacks__valid_bal__scoring': 'balanced_accuracy',
            'callbacks__valid_bal__name': 'valid_bal',
            'callbacks__valid_bal__on_train': False,
            'callbacks__valid_bal__lower_is_better': False
        })
    
    opts.update(net_kwargs)
    
    NeuralNetClassifier(
        module=getattr(models, model),
        module__dataset=ds,
        **opts
    ).fit(ds, ds.data.y)


def test(params_path: str = 'params.pt',
         num_points: int = 1024,
         model: str = 'PointNet',
         dataset_path: str = './dataset/ModelNet40/',
         **net_kwargs):
    ds = ModelNet(dataset_path, '40', train=False,
                  pre_transform=T.NormalizeScale(),
                  transform=T.SamplePoints(num=num_points))

    opts = dict(DEFAULT_NET_PARAMS)
    opts.update(net_kwargs)

    net = NeuralNetClassifier(
        module=getattr(models, model),
        module__dataset=ds,
        callbacks=[('progress_bar', ProgressBar)],
        dataset__length=len(ds),
        **opts
    ).initialize()

    net.load_params(params_path)
    return net.score(ds, ds.data.y)


if __name__ == "__main__":
    fire.Fire({
        'train': train,
        'test': test
    })
