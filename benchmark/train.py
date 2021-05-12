import warnings
import logging

import torch
import numpy as np

from torch_geometric.data import DataLoader
from torch_geometric.datasets import GNNBenchmarkDataset

from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler, ProgressBar, Checkpoint, EpochScoring
from skorch.helper import predefined_split

from benchmark import utils
from benchmark import models
from benchmark.callbacks import LateStopping, LRLowerBound


utils.fix_skorch()
warnings.filterwarnings("ignore", category=UserWarning)

device = 'cpu'

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    device = 'cuda'

DEFAULT_NET_PARAMS = {
    'max_epochs': 9999999999,
    'verbose': 1,
    'device': device,
    'criterion': torch.nn.CrossEntropyLoss,
    'callbacks': [
        ('progress_bar', ProgressBar),
        ('train_acc', EpochScoring),
        ('checkpoint', Checkpoint),
        ('late_stopping', LateStopping),
        ('lr_scheduler', LRScheduler),
        ('lr_lower_bound', LRLowerBound),
    ],
    'callbacks__train_acc__scoring': 'accuracy',
    'callbacks__train_acc__lower_is_better': False,
    'callbacks__train_acc__on_train': True,
    'callbacks__train_acc__name': 'train_acc',
    'callbacks__late_stopping__hours': 12,
    'callbacks__lr_scheduler__policy': 'ReduceLROnPlateau',
    'callbacks__lr_scheduler__monitor': 'valid_loss',
    'callbacks__lr_scheduler__verbose': False,
    'callbacks__lr_scheduler__factor': 0.5,
    'callbacks__lr_scheduler__patience': 10,
    'callbacks__lr_lower_bound__min_lr': 1e-5,
    'callbacks__checkpoint__monitor': 'valid_acc_best',
    'callbacks__checkpoint__f_params': None,
    'callbacks__checkpoint__f_optimizer': None,
    'callbacks__checkpoint__f_criterion': None,
    'callbacks__checkpoint__f_history': None,
    'callbacks__checkpoint__f_pickle': None,
    'iterator_train': DataLoader,
    'iterator_valid': DataLoader,
    'iterator_valid__shuffle': False,
    'iterator_valid__drop_last': False,
}


def train(model: str = 'GNN',
          dataset: str = 'MNIST',
          root: str = './dataset/',
          optimizer: str = 'Adam',
          lr: float = 0.001,
          batch_size: int = -1,
          shuffle: bool = True,
          drop_last: bool = True,
          num_workers: int = 0,
          save_params: str = None,
          save_history: str = None,
          logging_level: int = logging.INFO,
          seed: int = 42,
          **net_kwargs):
    """Train a model on a GNNBenchmark Dataset.
    Args:
        model (str, optional): The model to train. Defaults to 'GNN'.
        root (str, optional): Dataset root directory. Defaults to './data/'.
        optimizer (str, optional): The optimization algorithm. Must be the
            name of an optimizer in `torch.optim`. Defaults to 'Adam'.
        lr (float, optional): Learning rate. Defaults to 0.001.
        batch_size (int, optional): Batch size. If `precompute_batches` is
            `True`, this value represents the maximum size of the batch, which
            are split equally or in nearly equal sizes. Defaults to 256.
        shuffle (bool, optional): Whether to shuffle the data during
            training. If `precompute_batches` is `True`, shuffles the batch
            order in which they are sampled. Defaults to `True`.
        drop_last (bool, optional): Whether to drop the last batch if its size
            is less than `batch_size`. Has no effect if `precompute_batches`
            is `True`. Defaults to `True`.
        num_workers (int, optional): Number of threads used for DataLoading.
            Defaults to 8.
        save_params (str, optional): Path of the model parameters to be stored
            at every checkpoint. Defaults to None.
        save_history (str, optional): Path of the history JSON to be stored at
            every checkpoint. Defaults to None.
        logging_level (int, optional): Threshold level for logs (see `logging`
            package documentation). Defaults to 10 (`logging.INFO`).
    """

    opts = dict(DEFAULT_NET_PARAMS)
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging_level)

    dataset = dataset.upper()
    logging.info(f'Loading {dataset} Dataset')
    train_dataset = GNNBenchmarkDataset(root=root, name=dataset, split='train')
    valid_dataset = GNNBenchmarkDataset(root=root, name=dataset, split='val')

    opts.update({
        'module': getattr(models, model),
        'module__dataset': train_dataset,
        'module__node_level': dataset in {'CLUSTER', 'PATTERN'},
        'lr': lr,
        'batch_size': batch_size,
        'optimizer': getattr(torch.optim, optimizer),
        'train_split': predefined_split(valid_dataset),
        'criterion': torch.nn.CrossEntropyLoss,
        'iterator_train__num_workers': num_workers,
        'iterator_valid__num_workers': num_workers,
        'iterator_train__shuffle': shuffle,
        'iterator_train__drop_last': drop_last,
        'callbacks__checkpoint__f_params': save_params,
        'callbacks__checkpoint__f_history': save_history,
        'dataset__length': len(train_dataset),
    })

    logging.debug(f'Setting random seed to {seed}')
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    logging.info('Initializing NeuralNet')
    opts.update(net_kwargs)
    net = NeuralNetClassifier(**opts).initialize()

    config = '\n'.join([f'{f"{k} ".ljust(50, ".")} {v}' for k, v in opts.items()])
    logging.debug('Configuration:\n\n%s\n', config)
    logging.debug('Network architecture:\n\n%s\n', str(net))
    logging.info('Starting training\n')
    net.partial_fit(train_dataset)


def score(model: str = 'GNN',
          dataset: str = 'MNIST',
          root: str = './dataset/',
          params_path: str = 'params.pt',
          batch_size: int = -1,
          num_workers: int = 0,
          logging_level: int = logging.INFO,
          **net_kwargs):
    opts = dict(DEFAULT_NET_PARAMS)
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging_level)

    dataset = dataset.upper()
    logging.info(f'Loading {dataset} Dataset')
    test_dataset = GNNBenchmarkDataset(root=root, name=dataset, split='test')

    opts.update({
        'module': getattr(models, model),
        'module__dataset': test_dataset,
        'module__node_level': dataset in {'CLUSTER', 'PATTERN'},
        'batch_size': batch_size,
        'iterator_valid__num_workers': num_workers,
        'dataset__length': len(test_dataset),
    })

    logging.info('Initializing NeuralNet')
    opts.update(net_kwargs)
    net = NeuralNetClassifier(**opts).initialize()
    net.load_params(f_params=params_path)

    config = '\n'.join([f'{f"{k} ".ljust(50, ".")} {v}' for k, v in opts.items()])
    logging.debug('Configuration:\n\n%s\n', config)
    logging.debug('Network architecture:\n\n%s\n', str(net))
    logging.info('Starting inference\n')
    return net.score(test_dataset, test_dataset.data.y)
