import math
import warnings
import tempfile
from typing import Union

import fire
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.nn.modules.loss import BCEWithLogitsLoss, CrossEntropyLoss

from torch_geometric.data import DataLoader, Data

import skorch
from skorch import NeuralNetClassifier, NeuralNetBinaryClassifier
from skorch.callbacks import EpochScoring, LRScheduler, ProgressBar, Checkpoint

from benchmark import models
from benchmark.callbacks import LateStopping, LRLowerBound
from benchmark.datasets import get_dataset, merge_datasets, SkorchDataset

from sklearn.model_selection import GridSearchCV, cross_validate

from miss.transforms import MISSampling


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
    'module__ordering': 'max-norm',
    'optimizer': torch.optim.Adam,
    'device': device,
    'max_epochs': 999999999,
    'verbose': 1,
    'lr': 0.001,
    'batch_size': -1,
    'callbacks__late_stopping__hours': 12,
    'callbacks__lr_scheduler__policy': 'ReduceLROnPlateau',
    'callbacks__lr_scheduler__monitor': 'valid_loss',
    'callbacks__lr_scheduler__verbose': False,
    'callbacks__lr_scheduler__factor': 0.5,
    'callbacks__lr_scheduler__patience': 10,
    'callbacks__lr_lower_bound__min_lr': 1e-5,
    'callbacks__print_log__floatfmt': '.6f',
    'dataset': SkorchDataset,
    'iterator_train': DataLoader,
    'iterator_valid': DataLoader,
    'iterator_train__shuffle': True,
    'iterator_valid__shuffle': False,
    'iterator_train__drop_last': True,
    'iterator_valid__drop_last': False
}

DEFAULT_GRID_PARAMS = [{
    'module__normalize': [True, False],
    'module__aggr': ['mean', 'add', 'max']
}]


def get_scorer(score_name='valid_score'):
    def scorer(net, X, y=None):
        best_epoch = np.argwhere(net.history[:, f'{score_name}_best'])[-1, 0]
        return net.history[best_epoch, score_name]

    return scorer


def get_net(name, module__dataset, scorer, **net_kwargs):
    module = getattr(models, name)

    net_kwargs.setdefault("callbacks", [])
    net_kwargs.update({
        'callbacks': [
            ('train_score', EpochScoring(
                scoring=scorer,
                name='train_score',
                lower_is_better=False,
                on_train=True
            )),
            ('valid_score', EpochScoring(
                scoring=scorer,
                name='valid_score',
                lower_is_better=False,
                on_train=False
            ))
        ] + net_kwargs["callbacks"],
    })

    if not hasattr(module__dataset, 'num_tasks') or module__dataset.num_tasks == 1:
        criterion = CrossEntropyLoss
        net_cls = NeuralNetClassifier
    else:
        criterion = BCEWithLogitsLoss
        net_cls = NeuralNetBinaryClassifier

    net_cls.score = get_scorer('valid_score')
    net = net_cls(
        module=module,
        module__dataset=module__dataset,
        criterion=criterion,
        **net_kwargs
    )
    net.set_params(callbacks__valid_acc=None)

    return net


def cv_iter(train_idx, val_idx, repetitions=3):
    for _ in range(repetitions):
        yield train_idx, val_idx


def grid_search(model_name: str, dataset_name: str,
                param_grid: Union[list, dict] = None,
                root: str = './dataset/',
                repetitions: int = 3,
                max_nodes: int = None,
                cv_results_path: str = None,
                **net_kwargs):
    if param_grid is None:
        param_grid = DEFAULT_GRID_PARAMS
    elif not isinstance(param_grid, list):
        param_grid = list(param_grid)

    total = sum([math.prod(map(len, grid.values())) for grid in param_grid])
    pbar = tqdm(total=total*repetitions, leave=False, desc='Grid Search')

    (ds_train, ds_val, ds_test), scorer = get_dataset(dataset_name, root)
    X, [tr_split, val_split] = merge_datasets(ds_train, ds_val)

    if max_nodes is not None:
        X.transform = ds_test.transform = MISSampling(max_nodes)

    opts = dict(DEFAULT_NET_PARAMS)
    opts.update({
        'callbacks': [
            ('progress_bar', ProgressBar),
            ('late_stopping', LateStopping),
            ('lr_scheduler', LRScheduler),
            ('lr_lower_bound', LRLowerBound)
        ],
        'callbacks__print_log__sink': pbar.write,
        'callbacks__late_stopping__sink': pbar.write,
        'callbacks__lr_lower_bound__sink': pbar.write,
    })
    opts.update(net_kwargs)
    net = get_net(model_name, X, scorer, **opts)

    def _score_wrapper():
        def _wrapper(estimator, X, y=None):
            pbar.update()
            return estimator.score(X, y)

        return _wrapper

    gs = GridSearchCV(net, param_grid,
                      scoring=_score_wrapper(),
                      refit=False,
                      verbose=False,
                      cv=cv_iter(tr_split, val_split, repetitions))
    gs.fit(X, X.data.y)
    pbar.close()

    if cv_results_path is not None:
        df = pd.DataFrame.from_records(gs.cv_results_).drop('params', axis=1)
        df.to_csv(cv_results_path, sep='\t')

    return gs.best_params_, gs.best_score_


def count_params(model_name: str, dataset_name: str,
                 root: str = './dataset/', **net_kwargs):
    (dataset, _, _), scorer = get_dataset(dataset_name, root)
    net = get_net(model_name, dataset, scorer, **net_kwargs)
    net.initialize()

    return sum(p.numel() for p in net.module_.parameters() if p.requires_grad)


def cv(model_name: str, dataset_name: str,
       root: str = './dataset/',
       repetitions: int = 3,
       max_nodes: int = None,
       cv_results_path: str = None,
       **net_kwargs):
    (ds_train, ds_val, ds_test), scorer = get_dataset(dataset_name, root)
    X, [tr_split, val_split] = merge_datasets(ds_train, ds_val)
    tmp_fd = tempfile.NamedTemporaryFile(suffix='.pt')

    if max_nodes is not None:
        X.transform = ds_test.transform = MISSampling(max_nodes)

    opts = dict(DEFAULT_NET_PARAMS)
    opts.update({
        'callbacks': [
            ('progress_bar', ProgressBar),
            ('checkpoint', Checkpoint),
            ('late_stopping', LateStopping),
            ('lr_scheduler', LRScheduler),
            ('lr_lower_bound', LRLowerBound),
        ],
        'callbacks__checkpoint__monitor': 'valid_score_best',
        'callbacks__checkpoint__f_params': tmp_fd.name,
        'callbacks__checkpoint__f_optimizer': None,
        'callbacks__checkpoint__f_criterion': None,
        'callbacks__checkpoint__f_history': None,
        'callbacks__checkpoint__f_pickle': None,
    })
    opts.update(net_kwargs)
    net = get_net(model_name, X, scorer, **opts)

    def _get_test_scorer():
        X_test = SkorchDataset(ds_test, ds_test.data.y)

        def _test_scorer(net, X, y):
            net.load_params(tmp_fd.name)
            return scorer(net, X_test, X_test.y)

        return _test_scorer

    scores = cross_validate(net, X, X.data.y,
                            scoring=_get_test_scorer(),
                            return_train_score=True,
                            cv=cv_iter(tr_split, val_split, repetitions))
    tmp_fd.close()

    if cv_results_path is not None:
        df = pd.DataFrame.from_records(scores)
        df.to_csv(cv_results_path, sep='\t')

    return {k: list(v) for k, v in scores.items()}


if __name__ == "__main__":
    fire.Fire({
        'grid_search': grid_search,
        'count_params': count_params,
        'cv': cv
    })
