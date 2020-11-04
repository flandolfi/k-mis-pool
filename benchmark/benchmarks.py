import math
from typing import Union

import fire
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.nn.modules.loss import SmoothL1Loss, CrossEntropyLoss

from skorch import NeuralNetClassifier, NeuralNetRegressor
from skorch.callbacks import EpochScoring, LRScheduler, ProgressBar

from benchmark import models
from benchmark.callbacks import LateStopping, LRLowerBound
from benchmark.datasets import get_dataset, merge_datasets

from sklearn.model_selection import GridSearchCV


torch.manual_seed(42)
np.random.seed(42)
device = 'cpu'

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    device = 'cuda'


DEFAULT_NET_PARAMS = {
    'module__order_on': 'stride',
    'module__ordering': 'max-curvature',
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
    'callbacks__lr_lower_bound__min_lr': 1e-6,
    'callbacks__print_log__floatfmt': '.6f',
    'iterator_train__drop_last': True,
    'iterator_train__shuffle': True,
}

DEFAULT_GRID_PARAMS = [{
        'module__weighted_aggr': [True],
        'module__aggr': ['mean'],
        'module__pool_size': [1, 2]
    }, {
        'module__weighted_aggr': [False],
        'module__aggr': ['mean', 'add', 'max'],
        'module__pool_size': [1, 2]
}]


def get_scorer(score_name='valid_acc'):
    def scorer(net, X, y=None):
        best_epoch = np.argwhere(net.history[:, f'{score_name}_best'])[-1, 0]
        return net.history[best_epoch, score_name]

    return scorer


def get_net(name, dataset, **net_kwargs):
    module = getattr(models, name)
    regression = dataset.data.y.dtype == torch.float

    if regression:
        net_cls = NeuralNetRegressor
        criterion = SmoothL1Loss
        score_name = 'valid_mae'
        net_kwargs.setdefault("callbacks", [])
        net_kwargs.update({
            'callbacks': [(score_name, EpochScoring)] + net_kwargs["callbacks"],
            f'callbacks__{score_name}__scoring': 'neg_mean_absolute_error',
            f'callbacks__{score_name}__lower_is_better': False,
            f'callbacks__{score_name}__on_train': False,
            f'callbacks__{score_name}__name': score_name
        })
    else:
        net_cls = NeuralNetClassifier
        criterion = CrossEntropyLoss
        score_name = 'valid_acc'

    net_cls.score = get_scorer(score_name)
    net = net_cls(
        module=module,
        module__dataset=dataset,
        criterion=criterion,
        **net_kwargs
    )

    return net


def grid_search(model_name: str, dataset_name: str,
                param_grid: Union[list, dict] = None,
                root: str = './data/',
                repetitions: int = 3,
                cv_results_path: str = None,
                **net_kwargs):
    if param_grid is None:
        param_grid = DEFAULT_GRID_PARAMS
    elif not isinstance(param_grid, list):
        param_grid = list(param_grid)

    total = sum([math.prod(map(len, grid.values())) for grid in param_grid])
    pbar = tqdm(total=total*repetitions, leave=False, desc='Grid Search')
    dataset, tr_split, val_split, _ = merge_datasets(*get_dataset(dataset_name, root))

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
    net = get_net(model_name, dataset, **opts)

    def _cv_iter(train_idx, val_idx):
        for _ in range(repetitions):
            yield train_idx, val_idx

    def _score_wrapper():
        def scorer(estimator, X, y=None):
            pbar.update()
            return estimator.score(X, y)

        return scorer

    gs = GridSearchCV(net, param_grid,
                      scoring=_score_wrapper(),
                      refit=False,
                      verbose=False,
                      cv=_cv_iter(tr_split.X, val_split.X))
    gs.fit(np.arange(len(dataset)), dataset.data.y.numpy())
    pbar.close()

    if cv_results_path is not None:
        df = pd.DataFrame.from_records(gs.cv_results_).drop('params', axis=1)
        df.to_csv(cv_results_path, sep='\t')

    return gs.best_params_, gs.best_score_


if __name__ == "__main__":
    fire.Fire({
        'grid_search': grid_search
    })
