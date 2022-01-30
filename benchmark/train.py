from typing import Union, Optional
import warnings
import logging
import math
import os

from torch_geometric.data.lightning_datamodule import LightningDataset
from torch_geometric.datasets import TUDataset, MalNetTiny
from torch_geometric.transforms import Constant, Compose, ToUndirected
from torch_geometric import seed_everything

import pytorch_lightning as pl

from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from filelock import FileLock

from sklearn.model_selection import train_test_split

import pandas as pd

from benchmark import models

warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_datasets(dataset: str = 'DD',
                 root: str = './data/',
                 batch_size: int = -1,
                 num_workers: int = 0,
                 seed: int = 42):
    root = os.path.realpath(root)
    
    with FileLock(os.path.expanduser('~/.data.lock')):
        if dataset in {'mal-net', 'MalNet', 'MalNetTiny'}:
            dataset = MalNetTiny(root=os.path.join(root, 'MalNetTiny'),
                                 transform=Compose([Constant(), ToUndirected()]))
        else:
            dataset = TUDataset(root=root, name=dataset)
            
            if dataset.num_node_features == 0:
                dataset.transform = Constant()
    
    idx = list(range(len(dataset)))
    y = dataset.data.y.numpy()
    
    train_idx, test_idx = train_test_split(idx, test_size=0.2,
                                           random_state=seed,
                                           stratify=y)
    train_idx, valid_idx = train_test_split(train_idx, test_size=0.125,
                                            random_state=seed,
                                            stratify=y[train_idx])
    
    train_dataset = dataset[train_idx]
    valid_dataset = dataset[valid_idx]
    test_dataset = dataset[test_idx]
    
    return LightningDataset(train_dataset, valid_dataset, test_dataset,
                            batch_size=batch_size, num_workers=num_workers)
    

def train(model: str = 'Baseline',
          dataset: str = 'DD',
          root: str = './data/',
          config: Optional[dict] = None,
          num_workers: int = 0,
          test: bool = False,
          seed: int = 42,
          **trainer_kwargs):
    config = dict(config or {})
    batch_size = config.get('batch_size', 8)
    
    if 'batch_size' in config:
        config.pop('batch_size')

    seed_everything(seed)
    datamodule = get_datasets(dataset, root,
                              batch_size=batch_size,
                              num_workers=num_workers)
    model_cls = getattr(models, model)
    model = model_cls(dataset=datamodule.train_dataset, **config)

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule)  # noqa
    
    if test:
        return trainer.test(datamodule=datamodule)  # noqa
    
    return None


def grid_search(model: str = 'Baseline',
                dataset: str = 'DD',
                root: str = './data/',
                opt_grid: Optional[dict] = None,
                local_dir: str = "./results/",
                cpu_per_trial: int = 1,
                gpu_per_trial: float = 0,
                verbose: int = 1,
                refit: Union[bool, int] = 10,
                seed: int = 42):
    logging_level = logging.INFO if verbose > 0 else logging.WARNING
    logging.basicConfig(format='%(levelname)s: %(message)s',
                        level=logging_level)

    param_grid = {
        'lr': tune.grid_search([0.001, 0.0001]),
        'batch_size': tune.grid_search([32, 64, 128]),
        'channels': tune.grid_search([32, 64, 128]),
        'gnn_class': tune.grid_search(['GCNConv', 'GATv2Conv', 'GINConv']),
    }

    if opt_grid is not None:
        logging.info("Updating parameter grid...")

        for k, v in opt_grid.items():
            if isinstance(v, list):
                param_grid[k] = tune.grid_search(v)
            else:
                param_grid[k] = v
                
    root = os.path.realpath(root)
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])
    
    def _train_partial(config):
        train(model, dataset, root, config,
              num_workers=cpu_per_trial, test=False,
              seed=seed, gpus=math.ceil(gpu_per_trial),
              progress_bar_refresh_rate=0,
              callbacks=[
                  TuneReportCheckpointCallback({
                      "loss": "val_loss",
                      "accuracy": "val_acc"
                  }, on="validation_end")
              ])

    logging.info("Starting grid search...")
    exp_name = f"{model}_{dataset}"
    analysis = tune.run(_train_partial,
                        resources_per_trial={
                            'cpu': cpu_per_trial,
                            'gpu': gpu_per_trial,
                        },
                        metric="accuracy",
                        mode="max",
                        local_dir=local_dir,
                        config=param_grid,
                        num_samples=1,
                        verbose=verbose,
                        progress_reporter=reporter,
                        raise_on_failed_trial=False,
                        fail_fast=False,
                        name=exp_name)

    logging.info(f"Best config:\t{analysis.best_config}\n")
    
    if not refit:
        logging.info(f"Checkpoint at:\t{analysis.best_checkpoint}\n")
        return

    results = []
    best_config = analysis.best_config

    for test_seed in range(int(refit)):
        metric_list = train(model=model, dataset=dataset, root=root,
                            config=dict(best_config), num_workers=cpu_per_trial,
                            gpus=math.ceil(gpu_per_trial), test=True, seed=test_seed)
        results.append(metric_list[0])

    logging.info(f"Model assessment results:\n\n{results}")
    
    df_results = pd.DataFrame.from_records(results)
    df_config = pd.DataFrame.from_records([best_config])
    results_path = os.path.join(local_dir, exp_name, 'model_assessment.json')
    config_path = os.path.join(local_dir, exp_name, 'best_config.json')
    df_results.to_json(results_path)
    df_config.to_json(config_path)
    logging.info(f"Results stored in\n"
                 f" - {results_path}\n"
                 f" - {config_path}")

