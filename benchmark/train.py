import warnings
import os

from torch_geometric.data.lightning_datamodule import LightningDataset
from torch_geometric.datasets import TUDataset, MalNetTiny
from torch_geometric.transforms import Constant
from torch_geometric import seed_everything

import pytorch_lightning as pl

from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from filelock import FileLock

from sklearn.model_selection import train_test_split

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
        if dataset in {'mal-net', 'MalNet'}:
            dataset = MalNetTiny(root=os.path.join(root, 'MalNetTiny'))
        else:
            dataset = TUDataset(root=root, name=dataset.upper())
        
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
          config: dict = None,
          num_workers: int = 0,
          test: bool = False,
          seed: int = 42,
          **trainer_kwargs):
    config = config or {}
    batch_size = config.get('batch_size', 8)
    
    if 'batch_size' in config:
        config.pop('batch_size')

    seed_everything(seed)
    datamodule = get_datasets(dataset, root,
                              batch_size=batch_size,
                              num_workers=num_workers)
    model_cls = getattr(models, model)
    model = model_cls(datamodule.train_dataset, **config)

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule)  # noqa
    
    if test:
        return trainer.test(datamodule=datamodule)  # noqa
    
    return None


def grid_search(model: str = 'Baseline',
                dataset: str = 'DD',
                root: str = './data/',
                opt_grid: dict = None,
                local_dir: str = "./results/",
                cpu_per_trial: int = 1,
                gpu_per_trial: int = 0,
                verbose: int = 1,
                seed: int = 42):
    param_grid = {
        'lr': tune.grid_search([0.01, 0.001, 0.0001]),
        'batch_size': tune.grid_search([32, 64, 128]),
        'channels': tune.grid_search([32, 64, 128]),
        'channel_multiplier': tune.grid_search([1, 2]),
        'num_layers': tune.grid_search([2, 3]),
        'gnn_class': tune.grid_search(['GCNConv', 'GATv2Conv', 'GINConv']),
    }
    
    if opt_grid is not None:
        for k, v in opt_grid:
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
              seed=seed, gpus=gpu_per_trial,
              progress_bar_refresh_rate=0,
              callbacks=[
                  TuneReportCheckpointCallback({
                      "loss": "val_loss",
                      "accuracy": "val_acc"
                  }, on="validation_end")
              ])
    
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
                        name=f"{model}_{dataset}")
        
    print(f"Best config:\t{analysis.best_config}\n"
          f"Checkpoint at:\t{analysis.best_checkpoint}\n")
    
