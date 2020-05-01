
import argparse
import numpy as np
import pandas as pd

import torch

import torch_geometric
from torch_geometric.datasets import TUDataset

from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.helper import predefined_split
from skorch.dataset import Dataset

from benchmark import model
from gpool.utils import add_node_features
from gpool.data import NDPDataset, CustomDataset

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, ParameterGrid
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from .add_pool import add_pool, add_pool_x


torch_geometric.nn.add_pool = add_pool
torch_geometric.nn.add_pool_x = add_pool_x


class TestScoring:
    def __init__(self, test_dataset):
        self.test_dataset = test_dataset
        self.y_true = [y for _, y in test_dataset]
    
    def __call__(self, net, X=None, y=None):
        y_pred = net.predict(self.test_dataset)

        return accuracy_score(self.y_true, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-validate a given model.")
    parser.add_argument('-m', '--model', type=str, default='SimplePool',
                        help="Model to cross-validate (default: %(default)s).",
                        choices=['BaseModel', 'SimplePool', 'DiffPool', 'TopKPool',
                                 'SAGPool', 'EdgePool', 'Graclus'])
    parser.add_argument('-d', '--dataset', type=str, default='ENZYMES', metavar='DS',
                        help="Dataset on which the cross-validation is performed."
                             " Must be a dataset from the TU Dortmund collection"
                             " or NPDDataset (default: %(default)s).")
    parser.add_argument('--jumping_knowledge', type=str, default='cat',
                        help="Jumping knowledge aggregation type (default:"
                             " %(default)s).", 
                        choices=['cat', 'lstm', ''])
    parser.add_argument('--global_pool_op', type=str, nargs='+', default=['add'], metavar='POOL',
                        help="Global aggregation function(s) (default:"
                             " %(default)s).")
    parser.add_argument('-o', '--ordering', type=str, default='random', metavar='ORDER',
                        help="Node ordering (default: %(default)s).")
    parser.add_argument('-a', '--aggr', type=str, default='add', metavar='AGGR',
                        help="Local aggregation functions (default: %(default)s).",
                        choices=['add', 'mean', 'max'])
    parser.add_argument('--max_epochs', type=int, default=1000, metavar='E',
                        help="Number of maximum epochs per training (default:"
                             " %(default)s).")
    parser.add_argument('--dense', action='store_true',
                        help="Use the dense form computation (default:"
                             " %(default)s).")
    parser.add_argument('--dense_from', type=int, default=0, metavar='L',
                        help="Use the dense form starting from the given layer,"
                             " and use the sparse form for the other layers."
                             " Only applicable to BaseModel and SimplePool (default:"
                             " %(default)s).")
    parser.add_argument('--easy', action='store_true',
                        help="Easy dataset. Only applicable to NPDDataset (default:"
                             " %(default)s).")
    parser.add_argument('--small', action='store_true',
                        help="Small dataset. Only applicable to NPDDataset (default:"
                             " %(default)s).")
    parser.add_argument('--only_gcn', action='store_true',
                        help="Do not use SAGEConv in the grid search (default:"
                             " %(default)s).")
    parser.add_argument('--cached', type=str, default=None,
                        help="Cache coarsened graphs.",
                        choices=['both', 'train', 'test'])
    parser.add_argument('--shuffle', action='store_true',
                        help="Shuffle training samples.")
    parser.add_argument('--patience', type=int, default=20,
                        help="Early-stopping patience epochs (default:"
                             " %(default)s).")
    parser.add_argument('-b', '--batch_size', type=int, default=-1, metavar='B',
                        help="The size of the batches used during training"
                             "(default: %(default)s).")
    parser.add_argument('--dropout', type=float, default=0.3, metavar='P',
                        help="Dropout probability in the final dense layer"
                             " (default: %(default)s).")
    parser.add_argument('--folds', type=int, default=10,
                        help="Number of outer folds (default: %(default)s).")
    parser.add_argument('-c', '--hidden', type=int, default=None, metavar='H',
                        help="Fix the number of channels during the grid search.")
    parser.add_argument('--min_layers', type=int, default=2, metavar='L',
                        help="Minimum number of layers in the grid search"
                             "(default: %(default)s).")
    parser.add_argument('--max_layers', type=int, default=3, metavar='L',
                        help="Maximum number of layers in the grid search (default:"
                             " %(default)s).")
    parser.add_argument('--inner_layers', type=int, default=2, metavar='L',
                        help="Number of layers within each convolutional block"
                             " (default: %(default)s).")
    parser.add_argument('--to_pickle', type=str, default='cv_results.pickle', metavar='PATH',
                        help="Path of the output pickle storing the history of the"
                             " cross-validation (default: %(default)s).")
    parser.add_argument('--from_pickle', type=str, default=None, metavar='PATH',
                        help="Compute the outer-fold accuracy of the given history."
                             " If set, ignores every other parameter and does not perform"
                             " cross validation (default: %(default)s).")
    args = parser.parse_args()

    if args.from_pickle is not None:
        results = pd.read_pickle(args.from_pickle)
        outer = results[(results.cv_type == 'outer') & results.valid_acc_best]
        outer.index = range(len(outer))
        test_acc = outer.test_acc[outer.groupby('fold').epoch.idxmax()]

        print("\nAccuracy: {:.2f} ± {:.2f}\n".format(
            100.*np.mean(test_acc), 
            100.*np.std(test_acc)
        ))

        exit()

    torch.manual_seed(42)
    np.random.seed(42)
    device = 'cpu'

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        device = 'cuda' 
    
    if args.dataset == 'NDPDataset':
        train, val, test = (NDPDataset('data/', 
                                       split=key, 
                                       easy=args.easy, 
                                       small=args.small) 
                            for key in ['train', 'val', 'test'])
        train_stop = len(train)
        val_stop = train_stop + len(val)
        test_stop = val_stop + len(test)
        train_idx = np.arange(train_stop)
        val_idx = np.arange(train_stop, val_stop)
        test_idx = np.arange(val_stop, test_stop)

        dataset = CustomDataset(list(train) + list(val) + list(test))
        skf_pbar = tqdm([(train_idx, test_idx)], disable=True) 
            
        X = np.arange(len(dataset)).reshape((-1, 1))
        y = dataset.data.y.numpy()
    else:
        dataset = TUDataset(root='data/' + args.dataset, name=args.dataset)

        if dataset.data.x is None:
            dataset = add_node_features(dataset)
            
        X = np.arange(len(dataset)).reshape((-1, 1))
        y = dataset.data.y.numpy()

        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        skf_pbar = tqdm(list(skf.split(X, y)), leave=True, desc='Outer CV')
        sss_split = 1./(args.folds - 1)
    
    results = []
    test_acc = []

    shared_params = {
        'module': getattr(model, args.model), 
        'module__dataset': dataset,
        'module__dropout': args.dropout,
        'module__num_inner_layers': args.inner_layers,
        'module__jumping_knowledge': args.jumping_knowledge,
        'module__device': device,
        'module__global_pool_op': args.global_pool_op,
        'batch_size': args.batch_size,
        'criterion': model.PoolLoss if args.model == 'DiffPool' else torch.nn.modules.loss.NLLLoss,
        'optimizer': torch.optim.Adam,
        'optimizer__weight_decay': 1e-4,
        'callbacks__print_log__sink': skf_pbar.write,
        'iterator_train__shuffle': args.shuffle,
        'device': device
    }

    param_grid = {
        'optimizer__lr': [1e-3, 5e-4, 2e-4, 1e-4],
        'module__graph_sage': [False] if args.only_gcn else [True, False],
        'module__hidden': [64, 128] if args.hidden is None else [args.hidden],
        'module__num_layers': list(range(args.min_layers, args.max_layers + 1))
    }

    if args.model == 'SimplePool':
        param_grid['module__pool_size'] = [1, 2, 3]

        shared_params.update(
            module__ordering=args.ordering,
            module__cached=args.cached,
            module__aggr=args.aggr,
            module__dense=args.dense_from if args.dense else False
        )
    elif args.model == 'EdgePool':
        param_grid.update({
            'module__method': ['softmax', 'tanh'],
            'module__edge_dropout': [0.0, 0.2]
        })
    elif args.model == 'BaseModel':
        shared_params.update(module__dense=args.dense_from if args.dense else False)
    elif args.model == 'Graclus':
        shared_params.update(module__aggr=args.aggr)
    else:
        param_grid.update({
            'module__ratio': [0.25, 0.5, 0.75]
        })

    for out_iter, (train_idx, test_idx) in enumerate(skf_pbar):
        train_X = X[train_idx]
        train_y = y[train_idx]
        test_X = X[test_idx]
        test_y = y[test_idx]

        if args.dataset == 'NDPDataset':
            val_X = X[val_idx]
            val_y = y[val_idx]
        else:
            in_sss = StratifiedShuffleSplit(n_splits=1, test_size=sss_split, random_state=42)
            train_idx, val_idx = next(in_sss.split(train_X, train_y))

            val_X = train_X[val_idx]
            val_y = train_y[val_idx]
            train_X = train_X[train_idx]
            train_y = train_y[train_idx]
        
        valid_ds = Dataset(val_X, val_y)
        test_ds = Dataset(test_X, test_y)

        gs_pbar = tqdm(list(ParameterGrid(param_grid)), leave=True, desc='Grid Search')
        test_scoring = TestScoring(test_ds)
        drop_last = (args.batch_size != -1) and (len(train_X) % args.batch_size == 1)

        best_acc = 0.
        best_params = None

        for params in gs_pbar:
            gs_pbar.set_postfix({k.split('__')[1]: v for k, v in params.items()})
            net = NeuralNetClassifier(
                train_split=predefined_split(valid_ds), 
                max_epochs=args.max_epochs,
                callbacks=[('early_stopping', EarlyStopping)],
                callbacks__early_stopping__patience=args.patience,
                callbacks__early_stopping__sink=skf_pbar.write,
                iterator_train__drop_last=drop_last,
                **shared_params,
                **params
            ).fit(train_X, train_y)

            df = pd.DataFrame(net.history).drop('batches', 1)
            valid_acc = df[df.valid_acc_best].valid_acc.iloc[-1]

            df = pd.concat([df, pd.DataFrame([params for _ in df.iterrows()])], axis=1)
            df['cv_type'] = 'inner'
            df['fold'] = out_iter

            results.append(df)

            if valid_acc > best_acc:
                best_acc = valid_acc
                best_params = params
        
        gs_pbar.close()

        net = NeuralNetClassifier(
            train_split=predefined_split(valid_ds), 
            max_epochs=args.max_epochs,
            callbacks=[
                ('test_acc', EpochScoring),
                ('early_stopping', EarlyStopping)
            ],
            callbacks__early_stopping__patience=args.patience,
            callbacks__early_stopping__sink=skf_pbar.write,
            callbacks__test_acc__scoring=test_scoring,
            callbacks__test_acc__use_caching=False,
            callbacks__test_acc__lower_is_better=False,
            callbacks__test_acc__name='test_acc',
            iterator_train__drop_last=drop_last,
            **shared_params,
            **best_params
        ).fit(train_X, train_y)

        df = pd.DataFrame(net.history).drop('batches', 1)
        test_acc.append(df[df.valid_acc_best].test_acc.iloc[-1])

        df = pd.concat([df, pd.DataFrame([best_params for _ in df.iterrows()])], axis=1)
        df['cv_type'] = 'outer'
        df['fold'] = out_iter

        results.append(df)
        skf_pbar.set_postfix({
            'Accuracy': '{:.2f} ± {:.2f}\n'.format(
                    100.*np.mean(test_acc), 
                    100.*np.std(test_acc)
                )
            })
    
    skf_pbar.close()

    print('\nAccuracy: {:.2f} ± {:.2f}\n'.format(
        100.*np.mean(test_acc), 
        100.*np.std(test_acc)
    ))

    results = pd.concat(results, sort=False)
    results.to_pickle(args.to_pickle)
