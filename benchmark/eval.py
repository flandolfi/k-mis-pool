import argparse
import numpy as np

import torch

import torch_geometric
from torch_geometric.datasets import TUDataset

from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit, Dataset
from skorch.helper import predefined_split

from benchmark import model
from gpool.utils import add_node_features
from gpool.data import NDPDataset, CustomDataset

from sklearn.model_selection import StratifiedShuffleSplit

from benchmark.add_pool import add_pool, add_pool_x


torch_geometric.nn.add_pool = add_pool
torch_geometric.nn.add_pool_x = add_pool_x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a given model.")
    parser.add_argument('-m', '--model', type=str, default='SimplePool',
                        help="Model to evaluate (default: %(default)s).",
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
    parser.add_argument('-k', '--kernel_size', type=int, default=1, metavar='SIZE',
                        help="Pooling kernel size (default: %(default)s).")
    parser.add_argument('-s', '--stride', type=int, default=2, metavar='STRIDE',
                        help="Pooling stride (default: %(default)s).")
    parser.add_argument('-o', '--ordering', type=str, default='min-k-hop-degree', metavar='ORDER',
                        help="Node ordering (default: %(default)s).")
    parser.add_argument('-a', '--aggr', type=str, default='add', metavar='AGGR',
                        help="Local aggregation functions (default: %(default)s).",
                        choices=['add', 'avg', 'max'])
    parser.add_argument('--epochs', type=int, default=1000, metavar='E',
                        help="Number of epochs (default: %(default)s).")
    parser.add_argument('--dense', action='store_true',
                        help="Use the dense form computation.")
    parser.add_argument('--dense_from', type=int, default=0, metavar='L',
                        help="Use the dense form starting from the given layer,"
                             " and use the sparse form for the other layers."
                             " Only applicable to BaseModel and SimplePool (default:"
                             " %(default)s).")
    parser.add_argument('--easy', action='store_true',
                        help="Easy dataset. Only applicable to NPDDataset.")
    parser.add_argument('--small', action='store_true',
                        help="Small dataset. Only applicable to NPDDataset.")
    parser.add_argument('-b', '--batch_size', type=int, default=-1, metavar='B',
                        help="The size of the batches used during training"
                             " (default: %(default)s).")
    parser.add_argument('--dropout', type=float, default=0.3, metavar='P',
                        help="Dropout probability in the final dense layer"
                             " (default: %(default)s).")
    parser.add_argument('-c', '--hidden', type=int, default=64, metavar='H',
                        help="Number of channels (default: %(default)s).")
    parser.add_argument('-l', '--layers', type=int, default=2, metavar='L',
                        help="Number of convolutional blocks ")
    parser.add_argument('--inner_layers', type=int, default=2, metavar='L',
                        help="Number of layers within each convolutional block"
                             " (default: %(default)s).")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate (default: %(default)s).")
    parser.add_argument('--weight_decay', type=float, default=0.001, metavar='WD',
                        help="Weight decay (default: %(default)s).")
    parser.add_argument('--ratio', type=float, default=0.8,
                        help="Output/input number of nodes ratio. Only Applicable to"
                             " DiffPool, TopKPool, and SAGPool (default: %(default)s).")
    parser.add_argument('--split', type=float, default=0.1, metavar='S',
                        help="Test split (default: %(default)s).")
    parser.add_argument('--method', type=str, default='softmax',
                        help="Function to apply to compute the edge score from raw"
                             " edge scores. Only applicable to EdgePool (default:"
                             " %(default)s).",
                        choices=['softmax', 'sigmoid', 'tanh'])
    parser.add_argument('--edge_dropout', type=float, default=0.2, metavar='P',
                        help="probability with which to drop edge scores during"
                             " training. Only applicable to EdgePool (default:"
                             " %(default)s).")
    parser.add_argument('--graph_sage', action='store_true',
                        help="Use SAGEConv instead of GCNConv.")
    parser.add_argument('--skip_covered', action='store_true',
                        help="Give max priority to uncovered nodes. Only applicable"
                             " to CoverPool")
    parser.add_argument('--no_readout', action='store_false', 
                        help="Use only the final global pooling aggregation as input"
                             " to the dense layers.")
    parser.add_argument('--no_cache', action='store_false',
                        help="Do not pre-compute the graph covers.")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if args.dataset == 'NDPDataset':
        train, val, test = (NDPDataset('data/', 
                                       split=key, 
                                       easy=args.easy, 
                                       small=args.small) 
                            for key in ['train', 'val', 'test'])
        
        train_size = len(train) + len(val)
        dataset = CustomDataset(list(train) + list(val) + list(test))

        X = np.arange(len(dataset)).reshape((-1, 1))
        y = dataset.data.y.numpy()

        cv_split = predefined_split(Dataset(X[train_size:], y[train_size:]))
        X, y = X[:train_size], y[:train_size]
    else:
        dataset = TUDataset(root='data/' + args.dataset, name=args.dataset)

        if dataset.data.x is None:
            dataset = add_node_features(dataset)

        X = np.arange(len(dataset)).reshape((-1, 1))
        y = dataset.data.y.numpy()

        cv_split = CVSplit(cv=StratifiedShuffleSplit(test_size=args.split, n_splits=1, random_state=42))

    params = {
        'module': getattr(model, args.model), 
        'module__dataset': dataset,
        'module__num_layers': args.layers,
        'module__hidden': args.hidden,
        'module__graph_sage': args.graph_sage,
        'module__dropout': args.dropout,
        'module__num_inner_layers': args.inner_layers,
        'module__jumping_knowledge': args.jumping_knowledge,
        'module__readout': args.no_readout,
        'module__global_pool_op': args.global_pool_op,
        'module__device':device,
        'max_epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'criterion': model.PoolLoss if args.model == 'DiffPool' else torch.nn.modules.loss.NLLLoss,
        'optimizer': torch.optim.Adam,
        'optimizer__weight_decay': args.weight_decay,
        'iterator_train__shuffle': True,
        'train_split': cv_split,
        'device': device
    }

    if args.model == 'SimplePool':
        params.update(
            module__kernel_size=args.kernel_size,
            module__stride=args.stride,
            module__ordering=args.ordering,
            module__aggr=args.aggr,
            module__dense=args.dense_from if args.dense else False
        )
    elif args.model == 'EdgePool':
        params.update(module__method=args.method, module__edge_dropout=args.edge_dropout)
    elif args.model == 'BaseModel':
        params.update(module__dense=args.dense_from if args.dense else False)
    elif args.model == 'Graclus':
        params.update(module__aggr=args.aggr)
    else:
        params.update(module__ratio=args.ratio)

    NeuralNetClassifier(**params).fit(X, y)
