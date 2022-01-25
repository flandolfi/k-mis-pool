from typing import Union, Callable, Optional

import torch

from torch_geometric.nn import Sequential, conv, pool
from torch_geometric.nn.glob import global_mean_pool
from torch_geometric.nn.pool import TopKPooling, SAGPooling, ASAPooling, PANPooling
from torch_geometric.nn.models import MLP
from torch_geometric.data import InMemoryDataset, Data

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from kmis import KMISPooling


class Baseline(LightningModule):
    known_signatures = {
        'GCNConv': 'x, e_i, e_w -> x',
        'GATConv': 'x, e_i, e_w -> x',
        'GATv2Conv': 'x, e_i, e_w -> x',
        'SAGEConv': 'x, e_i -> x',
        'GraphConv': 'x, e_i, e_w -> x',
        'SGConv': 'x, e_i, e_w -> x',
        'ChebConv': 'x, e_i, e_w -> x',
        'LEConv': 'x, e_i, e_w -> x',
        'GINConv': 'x, e_i -> x',
        'PANConv': 'x, e_i -> x, M',
        'TopKPooling': 'x, e_i, e_w, b -> x, e_i, e_w, b, perm, score',
        'SAGPooling': 'x, e_i, e_w, b -> x, e_i, e_w, b, perm, score',
        'ASAPooling': 'x, e_i, e_w, b -> x, e_i, e_w, b, perm',
        'PANPooling': 'x, M, b -> x, e_i, e_w, b, perm, score',
        'KMISPooling': 'x, e_i, e_w, b -> x, e_i, e_w, b, perm, mis, score',
    }
    
    requires_nn = {
        'GINConv',
        'GINEConv',
    }
    
    def __init__(self, dataset: InMemoryDataset,
                 lr: float = 0.001,
                 patience: int = 30,
                 channels: int = 64,
                 channel_multiplier: int = 2,
                 num_layers: int = 3,
                 gnn_class: Union[str, Callable] = 'GCNConv',
                 gnn_signature: Optional[str] = None,
                 gnn_kwargs: Optional[dict] = None,
                 pool_class: Optional[Union[str, Callable]] = None,
                 pool_signature: Optional[str] = None,
                 pool_kwargs: Optional[dict] = None):
        super(Baseline, self).__init__()
        
        if isinstance(gnn_class, str):
            gnn_class = getattr(conv, gnn_class)
            
        if gnn_class.__name__ in self.requires_nn:
            _gnn_cls = gnn_class
            
            def gnn_class(in_channels, out_channels, **kwargs):
                return _gnn_cls(nn=MLP([in_channels, in_channels, out_channels],
                                       batch_norm=False), **kwargs)

        if gnn_kwargs is None:
            gnn_kwargs = {}
            
        if gnn_signature is None:
            gnn_signature = self.known_signatures.get(gnn_class.__name__,
                                                      'x, e_i -> x')
            
        if isinstance(pool_class, str):
            pool_class = getattr(pool, pool_class)
        
        if pool_kwargs is None:
            pool_kwargs = {}
            
        if pool_class is not None and pool_signature is None:
            pool_signature = self.known_signatures.get(pool_class.__name__,
                                                       'x, e_i, e_w, b -> x, e_i, e_w, b')
        
        in_channels = dataset.num_node_features or 1
        out_channels = dataset.num_classes
        
        layers = []
        
        for l_id in range(num_layers):
            layers.append((gnn_class(in_channels=in_channels, out_channels=channels, **gnn_kwargs), gnn_signature))
            
            if l_id == num_layers - 1:
                layers.append((global_mean_pool, 'x, b -> x'))
            elif pool_class is not None:
                layers.append((pool_class(in_channels=channels, **pool_kwargs), pool_signature))
                
            layers.append((torch.nn.ReLU(), 'x -> x'))
            
            in_channels = channels
            channels *= channel_multiplier
            
        layers.append((MLP([in_channels, in_channels//2, out_channels],
                           batch_norm=False, dropout=0.5), 'x -> x'))
        self.model = Sequential('x, e_i, e_w, b', layers)
        
        self.loss = torch.nn.CrossEntropyLoss()
        self.patience = patience
        self.lr = lr
        
    @staticmethod
    def accuracy(y_pred, y_true):
        y_class = torch.argmax(y_pred, dim=-1)
        return torch.mean(torch.eq(y_class, y_true).float())
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        return self.model(x, edge_index, edge_attr, batch)

    def training_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = self.loss(y_hat, data.y)
        self.log('train_loss', loss, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))
        return loss

    def validation_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = self.loss(y_hat, data.y)
        acc = self.accuracy(y_hat, data.y)
        self.log('val_loss', loss, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))
        self.log('val_acc', acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))
        return {
            'val_loss': loss,
            'val_acc': acc,
        }

    def test_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.edge_attr, data.batch)
        acc = self.accuracy(y_hat, data.y)
        self.log('test_acc', acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def configure_callbacks(self):
        early_stop = EarlyStopping(monitor="val_loss", mode="min",
                                   patience=self.patience)
        checkpoint = ModelCheckpoint(monitor="val_acc", mode="max")
        return [early_stop, checkpoint]


class TopKPool(Baseline):
    def __init__(self, dataset: InMemoryDataset, ratio: float, **kwargs):
        kwargs['pool_class'] = TopKPooling
        kwargs['pool_kwargs'] = {'ratio': ratio}
        super(TopKPool, self).__init__(dataset=dataset, **kwargs)


class SAGPool(Baseline):
    def __init__(self, dataset: InMemoryDataset, ratio: float, **kwargs):
        # Simulate the "augmentation" variant of
        # SAGPool paper with a 2-hop SGConv
        def _gnn_wrap(*gnn_args, **gnn_kwargs):
            return conv.SGConv(*gnn_args, K=2, **gnn_kwargs)

        kwargs['pool_class'] = SAGPooling
        kwargs['pool_kwargs'] = {
            'ratio': ratio,
            'GNN': _gnn_wrap,
        }

        super(SAGPool, self).__init__(dataset=dataset, **kwargs)


class ASAPool(Baseline):
    def __init__(self, dataset: InMemoryDataset, ratio: float, **kwargs):
        kwargs['pool_class'] = ASAPooling
        kwargs['pool_kwargs'] = {'ratio': ratio}
        super(ASAPool, self).__init__(dataset=dataset, **kwargs)


class PANPool(Baseline):
    def __init__(self, dataset: InMemoryDataset, ratio: float, filter_size: int, **kwargs):
        kwargs['gnn_class'] = conv.PANConv
        kwargs['gnn_kwargs'] = {'filter_size': filter_size}
        kwargs['pool_class'] = PANPooling
        kwargs['pool_kwargs'] = {'ratio': ratio}
        super(PANPool, self).__init__(dataset=dataset, **kwargs)


class GraclusPool(Baseline):
    def __init__(self, dataset: InMemoryDataset, **kwargs):
        def _graclus_wrap(in_channels=None):
            def _graclus(x, e_i, e_w, b):
                cluster = pool.graclus(edge_index=e_i, weight=e_w, num_nodes=x.size(0))
                data = pool.avg_pool(cluster, Data(x=x, edge_index=e_i, edge_attr=e_w, batch=b))
                return data.x, data.edge_index, data.edge_weight, data.batch

            return _graclus

        kwargs['pool_class'] = _graclus_wrap
        kwargs['pool_signature'] = 'x, e_i, e_w, b -> x, e_i, e_w, b'

        super(GraclusPool, self).__init__(dataset=dataset, **kwargs)


class KMISPool(Baseline):
    def __init__(self, dataset: InMemoryDataset, k: int,
                 scorer: str = 'linear',
                 ordering: str = 'div-k-sum',
                 **kwargs):
        kwargs['pool_class'] = KMISPooling
        kwargs['pool_kwargs'] = {
            'k': k,
            'scorer': scorer,
            'ordering': ordering,
        }

        if 'gnn_class' in kwargs:
            gnn_class = kwargs['gnn_class']

            if gnn_class in {'ChebConv', conv.ChebConv}:
                kwargs['gnn_kwargs'] = {'K': k + 1}
            elif gnn_class in {'SGConv', 'TAGConv', conv.SGConv, conv.TAGConv}:
                kwargs['gnn_kwargs'] = {'K': k}

        super(KMISPool, self).__init__(dataset=dataset, **kwargs)


class KMISPoolRandom(KMISPool):
    def __init__(self, dataset: InMemoryDataset, k: int, **kwargs):
        super(KMISPoolRandom, self).__init__(dataset=dataset, k=k, scorer='random',
                                             ordering='greedy', **kwargs)


class KMISPoolNorm(KMISPool):
    def __init__(self, dataset: InMemoryDataset, k: int, **kwargs):
        super(KMISPoolNorm, self).__init__(dataset=dataset, k=k, scorer='norm', **kwargs)

