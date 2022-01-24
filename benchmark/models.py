from typing import Union, Callable, Optional

import torch

from torch_geometric.nn import Sequential, conv, pool
from torch_geometric.nn.glob import global_mean_pool
from torch_geometric.nn.models import MLP
from torch_geometric.data import Dataset

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


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
    }
    
    requires_nn = {
        'GINConv',
        'GINEConv',
    }
    
    def __init__(self, dataset: Dataset,
                 lr: float = 0.001,
                 patience: int = 20,
                 channels: int = 64,
                 channel_multiplier: int = 1,
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
        out_channels = channels
        
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
            
        layers.append((MLP([in_channels, in_channels//2, out_channels], dropout=0.5), 'x -> x'))
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
