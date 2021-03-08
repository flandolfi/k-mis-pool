from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.data import Dataset
from torch_geometric.nn import conv, glob, knn_graph
from torch_geometric.typing import SparseTensor, Tensor, PairTensor, Adj, OptPairTensor, Union, OptTensor

from miss import MISSPool


class MLP(nn.Sequential):
    def __init__(self, *hidden, dropout=0., batch_norm=True):
        modules = []
        
        for ch_in, ch_out in zip(hidden[:-1], hidden[1:]):
            if batch_norm:
                modules.append(nn.BatchNorm1d(ch_in))
                
            modules.append(nn.LeakyReLU())
            
            if not dropout:
                modules.append(nn.Dropout(dropout))
            
            modules.append(nn.Linear(ch_in, ch_out))
        
        super(MLP, self).__init__(*modules)


class MPSeq(nn.Sequential):
    def forward(self, input, *args, **kwargs):
        for module in self:
            if isinstance(module, conv.MessagePassing):
                input = module(input, *args, **kwargs)
            else:
                input = module(input)
        
        return input


class WeightedEdgeConv(conv.EdgeConv):
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor, edge_weight: OptTensor)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
    
    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * self.nn(torch.cat([x_i, x_j - x_i], dim=-1))


class GNN(nn.Module):
    def __init__(self, dataset: Dataset, hidden=64, knn=16, aggr='add', **pool_kwargs):
        super(GNN, self).__init__()

        pos = dataset[0].pos
        pos_dim = 0 if pos is None else pos.size(1)
        in_dim = dataset.num_node_features + pos_dim
        
        self.knn = knn
        self.pool = MISSPool(**pool_kwargs)
        
        self.conv = nn.ModuleList([
            WeightedEdgeConv(MLP(2*in_dim, hidden, hidden, dropout=0), aggr=aggr),
            WeightedEdgeConv(MLP(2*hidden, hidden, hidden, dropout=0), aggr=aggr),
            WeightedEdgeConv(MLP(2*hidden, 2*hidden, 2*hidden, dropout=0), aggr=aggr),
            WeightedEdgeConv(MLP(4*hidden, 4*hidden, 4*hidden, dropout=0), aggr=aggr),
        ])
        
        self.lin_out = MLP(16*hidden, 8*hidden, 4*hidden, dataset.num_classes, dropout=0.5)

    def forward(self, data):
        x, pos, batch, n, b = data.x, data.pos, data.batch, data.num_nodes, data.num_graphs
        edge_index = knn_graph(pos, self.knn, batch, True)
        adj = SparseTensor.from_edge_index(edge_index).fill_value(1.)
        
        if x is None:
            x = pos
        elif pos is not None:
            x = torch.cat([x, pos], dim=-1)
        
        xs = []
            
        for i, gnn in enumerate(self.conv):
            x = gnn(x, adj)
            xs.append(glob.global_add_pool(x, batch, b))
            xs.append(glob.global_max_pool(x, batch, b))
            
            if i < len(self.conv) - 1:
                adj, x, batch = self.pool(adj, None, x, batch=batch)
        
        out = torch.cat(xs, dim=-1)
        out = self.lin_out(out)

        return F.softmax(out, dim=-1)


class GCN(GNN):
    def __init__(self, *args, **kwargs):
        super(GCN, self).__init__(gnn="GCNConv", *args, **kwargs)


class GIN(GNN):
    def __init__(self, dataset, hidden=90, *args, **kwargs):
        super(GIN, self).__init__(dataset, gnn="GINConv", hidden=hidden, *args, **kwargs)  # noqa


class GraphSAGEConv(conv.SAGEConv):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(GraphSAGEConv, self).__init__(in_channels, out_channels, *args, **kwargs)
        self.aggr = "max"
        self.pool_lin = nn.Linear(in_channels, in_channels)
        self.fuse = False

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return F.relu(self.pool_lin(x_j))

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
        return NotImplemented


class GraphSAGE(GNN):
    def __init__(self, dataset, hidden=90, *args, **kwargs):
        super(GraphSAGE, self).__init__(dataset, gnn=GraphSAGEConv, hidden=hidden, *args, **kwargs)  # noqa
