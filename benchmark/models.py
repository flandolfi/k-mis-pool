import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.data import Dataset
from torch_geometric.nn import conv, glob, knn_graph
from torch_geometric.typing import SparseTensor, Tensor, OptPairTensor

from miss import MISSPool


class MLP(nn.Sequential):
    def __init__(self, *hidden, dropout=0., batch_norm=True):
        modules = []
        
        for ch_in, ch_out in zip(hidden[:-1], hidden[1:]):
            if batch_norm:
                modules.append(nn.BatchNorm1d(ch_in))
                
            modules.append(nn.ReLU())
            
            if not dropout:
                modules.append(nn.Dropout(dropout))
            
            modules.append(nn.Linear(ch_in, ch_out))
        
        super(MLP, self).__init__(*modules[:-1])


class MPSeq(nn.Sequential):
    def forward(self, input, *args, **kwargs):
        for module in self:
            if isinstance(module, conv.MessagePassing):
                input = module(input, *args, **kwargs)
            else:
                input = module(input)
        
        return input
    

class Block(nn.Module):
    def __init__(self, hidden=128, num_layers=3, dropout=0., gnn='ChebConv', **kwargs):
        super(Block, self).__init__()
        
        if isinstance(gnn, str):
            gnn = getattr(conv, gnn)
            
        self.batch_norms = nn.ModuleList([
            nn.Sequential(nn.BatchNorm1d(hidden), nn.ReLU()) for _ in range(num_layers)
        ])
            
        self.gnn_convs = nn.ModuleList([
            gnn(hidden, hidden, **kwargs) for _ in range(num_layers)
        ])
        
        self.mlps = nn.ModuleList([
            MLP(hidden, hidden, hidden, dropout=dropout) for _ in range(num_layers)
        ])
        
    def forward(self, x, *args, **kwargs):
        for bn, gnn, mlp in zip(self.batch_norms, self.gnn_convs, self.mlps):
            res = x
            x = bn(x)
            x = gnn(x, *args, **kwargs)
            x = mlp(x)
            x = x + res
        
        return x


class GNN(nn.Module):
    def __init__(self, dataset: Dataset, hidden=32, num_layers=5, **pool_kwargs):
        super(GNN, self).__init__()

        pos = dataset[0].pos
        pos_dim = 0 if pos is None else pos.size(1)
        in_dim = dataset.num_node_features + pos_dim

        self.lin_in = nn.Linear(in_dim, hidden)
        self.pool = MISSPool(**pool_kwargs)
        
        self.blocks = nn.ModuleList([
            Block(hidden, num_layers=num_layers, dropout=0, gnn=conv.ChebConv, K=2),
            Block(hidden*2, num_layers=num_layers, dropout=0, gnn=conv.ChebConv, K=2),
            Block(hidden*4, num_layers=num_layers, dropout=0, gnn=conv.ChebConv, K=2),
        ])
        
        self.expanders = nn.ModuleList([
            nn.Linear(in_dim, hidden)
        ] + [
            nn.Linear(hidden*(2**i), hidden*(2**(i+1))) for i in range(len(self.blocks) - 1)
        ])

        out_dim = hidden*(2**(len(self.blocks)-1))
        self.lin_out = nn.Sequential(
            MLP(out_dim, out_dim//2, out_dim//4, dataset.num_classes, dropout=0.5)
        )

    def forward(self, data):
        x, pos, batch, n, b = data.x, data.pos, data.batch, data.num_nodes, data.num_graphs
        edge_index, edge_attr = knn_graph(pos, 5, batch, True), None
        
        if x is None:
            x = pos
        elif pos is not None:
            x = torch.cat([x, pos], dim=-1)

        for lin, block in zip(self.expanders, self.blocks):
            x = lin(x)
            x = block(x, edge_index, edge_attr, batch)
            adj, x, batch = self.pool(edge_index, edge_attr, x, batch=batch)
            row, col, edge_attr = adj.coo()
            edge_index = torch.stack([row, col])
            
        out = glob.global_mean_pool(x, batch, b)
        out = self.lin_out(out)

        return F.softmax(out)


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
