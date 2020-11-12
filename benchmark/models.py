import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.data import Dataset
from torch_geometric.nn import conv, glob
from torch_geometric.utils import add_self_loops
from torch_geometric.typing import SparseTensor, Tensor, OptPairTensor

from miss import MISSPool


class MLP(nn.Sequential):
    def __init__(self, *hidden, dropout=0., batch_norm=False):
        modules = []
        
        for ch_in, ch_out in zip(hidden[:-1], hidden[1:]):
            if not dropout:
                modules.append(nn.Dropout(dropout))
            
            modules.append(nn.Linear(ch_in, ch_out))
            
            if batch_norm:
                modules.append(nn.BatchNorm1d(ch_out))
                
            modules.append(nn.ReLU())
        
        super(MLP, self).__init__(*modules[:-1])


class MPSeq(nn.Sequential):
    def forward(self, input, *args, **kwargs):
        for module in self:
            if isinstance(module, conv.MessagePassing):
                input = module(input, *args, **kwargs)
            else:
                input = module(input)
        
        return input


class PointNet(nn.Module):
    def __init__(self, dataset: Dataset, hidden=64, dropout=0.5, **pool_kwargs):
        super(PointNet, self).__init__()

        self.dataset = dataset

        pos = dataset[0].pos
        pos_dim = 0 if pos is None else pos.size(1)
        x_dim = dataset.num_node_features

        self.conv = nn.ModuleList([
            conv.PointConv(local_nn=MLP(x_dim + pos_dim, hidden, hidden, hidden*2, batch_norm=True),
                           add_self_loops=True),
            conv.PointConv(local_nn=MLP(hidden*2 + pos_dim, hidden*2, hidden*2, hidden*4, batch_norm=True),
                           add_self_loops=False),
            conv.PointConv(local_nn=MLP(hidden*4 + pos_dim, hidden*4, hidden*8, hidden*16, batch_norm=True),
                           add_self_loops=False)
        ])

        self.pool = MISSPool(add_self_loops=False, **pool_kwargs)

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(hidden*16),
            nn.ReLU(),
            
            MLP(hidden*16, hidden*8, hidden*4, dataset.num_classes, dropout=dropout, batch_norm=True)
        )

    def forward(self, data):
        x, pos, adj = data.x, data.pos, data.edge_index
        batch, b, n = data.batch, data.num_graphs, data.num_nodes

        for gcn in self.conv:
            x = gcn(x, pos, adj)
            x, adj, pos, batch = self.pool(x, adj, pos=pos, batch=batch)

        out = self.conv[-1](x, pos, adj)
        out = glob.global_max_pool(out, batch, b)
        out = self.mlp(out)

        return out


class GNN(nn.Module):
    def __init__(self, dataset, gnn="GCNConv",
                 hidden=146, num_layers=4, blocks=1,
                 hidden_factor=1, readout=False,
                 **pool_kwargs):
        super(GNN, self).__init__()

        pos = dataset[0].pos
        pos_dim = 0 if pos is None else pos.size(1)
        in_dim = dataset.num_node_features + pos_dim
        out_dim = 0

        self.num_blocks = blocks
        self.lin_in = nn.Linear(in_dim, hidden)
        self.has_weights = False
        self.pool = MISSPool(add_self_loops=False, **pool_kwargs)
        self.hidden_factor = hidden_factor
        self.readout = readout

        if isinstance(gnn, str):
            gnn = getattr(conv, gnn)

        gnn_kwargs = {}

        if gnn is conv.GCNConv:
            self.has_weights = True
            gnn_kwargs['add_self_loops'] = False
        elif gnn is conv.SAGEConv:
            gnn_kwargs['aggr'] = 'max'
        elif gnn in {conv.GINConv, conv.GINEConv, conv.EdgeConv}:
            gnn_cls = gnn
            
            if gnn is conv.EdgeConv:
                gnn_kwargs['aggr'] = 'max'
            else:
                gnn_kwargs['train_eps'] = True
            
            def gnn(channels_in, channels_out, **kwargs):
                return gnn_cls(MLP(channels_in, 2*channels_in, channels_out, batch_norm=True), **kwargs)

        self.blocks = nn.ModuleList()

        for b in range(blocks):
            gnn_layers = []
            
            for _ in range(num_layers):
                gnn_layers.extend([
                    gnn(hidden, hidden, **gnn_kwargs),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU()
                ])
                
            block = MPSeq(*gnn_layers)
            
            if b == 0:
                self.in_block = block
            else:
                self.blocks.append(block)

            if readout or b == blocks - 1:
                out_dim += hidden

            if b < blocks - 1:
                hidden *= hidden_factor

        self.lin_out = nn.Sequential(
            nn.ReLU(),
            MLP(out_dim, hidden//2, hidden//4, dataset.num_classes)
        )

    def forward(self, data):
        x, pos, batch, n, b = data.x, data.pos, data.batch, data.num_nodes, data.num_graphs
        edge_index, edge_attr = add_self_loops(data.edge_index, data.edge_attr, num_nodes=n)
        
        if x is None:
            x = pos
        elif pos is not None:
            x = torch.cat([x, pos], dim=-1)

        x = self.lin_in(x)
        xs = []
        
        x = self.in_block(x, edge_index, edge_attr if self.has_weights else None)

        for i, block in enumerate(self.blocks):
            if self.readout:
                xs.append(glob.global_mean_pool(x, batch, b))
            
            x, edge_index, pos, batch = self.pool(x, edge_index, edge_attr, pos, batch)
            x = x.repeat(1, self.hidden_factor)
            x = block(x, edge_index, edge_attr if self.has_weights else None)
            
        xs.append(glob.global_mean_pool(x, batch, b))
        out = torch.cat(xs, dim=-1)
        out = self.lin_out(out)

        return out


class GCN(GNN):
    def __init__(self, *args, **kwargs):
        super(GCN, self).__init__(gnn="GCNConv", *args, **kwargs)


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
