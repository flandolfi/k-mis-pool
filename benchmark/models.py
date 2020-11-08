import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.data import Dataset
from torch_geometric.nn import conv, glob
from torch_geometric.utils import add_self_loops
from torch_geometric.typing import SparseTensor, Tensor, OptPairTensor

from miss import MISSPool


class PointNet(nn.Module):
    def __init__(self, dataset: Dataset, hidden=64, dropout=0.5, **pool_kwargs):
        super(PointNet, self).__init__()

        self.dataset = dataset

        pos = dataset[0].pos
        pos_dim = 0 if pos is None else pos.size(1)
        x_dim = dataset.num_node_features

        def _block(x_dim, pos_dim, *units):
            mlp = nn.Sequential()
            mlp.add_module(f"lin_0", nn.Linear(x_dim + pos_dim, units[0]))

            for idx, (u_in, u_out) in enumerate(zip(units[:-1], units[1:])):
                mlp.add_module(f"relu_{idx}", nn.ReLU())
                mlp.add_module(f"lin_{idx+1}", nn.Linear(u_in, u_out))

            return mlp

        self.conv = nn.ModuleList([
            conv.PointConv(local_nn=_block(x_dim, pos_dim, hidden, hidden, hidden*2),
                           add_self_loops=True),
            conv.PointConv(local_nn=_block(hidden*2, pos_dim, hidden*2, hidden*2, hidden*4),
                           add_self_loops=False),
            conv.PointConv(local_nn=_block(hidden*4, pos_dim, hidden*4, hidden*8, hidden*16),
                           add_self_loops=False)
        ])

        self.pool = nn.ModuleList([
            MISSPool(add_self_loops=True, **pool_kwargs),
            MISSPool(add_self_loops=False, **pool_kwargs)
        ])

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(hidden*16),
            nn.ReLU(),

            nn.Dropout(dropout),
            nn.Linear(hidden*16, hidden*8),
            nn.BatchNorm1d(hidden*8),
            nn.ReLU(),

            nn.Dropout(dropout),
            nn.Linear(hidden*8, hidden*4),
            nn.BatchNorm1d(hidden*4),
            nn.ReLU(),

            nn.Linear(hidden*4, dataset.num_classes)
        )

    def forward(self, data):
        for conv_l, pool_l in zip(self.conv, self.pool):
            data.x = conv_l(data.x, data.pos, data.edge_index)
            data = pool_l(data)

        out = self.conv[-1](data.x, data.pos, data.edge_index)
        out = glob.global_max_pool(out, data.batch, data.num_graphs)
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

        self.blocks = blocks
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

        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()

        for b in range(blocks):
            self.conv.append(nn.ModuleList([
                gnn(hidden, hidden, **gnn_kwargs) for _ in range(num_layers)
            ]))

            self.bn.append(nn.ModuleList([
                nn.BatchNorm1d(hidden) for _ in range(num_layers)
            ]))

            if readout or b == blocks - 1:
                out_dim += hidden

            if b < blocks - 1:
                hidden *= hidden_factor

        self.lin_out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(out_dim, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, hidden//4),
            nn.ReLU(),

            nn.Linear(hidden//4, dataset.num_classes)
        )

    def _gcn_block(self, index, x, edge_index, edge_attr):
        for gcn, bn in zip(self.conv[index], self.bn[index]):  # noqa
            if self.has_weights:
                x = gcn(x, edge_index, edge_attr) + x
            else:
                x = gcn(x, edge_index) + x

            x = bn(x)
            x = F.relu(x)

        return x

    def forward(self, data):
        x, pos, batch, n, b = data.x, data.pos, data.batch, data.num_nodes, data.num_graphs
        edge_index, edge_attr = add_self_loops(data.edge_index, data.edge_attr, num_nodes=n)
        
        if x is None:
            x = pos
        elif pos is not None:
            x = torch.cat([x, pos], dim=-1)

        x = self.lin_in(x)
        xs = []

        for idx in range(self.blocks - 1):
            x = self._gcn_block(idx, x, edge_index, edge_attr)

            if self.readout:
                xs.append(glob.global_mean_pool(x, batch, b))

            x, edge_index, pos, batch = self.pool(x, edge_index, edge_attr, pos, batch, (n, n))
            x = x.repeat(1, self.hidden_factor)

        x = self._gcn_block(-1, x, edge_index, edge_attr)
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
