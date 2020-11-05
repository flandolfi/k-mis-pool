import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.data import Dataset
from torch_geometric.nn import conv, glob

from miss import MISSPool


class PointNet(nn.Module):
    def __init__(self, dataset: Dataset, hidden=64, dropout=0.5, reduce_input=False, **pool_kwargs):
        super(PointNet, self).__init__()

        self.dataset = dataset
        self.reduce_input = reduce_input

        if reduce_input:
            self.pool_in = MISSPool(ordering="random", aggr="mean", weighted_aggr=False)

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
        if self.reduce_input:
            data = self.pool_in(data)

        for conv_l, pool_l in zip(self.conv, self.pool):
            data.x = conv_l(data.x, data.pos, data.edge_index)
            data = pool_l(data)

        out = self.conv[-1](data.x, data.pos, data.edge_index)
        out = glob.global_max_pool(out, data.batch, data.num_graphs)
        out = self.mlp(out)

        return out


class GCN(nn.Module):
    def __init__(self, dataset, hidden=146, num_layers=4, **pool_kwargs):
        super(GCN, self).__init__()

        self.dataset = dataset
        pos = dataset[0].pos
        pos_dim = 0 if pos is None else pos.size(1)
        x_dim = dataset.num_node_features + pos_dim

        self.lin_in = nn.Linear(x_dim, hidden)

        self.conv = nn.ModuleList([
            conv.GCNConv(hidden, hidden, add_self_loops=not l) for l in range(num_layers)
        ])

        self.pool = nn.ModuleList([
            MISSPool(add_self_loops=not l, **pool_kwargs) for l in range(num_layers - 1)
        ])

        self.lin_out = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.ReLU(),

            # nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),

            # nn.Dropout(0.5),
            nn.Linear(hidden//2, hidden//4),
            nn.BatchNorm1d(hidden//4),
            nn.ReLU(),

            nn.Linear(hidden//4, dataset.num_classes)
        )

    def forward(self, data):
        if 'pos' in data:
            data.x = torch.cat([data.x, data.pos], dim=-1)

        data.x = self.lin_in(data.x)

        for conv_l, pool_l in zip(self.conv, self.pool):
            data.x = conv_l(data.x, data.edge_index, data.edge_attr)
            data = pool_l(data)
            data.x = F.relu(data.x)

        out = self.conv[-1](data.x, data.edge_index, data.edge_attr)
        out = glob.global_mean_pool(out, data.batch, data.num_graphs)
        out = self.lin_out(out)

        return out
