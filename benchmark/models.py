from torch import nn

from torch_geometric.data import Batch, Dataset
from torch_geometric.nn import conv, glob

from miss import MISSPool


class PointNet(nn.Module):
    def __init__(self, dataset: Dataset, *pool_args, **pool_kwargs):
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
            conv.PointConv(local_nn=_block(x_dim, pos_dim, 64, 64, 128),
                           add_self_loops=True),
            conv.PointConv(local_nn=_block(128, pos_dim, 128, 128, 256),
                           add_self_loops=False),
            conv.PointConv(local_nn=_block(256, pos_dim, 256, 512, 1024),
                           add_self_loops=False)
        ])

        self.pool = nn.ModuleList([
            MISSPool(*pool_args, **pool_kwargs),
            MISSPool(*pool_args, **pool_kwargs)
        ])

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, dataset.num_classes)
        )

    def forward(self, index):
        data = Batch.from_data_list(self.dataset[index]).to(index.device)

        for conv_l, pool_l in zip(self.conv, self.pool):
            data.x = conv_l(data.x, data.pos, data.edge_index)
            data = pool_l(data)

        out = self.conv[-1](data.x, data.pos, data.edge_index)
        out = glob.global_max_pool(out, data.batch, data.num_graphs)
        out = self.mlp(out)

        return out

