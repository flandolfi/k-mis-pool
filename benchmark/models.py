import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.data import Dataset
from torch_geometric.nn import conv, glob, Sequential
from torch_geometric.typing import SparseTensor, Tensor, OptPairTensor
from torch_geometric.datasets import GNNBenchmarkDataset

from kmis import KMISCoarsening


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


class GNN(nn.Module):
    def __init__(self, dataset: Dataset, gnn="GCNConv",
                 hidden=146, num_layers=4, blocks=1,
                 node_level=False, readout=False,
                 k=1, ordering='random', eps=0.5,
                 sample_partition='on_train',
                 sample_aggregate=False,
                 **gnn_kwargs):
        super(GNN, self).__init__()

        pos = dataset[0].pos
        pos_dim = 0 if pos is None else pos.size(1)
        in_dim = dataset.num_node_features + pos_dim
        out_dim = 0

        self.num_blocks = blocks
        self.lin_in = nn.Linear(in_dim, hidden)
        self.pool = KMISCoarsening(k=k, ordering=ordering, eps=eps,
                                   sample_partition=sample_partition)
        self.ordering = self.pool.ordering
        self.pool.ordering = None
        self.readout = readout
        self.node_level = node_level

        if isinstance(gnn, str):
            gnn = getattr(conv, gnn)

        self.blocks = nn.ModuleList()

        if gnn in {conv.GCNConv, conv.ChebConv}:
            signature = 'x, edge_index, edge_attr -> x'
        else:
            signature = 'x, edge_index -> x'

        for b in range(blocks):
            gnn_layers = []

            for _ in range(num_layers):
                gnn_layers.extend([
                    (gnn(hidden, hidden, **gnn_kwargs), signature),  # noqa
                    nn.BatchNorm1d(hidden),
                    nn.ReLU()
                ])

            block = Sequential('x, edge_index, edge_attr', gnn_layers)

            if b == 0:
                self.in_block = block
            else:
                self.blocks.append(block)

            if readout or b == blocks - 1:
                out_dim += hidden

        self.lin_out = nn.Sequential(
            nn.ReLU(),
            MLP(out_dim, hidden // 2, hidden // 4, dataset.num_classes)
        )

    def forward(self, data):
        x, pos, batch, n, b = data.x, data.pos, data.batch, data.num_nodes, data.num_graphs
        edge_index, edge_attr = data.edge_index, data.edge_attr

        if edge_attr is None:
            edge_attr = torch.ones_like(edge_index[0], dtype=torch.float)

        if pos is not None:
            x = torch.cat([x, pos], dim=-1)

        adj = SparseTensor.from_edge_index(edge_index, edge_attr, sparse_sizes=(n, n))

        rank = self.ordering(x, adj)
        self.pool.ordering = lambda *args, **kwargs: rank

        xs = []
        c_mats = []

        x = self.lin_in(x)
        out = x = self.in_block(x, edge_index, edge_attr)

        for i, block in enumerate(self.blocks):
            if self.node_level:
                xs.append(out)
            elif self.readout:
                xs.append(glob.global_mean_pool(out, batch, b))

            x, edge_index, batch, mis, (c_mat, p_mat) = self.pool(x, edge_index, edge_attr, batch)
            _, rank = torch.unique(rank[mis], sorted=True, return_inverse=True)

            row, col, edge_attr = edge_index.coo()
            edge_index = torch.stack([row, col])
            out = x = block(x, edge_index, edge_attr)

            if self.node_level:
                c_mats.append(c_mat)

                for m in c_mats[::-1]:
                    out = m @ out

        if self.node_level:
            xs.append(out)
        else:
            xs.append(glob.global_mean_pool(out, batch, b))

        if self.readout:
            out = torch.cat(xs, dim=-1)
        elif self.node_level:
            out = sum(xs)
        else:
            out = xs[-1]

        return self.lin_out(out)


class GCN(GNN):
    def __init__(self, *args, **kwargs):
        super(GCN, self).__init__(gnn="GCNConv", *args, **kwargs)


class ChebNet(GNN):
    def __init__(self, dataset, hidden=107, K=2, normalization="sym", *args, **kwargs):
        super(ChebNet, self).__init__(dataset, gnn="ChebConv", hidden=hidden, 
                                      K=K, normalization=normalization, *args, **kwargs)


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


def partial_class(cls, *args, **kwargs):
    class Wrapper(cls):
        def __init__(self, *opts, **kwopts):
            super(Wrapper, self).__init__(*args, *opts, **kwargs, **kwopts)
    
    return Wrapper


def count_params(model: str = 'GCN', dataset: str = 'MNIST',
                 root: str = './dataset/', **net_kwargs):
    dataset = GNNBenchmarkDataset(root, dataset)
    net = globals()[model](dataset, **net_kwargs)

    return sum(p.numel() for p in net.parameters() if p.requires_grad)


GCN_100K = GCN
GCN_500K = partial_class(GCN, hidden=198, num_layers=12)
GraphSAGE_100K = GraphSAGE
GraphSAGE_500K = partial_class(GraphSAGE, hidden=117, num_layers=12)
ChebNet_100K = ChebNet
ChebNet_500K = partial_class(ChebNet, hidden=142, num_layers=12)

GCN_P_100K = partial_class(GCN, hidden=106, blocks=2)
GCN_P_500K = partial_class(GCN, hidden=198, blocks=3)
GraphSAGE_P_100K = partial_class(GraphSAGE, hidden=63, blocks=2)
GraphSAGE_P_500K = partial_class(GraphSAGE, hidden=117, blocks=3)
ChebNet_P_100K = partial_class(ChebNet, hidden=77, blocks=2)
ChebNet_P_500K = partial_class(ChebNet, hidden=142, blocks=3)
