import torch
from torch import nn

from torch_geometric.nn import conv, glob
from torch_geometric.typing import Tensor, PairTensor, Adj, Union, OptTensor, PairOptTensor, Optional
from torch_geometric.utils import to_undirected, degree, remove_self_loops, add_self_loops

from torch_cluster import radius_graph

from torch_sparse import SparseTensor

from miss import MISSPool


class MLP(nn.Sequential):
    def __init__(self, *hidden, dropout=0., negative_slope=0.2, bias=False, norm=None):
        modules = []
        
        for ch_in, ch_out in zip(hidden[:-1], hidden[1:]):
            if norm == 'batch' or norm is True:
                modules.append(nn.BatchNorm1d(ch_in))
            elif norm == 'layer':
                modules.append(nn.LayerNorm(ch_in))
                
            modules.append(nn.LeakyReLU(negative_slope=negative_slope))
            
            if not dropout:
                modules.append(nn.Dropout(dropout))
            
            modules.append(nn.Linear(ch_in, ch_out, bias=bias))
        
        super(MLP, self).__init__(*modules)


class WeightedEdgeConv(conv.EdgeConv):
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor, edge_weight: OptTensor)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        msg = x_j - x_i
        if edge_weight is not None:
            msg = edge_weight.view(-1, 1) * msg
        return self.nn(torch.cat([x_i, msg], dim=-1))


class WeightedPointConv(conv.PointConv):
    def forward(self, x: Union[OptTensor, PairOptTensor],
                pos: Union[Tensor, PairTensor], edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        """"""
        if not isinstance(x, tuple):
            x: PairOptTensor = (x, None)

        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
                edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=pos[1].size(0))
            elif isinstance(edge_index, SparseTensor):
                edge_index = SparseTensor.set_diag(edge_index)

        # propagate_type: (x: PairOptTensor, pos: PairTensor)
        out = self.propagate(edge_index, edge_weight=edge_weight, x=x, pos=pos, size=None)

        if self.global_nn is not None:
            out = self.global_nn(out)

        return out

    def message(self, edge_weight: OptTensor, x_j: Optional[Tensor], pos_i: Tensor, pos_j: Tensor) -> Tensor:
        msg = pos_j - pos_i

        if edge_weight is not None:
            msg = edge_weight.view(-1, 1) * msg

        if x_j is not None:
            msg = torch.cat([x_j, msg], dim=1)
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg


class DGCNN(nn.Module):
    def __init__(self, dataset, hidden=64, radius=0.2, conv_aggr='max', edge_weights=True, norm='batch', **pool_kwargs):
        super(DGCNN, self).__init__()

        pos = dataset[0].pos
        pos_dim = 0 if pos is None else pos.size(1)
        in_dim = dataset.num_node_features + pos_dim

        self.radius = radius
        self.pool = MISSPool(**pool_kwargs)
        self.edge_weights = edge_weights

        self.conv = nn.ModuleList([
            WeightedEdgeConv(MLP(2*in_dim, hidden, dropout=0, norm=norm), aggr=conv_aggr),
            WeightedEdgeConv(MLP(2*hidden, hidden, dropout=0, norm=norm), aggr=conv_aggr),
            WeightedEdgeConv(MLP(2*hidden, 2*hidden, dropout=0, norm=norm), aggr=conv_aggr),
            WeightedEdgeConv(MLP(4*hidden, 4*hidden, dropout=0, norm=norm), aggr=conv_aggr),
        ])

        self.jk = MLP(8*hidden, 16*hidden, dropout=0, norm=norm)
        self.lin_out = MLP(32*hidden, 8*hidden, 4*hidden, dataset.num_classes, dropout=0.5, bias=True, norm='batch')

    def forward(self, data):
        x, batch, n, b = data.pos, data.batch, data.num_nodes, data.num_graphs
        edge_index = radius_graph(x, self.radius, batch, loop=True, max_num_neighbors=1024)
        row, col = edge_index = to_undirected(edge_index, n)

        edge_weight = 1. / degree(row, n)[row]

        xs = []
        p_mats = []

        for i, gnn in enumerate(self.conv):
            x = gnn(x, edge_index, edge_weight if self.edge_weights else None)
            x_exp = x

            for p_mat in reversed(p_mats):
                x_exp = self.pool.unpool(p_mat, x_exp)[0]

            xs.append(x_exp)

            if i < len(self.conv) - 1:
                adj, p_mat, _, _, x, _ = self.pool(edge_index, edge_weight, x)
                p_mats.append(p_mat)

                row, col, edge_weight = adj.coo()
                edge_index = torch.stack([row, col])

        x = torch.cat(xs, dim=-1)
        x = self.jk(x)

        out = torch.cat([
            glob.global_max_pool(x, batch, b),
            glob.global_mean_pool(x, batch, b)
        ], dim=-1)
        out = self.lin_out(out)

        return out


class PointNet(torch.nn.Module):
    def __init__(self, dataset, radius=0.2, conv_aggr='max', edge_weights=True, norm='batch', **pool_kwargs):
        super(PointNet, self).__init__()

        self.radius = radius
        self.pool = MISSPool(**pool_kwargs)
        self.edge_weights = edge_weights

        self.local_sa = nn.ModuleList([
            WeightedPointConv(MLP(3, 64, 64, 128, negative_slope=0., norm=norm),
                              add_self_loops=False, aggr=conv_aggr),
            WeightedPointConv(MLP(128 + 3, 128, 128, 256, negative_slope=0., norm=norm),
                              add_self_loops=False, aggr=conv_aggr)
        ])
        self.global_sa = MLP(256 + 3, 256, 512, 1024, negative_slope=0., norm=norm)

        self.mlp = MLP(1024, 512, 256, dataset.num_classes, dropout=0.5, bias=True, negative_slope=0., norm=norm)

    def forward(self, data):
        x, pos, batch, n, b = data.x, data.pos, data.batch, data.num_nodes, data.num_graphs
        edge_index = radius_graph(pos, self.radius, batch, loop=True, max_num_neighbors=32)
        row, col = edge_index = to_undirected(edge_index, n)

        edge_weight = 1. / degree(row, n)[row]

        for sa in self.local_sa:
            x = sa(x, pos, edge_index, edge_weight if self.edge_weights else None)
            adj, _, _, _, x, pos, batch = self.pool(edge_index, edge_weight, x, pos, batch=batch)

            row, col, edge_weight = adj.coo()
            edge_index = torch.stack([row, col])

        x = self.global_sa(torch.cat([x, pos], dim=-1))
        x = glob.global_max_pool(x, batch, b)
        x = self.mlp(x)

        return x
