import torch
from torch import nn

from torch_geometric.data import Dataset
from torch_geometric.nn import conv, glob
from torch_geometric.typing import Tensor, PairTensor, Adj, Union, OptTensor
from torch_geometric.utils import to_undirected, degree

from torch_cluster.knn import knn_graph

from miss import MISSPool


class MLP(nn.Sequential):
    def __init__(self, *hidden, dropout=0., bias=False, norm=None):
        modules = []
        
        for ch_in, ch_out in zip(hidden[:-1], hidden[1:]):
            if norm == 'batch' or norm is True:
                modules.append(nn.BatchNorm1d(ch_in, momentum=0.9))
            elif norm == 'layer':
                modules.append(nn.LayerNorm(ch_in))
                
            modules.append(nn.LeakyReLU(negative_slope=0.2))
            
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
    
    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * self.nn(torch.cat([x_i, x_j - x_i], dim=-1))


class GNN(nn.Module):
    def __init__(self, dataset: Dataset, hidden=64, knn=16, conv_aggr='add', **pool_kwargs):
        super(GNN, self).__init__()

        pos = dataset[0].pos
        pos_dim = 0 if pos is None else pos.size(1)
        in_dim = dataset.num_node_features + pos_dim
        
        self.knn = knn
        self.pool = MISSPool(**pool_kwargs)
        
        self.conv = nn.ModuleList([
            WeightedEdgeConv(MLP(2*in_dim, hidden, hidden, dropout=0, norm='batch'), aggr=conv_aggr),
            WeightedEdgeConv(MLP(2*hidden, hidden, hidden, dropout=0, norm='layer'), aggr=conv_aggr),
            WeightedEdgeConv(MLP(2*hidden, 2*hidden, 2*hidden, dropout=0, norm='layer'), aggr=conv_aggr),
            WeightedEdgeConv(MLP(4*hidden, 4*hidden, 4*hidden, dropout=0, norm='layer'), aggr=conv_aggr),
        ])

        self.jk = MLP(8*hidden, 16*hidden, dropout=0, norm='layer')
        self.lin_out = MLP(16*hidden, 8*hidden, 4*hidden, dataset.num_classes, dropout=0.5, bias=True, norm='batch')

    def forward(self, data):
        x, batch, n, b = data.pos, data.batch, data.num_nodes, data.num_graphs
        edge_index = knn_graph(x, self.knn, batch, True)
        row, col = edge_index = to_undirected(edge_index, n)
        edge_weight = 1./degree(row, n)[row]
        
        xs = []
        p_mats = []
            
        for i, gnn in enumerate(self.conv):
            x = gnn(x, edge_index, edge_weight)
            x_exp = x

            for p_mat in reversed(p_mats):
                x_exp = self.pool.unpool(p_mat, x_exp)[0]

            xs.append(x_exp)
            
            if i < len(self.conv) - 1:
                adj, p_mat, _, x, _ = self.pool(edge_index, edge_weight, x)
                p_mats.append(p_mat)

                row, col, edge_weight = adj.coo()
                edge_index = torch.stack([row, col])
        
        x = torch.cat(xs, dim=-1)
        x = self.jk(x)

        out = glob.global_max_pool(x, batch, b)
        out = self.lin_out(out)

        return out
