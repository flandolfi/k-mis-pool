from typing import Callable
from abc import ABC, abstractmethod

import torch
from torch.nn import Module, Parameter, functional as F

from torch_geometric.nn.dense import Linear
from torch_geometric.nn.conv import SGConv, LEConv
from torch_geometric.utils import add_remaining_self_loops, softmax
from torch_geometric.typing import Adj, Tensor, OptTensor, Tuple, Union, SparseTensor
from torch_scatter import scatter_max, scatter_add


class Scorer(ABC):
    @abstractmethod
    def __call__(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError
    

class ConstantScorer(Scorer):
    def __call__(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None):
        return x, torch.ones_like(x[:, 0])


class RandomScorer(Scorer):
    def __call__(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None):
        return x, torch.randperm(x.size(0), device=x.device)


class CanonicalScorer(Scorer):
    def __call__(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None):
        return x, torch.arange(x.size(0), device=x.device)
    

class LambdaScorer(Scorer):
    def __init__(self, scorer: Union[Callable, str]):
        if isinstance(scorer, str):
            scorer = getattr(torch, scorer)
            
        self.scorer = scorer
    
    def __call__(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None):
        score = self.scorer(x, dim=-1)
        
        if isinstance(score, tuple):  # Max-like functions
            score = score[0]
        
        return x, score.view(-1)


# The scoring function hereafter are adapted from the code already available at
# https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric/nn/pool

class LinearScorer(Module, Scorer):
    def __init__(self, in_channels: int):
        Module.__init__(self)
        self.lin = Linear(in_channels=in_channels, out_channels=1,
                          bias=False, weight_initializer='uniform')

    def forward(self, x: Tensor):
        score = self.lin(x)
        score = torch.sigmoid(score / self.lin.weight.norm(p=2, dim=-1))
        return x*score, score.view(-1)

    def __call__(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None):
        return Module.__call__(self, x)


class SAGScorer(Module, Scorer):
    def __init__(self, in_channels: int, k: int = 2):
        Module.__init__(self)
        self.gnn = SGConv(in_channels=in_channels, out_channels=1, K=k)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None):
        score = self.gnn(x, edge_index, edge_attr).sigmoid()
        return x*score, score.view(-1)

    def __call__(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None):
        return Module.__call__(self, x, edge_index, edge_attr)


class ASAScorer(Module, Scorer):
    def __init__(self, in_channels: int, dropout: float = 0.,
                 negative_slope: float = 0.2):
        Module.__init__(self)
    
        self.gnn_score = LEConv(in_channels=in_channels, out_channels=1)
        self.lin = Linear(in_channels, in_channels)
        self.att = Linear(2 * in_channels, 1)
        self.dropout = dropout
        self.negative_slope = negative_slope

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None):
        num_nodes = x.size(0)
        
        if not torch.is_tensor(edge_index):
            row, col, edge_attr = edge_index.coo()
            edge_index = torch.stack([row, col])
    
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_attr, fill_value=1., num_nodes=num_nodes)
    
        x = x.unsqueeze(-1) if x.dim() == 1 else x
    
        x_pool_j = x[edge_index[0]]
        x_q, _ = scatter_max(x_pool_j, edge_index[1], dim=0)
        x_q = self.lin(x_q)[edge_index[1]]
    
        score = self.att(torch.cat([x_q, x_pool_j], dim=-1)).view(-1)
        score = F.leaky_relu(score, self.negative_slope)
        score = softmax(score, edge_index[1], num_nodes=num_nodes)
    
        score = F.dropout(score, p=self.dropout, training=self.training)
    
        v_j = x[edge_index[0]] * score.view(-1, 1)
        x = scatter_add(v_j, edge_index[1], dim=0)
  
        score = self.gnn_score(x, edge_index).sigmoid()
        return x*score, score.view(-1)

    def __call__(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None):
        return Module.__call__(self, x, edge_index, edge_attr)


class PANScorer(Module, Scorer):
    def __init__(self, in_channels: int, M: SparseTensor = None):
        Module.__init__(self)
    
        self.p = Parameter(torch.Tensor(in_channels))
        self.beta = Parameter(torch.Tensor(2))
        self.M = M
        
        self.reset_parameters()

    def reset_parameters(self):
        self.p.data.fill_(1)
        self.beta.data.fill_(0.5)

    def update_met_matrix(self, M: SparseTensor):
        self.M = M

    def forward(self, x: Tensor, M: SparseTensor):
        row, col, edge_weight = M.coo()

        score1 = (x * self.p).sum(dim=-1)
        score2 = scatter_add(edge_weight, col, dim=0, dim_size=x.size(0))
        score = self.beta[0] * score1 + self.beta[1] * score2
        score = torch.sigmoid(score)

        return x*score.view(-1, 1), score

    def __call__(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None):
        return Module.__call__(self, x, self.M)


# Alias
TOPKScorer = LinearScorer
