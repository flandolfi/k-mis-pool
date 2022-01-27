from typing import Callable, Tuple, Union, Optional

import torch
from torch.nn import Module
from torch_geometric.typing import Adj, Tensor, OptTensor
from torch_sparse import SparseTensor
from torch_scatter import scatter_min, scatter

from kmis import orderings, sample, scorers

Ordering = Callable[[Tensor, SparseTensor], Tensor]
Scoring = Callable[[Tensor, Adj, OptTensor], Tuple[Tensor, Tensor]]


def _cluster_k_mis(mis: Tensor, adj: SparseTensor, k: int = 1, rank: OptTensor = None) -> Tensor:
    n, device = mis.size(0), mis.device
    row, col, val = adj.coo()

    if rank is None:
        rank = torch.arange(n, dtype=torch.long, device=device)

    min_rank = torch.full((n,), fill_value=n, dtype=torch.long, device=device)
    rank_mis = rank[mis]
    min_rank[mis] = rank_mis

    for _ in range(k):
        min_neigh = torch.full_like(min_rank, fill_value=n)
        scatter_min(min_rank[row], col, out=min_neigh)
        torch.minimum(min_neigh, min_rank, out=min_rank)

    _, clusters = torch.unique(min_rank, return_inverse=True)
    perm = torch.argsort(rank_mis)
    return perm[clusters]


@torch.no_grad()
@torch.jit.script
def cluster_k_mis(mis: Tensor, adj: SparseTensor, k: int = 1, rank: OptTensor = None) -> Tensor:
    return _cluster_k_mis(mis, adj, k, rank)


@torch.no_grad()
@torch.jit.script
def sparse_reduce_adj(mis: Tensor, adj: SparseTensor, k: int = 1, rank: OptTensor = None,
                      remove_self_loops: bool = True, reduce: str = 'sum') -> Tuple[SparseTensor, Tensor]:
    cluster = _cluster_k_mis(mis, adj, k, rank)
    row, col, val = adj.coo()
    c = mis.sum()
    
    if val is None:
        val = torch.ones_like(row, dtype=torch.float)
    
    adj = SparseTensor(row=cluster[row], col=cluster[col],
                       value=val, is_sorted=False,
                       sparse_sizes=(c, c)).coalesce(reduce)
    
    if remove_self_loops:
        adj = adj.remove_diag()

    return adj, cluster


@torch.no_grad()
@torch.jit.script
def dense_reduce_adj(mis: Tensor, adj: SparseTensor, k: int = 1,
                     remove_self_loops: bool = True) -> Tuple[SparseTensor, SparseTensor]:
    r_mat = adj[mis, :]
    
    for _ in range(k - 1):
        r_mat = r_mat @ adj
        
    adj = (r_mat @ adj) @ r_mat.t()
    
    if remove_self_loops:
        adj = adj.remove_diag()
    
    return adj, r_mat


class KMISPooling(Module):
    def __init__(self, in_channels: Optional[int] = None,
                 k: int = 1,
                 scorer: Optional[Union[Scoring, str]] = 'random',
                 ordering: Optional[Union[Ordering, str]] = 'greedy',
                 reduce_x: Optional[str] = None,
                 reduce_edge: Optional[str] = 'sum',
                 remove_self_loops: bool = True):
        super(KMISPooling, self).__init__()
        
        if ordering is None:
            ordering = orderings.Greedy()
        elif isinstance(ordering, str):
            ordering = ''.join(t.title() for t in ordering.split('-'))
            
            if ordering == 'Greedy':
                ordering = orderings.Greedy()
            else:
                ordering_cls = getattr(orderings, ordering)
                ordering = ordering_cls(k=k)
                
        if scorer is None:
            scorer = scorers.CanonicalScorer()
        elif isinstance(scorer, str):
            if scorer == 'random':
                scorer = scorers.RandomScorer()
            elif scorer in {'const', 'constant'}:
                scorer = scorers.ConstantScorer()
            elif scorer in {'linear', 'nn'}:
                scorer = scorers.LinearScorer(in_channels=in_channels)
            elif scorer in {'sagpool', 'gnn'}:
                scorer = scorers.SAGScorer(in_channels=in_channels, k=k)
            elif scorer.endswith('pool'):
                scorer_cls = getattr(scorers, scorer[:-4].upper() + 'Scorer')
                scorer = scorer_cls(in_channels=in_channels)
            else:
                scorer = scorers.LambdaScorer(scorer)
        
        self.k = k
        self.ordering = ordering
        self.scorer = scorer
        self.reduce_edge = reduce_edge
        self.reduce_x = reduce_x
        self.remove_self_loops = remove_self_loops

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None,
                batch: OptTensor = None, **kwargs) -> Tuple[Tensor, Adj, OptTensor, OptTensor,
                                                            Union[SparseTensor, Tensor], Tensor, Tensor]:
        adj, n = edge_index, x.size(0)
        x, score = self.scorer(x, edge_index, edge_attr, **kwargs)
        
        if torch.is_tensor(edge_index):
            adj = SparseTensor.from_edge_index(edge_index, edge_attr, (n, n))

        rank = self.ordering(score, adj)
        mis = sample.maximal_k_independent_set(adj, self.k, rank)
        
        if self.reduce_edge == 'dense':
            adj, cluster = dense_reduce_adj(mis, adj, self.k, self.remove_self_loops)
            
            if self.reduce_x is None:
                x = x[mis]
            else:
                x = cluster.spmm(x, reduce=self.reduce_x)
        else:
            adj, cluster = sparse_reduce_adj(mis, adj, self.k, rank,
                                             remove_self_loops=self.remove_self_loops,
                                             reduce=self.reduce_edge)
        
            if self.reduce_x is None:
                x = x[mis]
            else:
                x = scatter(x, cluster, dim=0, dim_size=mis.sum(), reduce=self.reduce_x)

        if self.remove_self_loops:
            adj = adj.remove_diag()
        
        if torch.is_tensor(edge_index):
            row, col, edge_attr = adj.coo()
            edge_index = torch.stack([row, col])
        else:
            edge_index, edge_attr = adj, None
        
        if batch is not None:
            batch = batch[mis]
        
        return x, edge_index, edge_attr, batch, cluster, mis, score
