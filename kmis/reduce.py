from typing import Callable, Tuple, Union, Optional

import torch
from torch_geometric.typing import Adj, Tensor, OptTensor
from torch_sparse import SparseTensor
from torch_scatter import scatter_min, scatter

from kmis import orderings, sample

Ordering = Callable[[Tensor, SparseTensor], Tensor]
Scoring = Callable[[Tensor], OptTensor]


@torch.no_grad()
@torch.jit.script
def cluster_k_mis(mis: Tensor, adj: SparseTensor, k: int = 1, rank: OptTensor = None) -> Tensor:
    n, device = mis.size(0), mis.device
    row, col, val = adj.coo()
    
    if rank is None:
        rank = torch.arange(n, dtype=torch.long, device=device)

    min_rank = torch.full((n,), fill_value=n, dtype=torch.long, device=device)
    min_rank[mis] = rank[mis]

    for _ in range(k):
        scatter_min(min_rank[row], col, out=min_rank)

    _, clusters = torch.unique(min_rank, return_inverse=True)
    return clusters


class KMISPool(torch.nn.Module):
    def __init__(self, k: int = 1,
                 scorer: Optional[Union[Scoring, str]] = 'random',
                 adaptive: Union[bool, str] = 'infer',
                 ordering: Optional[Union[Ordering, str]] = 'greedy',
                 reduce_x: Optional[str] = None,
                 reduce_edge: Optional[str] = 'sum',
                 remove_self_loops: bool = True):
        super(KMISPool, self).__init__()
        
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
            scorer: Scoring = lambda x: torch.arange(x.size(0), device=x.device, dtype=x.dtype)
        elif isinstance(scorer, str):
            if scorer == 'random':
                scorer: Scoring = lambda x: torch.rand_like(x)
            else:
                scoring_fun = getattr(torch, scorer)
                scorer: Scoring = lambda x: scoring_fun(x, dim=-1)
                
        if adaptive == 'infer':
            self.adaptive = isinstance(scorer, torch.nn.Module)
        
        self.k = k
        self.ordering = ordering
        self.scorer = scorer
        self.reduce_edge = reduce_edge
        self.reduce_x = reduce_x
        self.remove_self_loops = remove_self_loops
        
        if adaptive == 'infer':
            self.adaptive = isinstance(scorer, torch.nn.Module)
        else:
            self.adaptive = adaptive

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None,
                batch: OptTensor = None) -> Tuple[Tensor, Adj, OptTensor, OptTensor]:
        adj, n = edge_index, x.size(0)
        
        if torch.is_tensor(edge_index):
            adj = SparseTensor.from_edge_index(edge_index, edge_attr, (n, n))

        score = self.scorer(x)
        rank = self.ordering(score, adj)
        mis = sample.maximal_k_independent_set(adj, self.k, rank)
        cluster = cluster_k_mis(mis, adj, self.k, rank)
        
        row, col, val = adj.coo()
        c = mis.sum()
        
        adj = SparseTensor(row=cluster[row], col=cluster[col],
                           val=val, size=(c, c)).coalesce(self.reduce_edge)
        
        if self.remove_self_loops:
            adj = adj.remove_diag()
        
        if torch.is_tensor(edge_index):
            row, col, edge_attr = adj.coo()
            edge_index = torch.stack([row, col])
        else:
            edge_index, edge_attr = adj, None
        
        if self.adaptive:
            x = x*score
        
        if self.reduce_x is None:
            x = x[mis]
        else:
            x = scatter(x, cluster, dim=0, dim_size=c, reduce=self.reduce_x)
        
        if batch is not None:
            batch = batch[mis]
        
        return x, edge_index, edge_attr, batch
