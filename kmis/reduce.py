import torch

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, Tensor, Tuple, OptTensor, Size, Union, Optional
from torch_sparse import SparseTensor

from kmis import orderings, sample, utils


def get_coarsening_matrix(adj: SparseTensor, mis: Tensor, k: int) -> SparseTensor:
    c_mat = adj.eye(adj.size(0), device=adj.device())[mis]
    
    for _ in range(k):
        c_mat = c_mat @ adj
    
    return c_mat


class KMISPool(MessagePassing):
    propagate_type = {'x': Tensor}

    def __init__(self, k=1, ordering='random', add_self_loops=False,
                 normalize=True, aggr='add', weighted=True):
        super(KMISPool, self).__init__(aggr=aggr)

        self.k = k
        self.add_self_loops = add_self_loops
        self.ordering: orderings.Ordering = self._get_ordering(ordering)
        self.normalize = normalize
        self.weighted = weighted

    @staticmethod
    def _get_ordering(ordering):
        if ordering is None:
            return None

        if callable(ordering):
            return ordering

        if not isinstance(ordering, str):
            raise ValueError(f"Expected string or callable, got {ordering} instead.")

        tokens = ordering.split('-')
        opts = {'descending': True}

        if len(tokens) > 1 and tokens[0] in {'min', 'max'}:
            opts['descending'] = tokens[0] == 'max'
            tokens = tokens[1:]

        if tokens[-1] == 'paths':
            opts['k'] = int(tokens[0])
            tokens[0] = 'k'

        cls_name = ''.join(t.title() for t in tokens)

        if not hasattr(orderings, cls_name):
            return orderings.Lambda(getattr(torch, tokens[-1]), **opts)

        return getattr(orderings, cls_name)(**opts)

    def pool(self, c_mat: SparseTensor, *xs: OptTensor) -> Tuple[OptTensor, ...]:
        if not self.weighted:
            c_mat = c_mat.set_value(torch.ones_like(c_mat.storage.value()), layout="coo")
        elif self.normalize:
            c_mat = utils.normalize_dim(c_mat, 0)

        out = []

        for x in xs:
            if x is None:
                out.append(None)
            else:
                out.append(self.propagate(c_mat, x=x))
                
        return tuple(out)

    def unpool(self, p_mat: SparseTensor, *xs: OptTensor) -> Tuple[OptTensor, ...]:
        if not self.weighted:
            p_mat = p_mat.set_value(torch.ones_like(p_mat.storage.value()), layout="coo")

        out = []

        for x in xs:
            if x is None:
                out.append(None)
            else:
                out.append(self.propagate(p_mat.t(), x=x))
                
        return tuple(out)

    def coarsen(self, s_mat: SparseTensor, adj: SparseTensor) -> SparseTensor:
        if self.normalize:
            return (s_mat @ adj) @ utils.normalize_dim(s_mat, 0).t()

        return (s_mat @ adj) @ s_mat.t()

    def expand(self, s_mat: SparseTensor, adj: SparseTensor) -> SparseTensor:
        if self.normalize:
            return utils.normalize_dim(s_mat, 0).t() @ (adj @ s_mat)

        return s_mat.t() @ (adj @ s_mat)

    @staticmethod
    def get_coarsening_matrices(adj: SparseTensor, mis: Tensor, p: int, s: int) -> Tuple[SparseTensor, SparseTensor]:
        if p > s:
            return KMISPool.get_coarsening_matrices(adj, mis, s, p)[::-1]

        p_mat = adj.eye(adj.size(0), device=adj.device())[mis]

        for _ in range(p):
            p_mat = p_mat @ adj

        s_mat = p_mat.clone()

        for _ in range(p, s):
            s_mat = s_mat @ adj

        return p_mat, s_mat

    @staticmethod
    def _maybe_size(*xs: OptTensor) -> Optional[Size]:
        for x in xs:
            if x is not None:
                n = x.size(0)
                return n, n

        return None

    def _get_adj(self, edge_index: Adj, edge_attr: OptTensor = None, size: Size = None) -> Adj:
        adj: Adj = edge_index

        if isinstance(adj, Tensor):
            adj = SparseTensor.from_edge_index(edge_index, edge_attr, size)

        if self.add_self_loops:
            adj = adj.fill_diag(1.)

        if self.normalize:
            adj = utils.normalize_dim(adj, -1)

        return adj

    def _get_mis(self, adj: Adj, *xs: OptTensor) -> Tensor:
        x = None

        for x in xs:
            if x is not None:
                break

        perm = None if self.ordering is None else self.ordering(x, adj)
        return sample.maximal_k_independent_set(adj, self.stride, perm)

    def forward(self, edge_index: Adj, edge_attr: OptTensor = None,
                *xs: OptTensor, batch: OptTensor = None) -> Tuple[Union[SparseTensor, OptTensor], ...]:
        size = self._maybe_size(batch, *xs)
        adj = self._get_adj(edge_index, edge_attr, size)

        if self.training or not self.ensemble:
            mis = self._get_mis(adj, *xs)
        elif self.ensemble == 'all':
            mis = torch.ones(adj.size(0), dtype=torch.bool, device=adj.device())
        else:
            mis = torch.zeros(adj.size(0), dtype=torch.bool, device=adj.device())

            for _ in range(self.test_iterations):
                mis = mis | self._get_mis(adj, *xs)

        p_mat, s_mat = self.get_coarsening_matrices(adj, mis, self.pool_size, self.stride)

        x = self.pool(p_mat, *xs)
        adj = self.coarsen(s_mat, adj)
        
        if batch is not None:
            batch = batch[mis]
        
        return adj, p_mat, s_mat, mis, *x, batch

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:  # noqa
        return adj_t.matmul(x, reduce=self.aggr)
