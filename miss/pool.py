import torch

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, Tensor, Tuple, OptTensor, Size, Union, Optional
from torch_sparse import SparseTensor

from miss import orderings, utils


class MISSPool(MessagePassing):
    propagate_type = {'x': Tensor}

    def __init__(self, pool_size=1, stride=None, ordering='random',
                 add_self_loops=True, normalize=True, weighted=True):
        super(MISSPool, self).__init__(aggr='add')

        self.pool_size = pool_size
        self.stride = stride if stride else pool_size
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

    def pool(self, p_mat: SparseTensor, *xs: OptTensor) -> Tuple[OptTensor, ...]:
        if self.pool_size <= 0:
            return xs

        if not self.weighted:
            p_mat.set_value_(torch.ones_like(p_mat.storage.value()), layout="coo")
            count = p_mat.sum(-1).unsqueeze(-1)
            p_mat *= torch.where(count == 0, torch.zeros_like(count), 1. / count)

        out = []

        for x in xs:
            if x is None:
                out.append(None)
            else:
                out.append(self.propagate(p_mat, x=x))  # noqa

        return tuple(out)

    def coarsen(self, s_mat: SparseTensor, adj: SparseTensor) -> SparseTensor:
        if self.stride <= 0:
            return adj

        if self.normalize:
            norm = s_mat.t().sum(-1).unsqueeze(-1)
            norm = torch.where(norm == 0, torch.ones_like(norm), 1. / norm)

            return (s_mat @ adj) @ (s_mat.t() * norm)

        return (s_mat @ adj) @ s_mat.t()

    @staticmethod
    def get_coarsening_matrices(adj: SparseTensor, mis: Tensor, p: int, s: int) -> Tuple[SparseTensor, SparseTensor]:
        if p > s:
            return MISSPool.get_coarsening_matrices(adj, mis, s, p)[::-1]

        p_mat = adj[mis]

        for _ in range(1, p):
            p_mat @= adj

        s_mat = p_mat.clone()

        for _ in range(p, s):
            s_mat @= adj

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
            deg = adj.sum(-1).unsqueeze(-1)
            adj *= torch.where(deg == 0, torch.zeros_like(deg), 1. / deg)

        return adj

    def _get_mis(self, adj: Adj, *xs: OptTensor) -> Tensor:
        x = None

        for x in xs:
            if x is not None:
                break

        perm = None if self.ordering is None else self.ordering(x, adj)
        return utils.maximal_k_independent_set(adj, self.stride, perm)

    def forward(self, edge_index: Adj, edge_attr: OptTensor = None,
                *xs: OptTensor, batch: OptTensor = None) -> Tuple[Union[SparseTensor, OptTensor], ...]:
        size = self._maybe_size(batch, *xs)
        adj = self._get_adj(edge_index, edge_attr, size)
        mis = self._get_mis(adj, *xs)
        p_mat, s_mat = self.get_coarsening_matrices(adj, mis, self.pool_size, self.stride)

        x = self.pool(p_mat, *xs)
        adj = self.coarsen(s_mat, adj)
        
        if batch is not None:
            batch = batch[mis]
        
        return adj, *x, batch

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:  # noqa
        return adj_t @ x
