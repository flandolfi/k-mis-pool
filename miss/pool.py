import torch

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, Tensor, Tuple, OptTensor, Size
from torch_sparse import SparseTensor

from miss import kernels, orderings, utils


class MISSPool(MessagePassing):
    propagate_type = {'x': Tensor}

    def __init__(self, pool_size=1, stride=None, aggr='mean', ordering='random',
                 add_self_loops=True, normalize=False, distances=False, kernel=None):
        super(MISSPool, self).__init__(aggr=aggr)

        self.pool_size = pool_size
        self.stride = stride if stride else pool_size
        self.add_self_loops = add_self_loops
        self.ordering: orderings.Ordering = self._get_ordering(ordering)
        self.normalize = normalize
        self.distances = distances
        self.kernel = kernel

        if isinstance(kernel, str):
            self.kernel = getattr(kernels, kernel.title().replace('-', ''))()

        if self.distances:
            self._mat_mul = utils.sparse_min_sum_mm
        else:
            self._mat_mul = SparseTensor.matmul

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

    def pool(self, adj: SparseTensor, mis: Tensor,
             x: OptTensor = None, pos: OptTensor = None) -> Tuple[OptTensor, OptTensor]:
        if self.kernel is not None:
            adj = adj.set_value(self.kernel(adj.storage.value()))

        if x is not None:
            for _ in range(self.pool_size):
                x = self.propagate(edge_index=adj, x=x)

            x = x[mis]

        if pos is not None:
            for _ in range(self.pool_size):
                pos = self.propagate(edge_index=adj, x=pos)

            pos = pos[mis]

        return x, pos

    def coarsen(self, adj: SparseTensor, mis: Tensor) -> SparseTensor:
        adj_s = adj[mis]

        for _ in range(1, self.stride):
            adj_s = self._mat_mul(adj_s, adj)

        return self._mat_mul(self._mat_mul(adj_s, adj), adj_s.t())

    def forward(self, x: OptTensor, edge_index: Adj,
                edge_attr: OptTensor = None,
                pos: OptTensor = None,
                batch: OptTensor = None,
                size: Size = None) -> Tuple[OptTensor, Adj, OptTensor, OptTensor]:
        adj: Adj = edge_index
        
        if isinstance(adj, Tensor):
            if size is None:
                if x is not None:
                    n = x.size(0)
                elif pos is not None:
                    n = pos.size(0)
                elif batch is not None:
                    n = batch.size(0)
                else:
                    n = int(edge_index.max()) + 1
                size = (n, n)
                
            adj = SparseTensor.from_edge_index(edge_index, edge_attr, size)
        
        if self.add_self_loops:
            adj = adj.fill_diag(int(not self.distances))
        
        if self.normalize:
            deg = adj.sum(-1).unsqueeze(-1)
            adj *= torch.where(deg == 0, torch.zeros_like(deg), 1. / deg)
        
        perm = None if self.ordering is None else self.ordering(x, adj)
        mis = utils.maximal_k_independent_set(adj, self.stride, perm)
        
        x, pos = self.pool(adj, mis, x, pos)
        adj = self.coarsen(adj, mis)
        
        if batch is not None:
            batch = batch[mis]
        
        return x, adj, pos, batch

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:  # noqa
        return adj_t.matmul(x, reduce=self.aggr)
