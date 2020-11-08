import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Batch, Data
from torch_geometric.typing import Tuple, OptTensor, OptPairTensor
from torch_sparse import SparseTensor

from miss import kernels, orderings, utils


class MISSPool(MessagePassing):
    def __init__(self, pool_size=1, stride=None, aggr='add', weighted_aggr=True, normalize=True,
                 add_self_loops=True, ordering='random', distances=False, kernel=None):
        super(MISSPool, self).__init__(aggr=aggr)

        self.pool_size = pool_size
        self.stride = stride if stride else pool_size
        self.weighted_aggr = weighted_aggr
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.ordering = self._get_ordering(ordering)
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

        if tokens[0] in {'min', 'max'}:
            opts['descending'] = tokens[0] == 'max'
            tokens = tokens[1:]

        if tokens[-1] == 'paths':
            opts['k'] = int(tokens[0])
            tokens[0] = 'k'

        return getattr(orderings, ''.join(t.title() for t in tokens))(**opts)

    def pool(self, adj: SparseTensor, mis: Tensor,
             x: OptTensor = None, pos: OptTensor = None) -> OptPairTensor:
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

    def forward(self, data: Data):
        x, pos, n = data.x, data.pos, data.num_nodes
        (row, col), val = data.edge_index, data.edge_attr

        if val is None:
            val = torch.ones_like(row, dtype=torch.float)

        adj = SparseTensor(row=row, col=col, value=val, sparse_sizes=(n, n))

        if self.add_self_loops:
            adj = adj.fill_diag(int(not self.distances))

        if self.normalize:
            deg = adj.sum(-1).unsqueeze(-1)
            adj *= torch.where(deg == 0, torch.zeros_like(deg), 1. / deg)

        perm = None if self.ordering is None else self.ordering(x, adj)
        mis = utils.maximal_k_independent_set(adj, self.stride, perm)

        x_out, pos_out = self.pool(adj, mis, x, pos)
        r_out, c_out, v_out = self.coarsen(adj, mis).coo()
        batch_out = data['batch'][mis] if 'batch' in data else None

        return Batch(x=x_out, pos=pos_out,
                     edge_index=torch.stack((r_out, c_out)),
                     edge_attr=v_out,
                     batch=batch_out)

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:  # noqa
        if not self.weighted_aggr:
            adj_t = adj_t.fill_value(1.)

        return adj_t.matmul(x, reduce=self.aggr)
