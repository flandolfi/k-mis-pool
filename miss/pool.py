import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Batch, Data
from torch_geometric.typing import Tuple, OptTensor, OptPairTensor
from torch_sparse import SparseTensor

from miss import kernels, orderings, utils


class MISSPool(MessagePassing):
    def __init__(self, pool_size=1, stride=None, aggr='mean', weighted_aggr=False, add_self_loops=True,
                 ordering='min-curvature', order_on='stride', distances=False, kernel=None):
        super(MISSPool, self).__init__()

        self.pool_size = pool_size
        self.stride = stride if stride else pool_size
        self.normalize = weighted_aggr and aggr == 'mean'
        self.aggr = 'add' if self.normalize else aggr
        self.weighted_aggr = weighted_aggr
        self.add_self_loops = add_self_loops
        self.ordering = self._get_ordering(ordering)
        self.order_on = order_on
        self.distances = distances
        self.kernel = kernel

        if isinstance(kernel, str):
            self.kernel = getattr(kernels, kernel.title().replace('-', ''))()

        if self.distances:
            self.fuse = False
            self._mat_mul = utils.sparse_min_sum_mm
        else:
            self._mat_mul = SparseTensor.matmul

            if self.aggr == 'max':
                self.fuse = False

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

        return getattr(orderings, ''.join(t.title() for t in tokens))(**opts)

    def _compute_paths(self, adj: SparseTensor, p: int, s: int) -> Tuple[SparseTensor, SparseTensor]:
        if p < s:
            return self._compute_paths(adj, s, p)[::-1]

        if p == s:
            adj_pow = utils.sparse_matrix_power(adj, p, self.distances)
            return adj_pow, adj_pow.clone()

        adj_p, adj_s = self._compute_paths(adj, p - s, s // 2)
        adj_s = self._mat_mul(adj_s, adj_s)

        if s % 2 == 1:
            adj_s = self._mat_mul(adj_s, adj)

        return self._mat_mul(adj_p, adj_s), adj_s

    def pool(self, adj: SparseTensor, x: OptTensor = None, pos: OptTensor = None) -> OptPairTensor:
        if x is not None:
            x = self.propagate(edge_index=adj, x=x)

        if pos is not None:
            pos = self.propagate(edge_index=adj, x=pos)

        return x, pos

    def coarsen(self, adj: SparseTensor, adj_s: SparseTensor) -> SparseTensor:
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

        adj_p, adj_s = self._compute_paths(adj, self.pool_size, self.stride)

        if self.kernel is not None:
            adj_p = adj_p.set_value(self.kernel(adj_p.storage.value()))

        perm = None

        if self.ordering is not None:
            if self.order_on == "pool":
                perm = self.ordering(x, adj_p)
            elif self.order_on == "stride":
                perm = self.ordering(x, adj_s)
            else:
                perm = self.ordering(x, adj)

        mask = utils.maximal_independent_set(adj_s, perm)
        adj_p = adj_p[mask]
        adj_s = adj_s[mask]

        x_out, pos_out = self.pool(adj_p, x, pos)
        r_out, c_out, v_out = self.coarsen(adj, adj_s).coo()
        batch_out = data['batch'][mask] if 'batch' in data else None

        return Batch(x=x_out, pos=pos_out,
                     edge_index=torch.stack((r_out, c_out)),
                     edge_attr=v_out,
                     batch=batch_out)

    def message(self, x_j, edge_attr):  # noqa
        return x_j * edge_attr.view((-1, 1)) if self.weighted_aggr else x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:  # noqa
        return adj_t @ x
