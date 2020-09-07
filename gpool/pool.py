from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Batch, Data
from torch_sparse import SparseTensor

from gpool import kernels, orderings, utils


class _Pool(ABC, torch.nn.Module):
    def __init__(self, pool_size=1, stride=None, aggr='add', weighted_aggr=True,
                 ordering=None, order_on='raw', distances=False, kernel=None, cached=False):
        super(_Pool, self).__init__()

        self.pool_size = pool_size
        self.stride = stride if stride else pool_size
        self.aggr = aggr
        self.weighted_aggr = weighted_aggr
        self.ordering = self._get_ordering(ordering)
        self.order_on = order_on
        self.distances = distances
        self.kernel = kernel

        if isinstance(kernel, str):
            self.kernel = getattr(kernels, kernel.title().replace('-', ''))()

        self.cache = {True: None, False: None}

        if cached and isinstance(self.ordering, orderings.Ordering) and self.ordering.cacheable:
            if cached == 'train' or cached == 'both':
                self.cache[True] = {}
            if cached == 'test' or cached == 'both':
                self.cache[False] = {}

    @property
    def cached(self):
        return self.cache[self.training] is not None

    @abstractmethod
    def _is_same_data(self, data, cache):
        raise NotImplementedError

    def _set_cache(self, **kwargs):
        self.cache[self.training] = kwargs

    def _maybe_cache(self, data, *keys):
        cache = self.cache[self.training]

        if bool(cache) and self._is_same_data(data, cache):
            return (cache[k] for k in keys)

        return None

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


class MISPool(_Pool, MessagePassing):  # noqa
    def _is_same_data(self, data, cache):
        return data.edge_index.equal(cache['edge_index'])

    @staticmethod
    def _compute_paths(adj: SparseTensor, p: int, s: int) -> Tuple[SparseTensor, SparseTensor]:
        if p < s:
            return MISPool._compute_paths(adj, s, p)[::-1]

        if p == s:
            adj_pow = utils.sparse_matrix_power(adj, p)
            return adj_pow, adj_pow

        adj_p, adj_s = MISPool._compute_paths(adj, p - s, s//2)
        adj_s @= adj_s

        if s % 2 == 1:
            adj_s @= adj

        return adj_p @ adj_s, adj_s

    @staticmethod
    def _compute_distances(adj: SparseTensor, p: float, s: float) -> Tuple[SparseTensor, SparseTensor]:
        if p < s:
            return MISPool._compute_distances(adj, s, p)[::-1]

        dist_p = utils.pairwise_distances(adj, p)

        if p == s:
            return dist_p, dist_p

        row, col, val = dist_p.coo()
        mask = val <= s
        dist_s = SparseTensor(row=row[mask], col=col[mask], value=val[mask],
                              sparse_sizes=adj.sparse_sizes(), is_sorted=True)

        return dist_p, dist_s

    def pool(self, x, adj, pos=None):
        if pos is not None:
            x = torch.cat([x, pos], dim=-1)

        x = self.propagate(edge_index=adj, x=x)

        if pos is not None:
            return x[:, :-pos.size(-1)], x[:, -pos.size(-1):]

        return x, None

    def coarsen(self, adj, adj_s):
        if self.distances:
            adj_out = utils.sparse_min_sum_mm(utils.sparse_min_sum_mm(adj_s, adj), adj_s.t())
        else:
            adj_out = adj_s @ adj @ adj_s.t()

        return adj_out

    def forward(self, data: Data):
        cache = self._maybe_cache(data, 'out', 'adj')

        if cache is not None:
            out, adj = cache
            out.x, _ = self.pool(data.x, adj)

            return out

        x, pos, n = data.x, data.pos, data.num_nodes
        (row, col), val = data.edge_index, data.edge_attr

        if val is None:
            val = torch.ones_like(row, dtype=torch.float)

        adj = SparseTensor(row=row, col=col, value=val, sparse_sizes=(n, n), is_sorted=True)

        if self.distances:
            adj = adj.fill_diag(0.)
            adj_p, adj_s = self._compute_distances(adj, self.pool_size, self.stride)
        else:
            adj = adj.fill_diag(1.)
            adj_p, adj_s = self._compute_paths(adj, self.pool_size, self.stride)

        if self.kernel is not None:
            adj_p = adj_p.set_value(self.kernel(adj_p.storage.value()))

        perm = None

        if self.ordering is not None:
            if self.order_on == "raw":
                perm = self.ordering(x, adj)
            elif self.order_on == "pool":
                perm = self.ordering(x, adj_p)
            elif self.order_on == "stride":
                perm = self.ordering(x, adj_s)

        mask = utils.maximal_independent_set(adj_s, perm)
        adj_p = adj_p[mask]
        adj_s = adj_s[mask]

        x_out, pos_out = self.pool(x, adj_p, pos)
        r_out, c_out, v_out = self.coarsen(adj, adj_s).coo()
        batch_out = data['batch'][mask] if 'batch' in data else None

        out = Batch(x=x_out, pos=pos_out, edge_index=torch.stack((r_out, c_out)), edge_attr=v_out, batch=batch_out)

        if self.cached:
            self._set_cache(out=out, adj=adj_p)

        return out

    def message(self, x_j, edge_attr):  # noqa
        return x_j * edge_attr.view((-1, 1)) if self.weighted_aggr else x_j
