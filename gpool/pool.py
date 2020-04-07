import torch
from abc import ABC, abstractmethod
from torch_scatter import scatter_min
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Batch
from gpool import orderings, utils


class _Pool(ABC, torch.nn.Module):
    def __init__(self, pool_size=1, stride=None, ordering='random', aggr='add', add_loops=True, cached=False):
        self.pool_size = pool_size
        self.stride = stride if stride else pool_size + 1
        self.aggr = aggr
        self.add_loops = add_loops
        self.ordering = self._get_ordering(ordering)

        self.cache = {True: None, False: None}

        if ordering != 'random':
            if cached == 'both' or cached == 'train':
                self.cache[True] = {}
            elif cached == 'both' or cached == 'test':
                self.cache[False] = {}

        super(_Pool, self).__init__()

    @property
    def cached(self):
        return self.cache[self.training] is not None

    @abstractmethod
    def _summarize(self, data):
        raise NotImplementedError

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

    def _get_ordering(self, ordering):
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
            k = tokens[-2]

            if k in {'d', 'diameter'}:
                return orderings.DPaths(**opts)
            if k in {'s', 'stride'}:
                opts['k'] = self.stride
            elif k in {'p', 'pool', 'pooling'}:
                opts['k'] = self.pool_size
            else:
                opts['k'] = int(k)

            return orderings.KPaths(**opts)

        return getattr(orderings, ''.join(t.title() for t in tokens))(**opts)


class SparsePool(MessagePassing, _Pool):
    def __init__(self, pool_size=1, stride=None, ordering='random', aggr='add', add_loops=True, cached=False):
        MessagePassing.__init__(self, aggr=aggr)
        _Pool.__init__(self, pool_size, stride, ordering, aggr, add_loops, cached)

    def _summarize(self, data):
        pass

    def _is_same_data(self, data, cache):
        return data.edge_index.equal(cache['edge_index'])

    def select(self, data: Batch):
        device = data.x.device
        selected = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        available = torch.ones_like(selected)
        frontier = available.clone()
        neighbors = selected.clone()

        order = self.ordering(data)
        index = torch.arange(data.num_nodes, device=device)
        batch = data['batch']

        while available.any():
            av_order, av_index = order[frontier], index[frontier]

            if batch is None:
                idx = torch.argmin(av_order, dim=0)
            else:
                av_batch = batch[frontier]
                idx = scatter_min(av_order, av_batch, dim=0, dim_size=data.num_graphs)[1]
                idx = idx[idx < av_index.size(0)]

            if frontier.all():
                frontier[:] = False

            idx = av_index[idx]
            neighbors[:] = False
            neighbors[idx], selected[idx], available[idx] = True, True, False
            row, col = data.edge_index

            for _ in range(self.stride - 1):
                mask = ~neighbors[col]
                row, col = row[mask], col[mask]
                neighbors[col[neighbors[row]]] = True

            available &= ~neighbors
            mask = available[col]
            row, col = row[mask], col[mask]
            frontier[col[neighbors[row]]] = True
            frontier &= available

            if not frontier.any():
                frontier = available.clone()

        return selected

    def pool(self, x, edge_index, edge_attr, mask, pos=None):
        if pos is not None:
            x = torch.cat([x, pos], dim=-1)

        for _ in range(self.pool_size):
            if self.add_loops:
                x += self.propagate(edge_index=edge_index, edge_attr=edge_attr, x=x)
            else:
                x = self.propagate(edge_index=edge_index, edge_attr=edge_attr, x=x)

        x = x[mask]

        if pos is not None:
            return x[:, :-pos.size(-1)], x[:, -pos.size(-1):]

        return x, None

    def forward(self, data: Batch):
        cache = self._maybe_cache(data, 'out', 'edge_index', 'edge_attr', 'mask')

        if cache is not None:
            out, edge_index, edge_attr, mask = cache
            out.x, _ = self.pool(data.x, edge_index, edge_attr, mask)

            return out

        device = data.x.device
        mask = self.select(data)
        labels = torch.empty(data.num_nodes, dtype=torch.long, device=device)
        labels[mask] = torch.arange(mask.sum().item(), dtype=torch.long, device=device)
        edge_index, edge_attr = data.edge_index, data.edge_attr
        out = data.clone()

        if edge_attr is None:
            edge_attr = torch.ones_like(edge_index[0], dtype=torch.float)

        indices, out.edge_attr = utils.k_hop(edge_index, edge_attr, self.stride, out.num_nodes, mask)
        out.edge_index, out.batch, out.num_nodes = labels[indices], out.batch[mask], mask.sum().item()
        out.x, out.pos = self.pool(data.x, edge_index, edge_attr, mask, out.pos)

        if self.cached:
            self._set_cache(out=out, edge_index=edge_index, edge_attr=edge_attr, mask=mask)

        return out

    def message(self, x_j, edge_attr):
        return x_j * edge_attr.view((-1, 1))


class DensePool(_Pool):
    def __init__(self, pool_size=1, stride=None, ordering='random', aggr='add', add_loops=True, cached=False):
        _Pool.__init__(self, pool_size, stride, ordering, aggr, add_loops, cached)

        self.__pool_op__ = getattr(self, aggr + '_pool')

    def _summarize(self, data):
        return data.adj[::max(2, data.adj.size(0) // 10)].sum().item()

    def _is_same_data(self, data, cache):
        return cache['adj'] == self._summarize(data)

    def pool(self, x, adj, mask, pos=None):
        x = self.__pool_op__(x, adj)
        x *= mask.unsqueeze(-1)

        if pos is not None:
            pos = self.mean_pool(pos, adj)
            pos *= mask.unsqueeze(-1)

        return x, pos

    def forward(self, data: Batch):
        cache = self._maybe_cache(data, 'out', 'adj_kernel', 'mask')

        if cache is not None:
            out, adj_kernel, mask = cache
            out.x, _ = self.pool(data.x, adj_kernel, mask)

            return out

        device = data.x.device
        out = data.clone()

        if out.adj.dim() < 3:
            for key in out.keys:
                out[key].unsqueeze_(0)

        adj = out.adj
        adj_k = {}

        if self.add_loops:
            adj += torch.eye(adj.size(1), dtype=torch.float, device=device).unsqueeze(0).expand_as(adj)
            adj[~out.mask] = 0.

        if self.stride == 1 or self.pool_size == 0:
            adj_k[0] = torch.eye(adj.size(1), dtype=torch.float, device=device).unsqueeze(0).expand_as(adj)
            adj_k[0][~out.mask] = 0.

        if self.stride == 2 or self.pool_size == 1:
            adj_k = {1: adj.clone()}

        for p in range(2, max(self.stride, self.pool_size) + 1):
            adj @= data.adj

            if p in {self.pool_size, self.stride - 1, self.stride}:
                adj_k[p] = adj.clone()

        available = out.mask.clone()
        frontier = torch.zeros_like(available)
        selected = torch.zeros_like(available)

        excluded = adj_k[self.stride - 1].bool()
        included = adj_k[self.stride].bool()  # & ~excluded

        order = self.ordering(out)
        idx = torch.argmin(order, dim=-1).unsqueeze(-1)
        frontier.scatter_(-1, idx, True)

        while available.any():
            current = torch.zeros_like(available)
            current.scatter_(-1, idx, True)

            selected |= current
            available &= ~current & ~excluded[current]
            frontier |= included[current] | ~frontier.any(-1, keepdim=True)
            frontier &= available

            masked_order = order.masked_fill(~frontier, float('Inf'))
            masked_idx = torch.argmin(masked_order, dim=-1).unsqueeze(-1)
            fr_mask = frontier.any(-1)
            idx[fr_mask] = masked_idx[fr_mask]

        out.num_nodes = selected.sum(-1).max().item()
        mask, perm = torch.topk(selected.float(), out.num_nodes, dim=-1, sorted=False)

        adj_size = out.adj.size()
        adj_kernel = adj_k[self.pool_size]
        adj_kernel = adj_kernel.gather(2, perm.unsqueeze(-2).expand(-1, adj_size[1], -1, *adj_size[3:]))

        out.x, out.pos = self.pool(out.x, adj_kernel, mask, out.pos)
        out.adj = adj_k[self.stride]
        out.adj = out.adj.gather(2, perm.unsqueeze(-2).expand(-1, adj_size[1], -1, *adj_size[3:]))
        out.adj = out.adj.gather(1, perm.unsqueeze(-1).expand(-1, -1, *out.adj.size()[2:]))
        out.adj *= mask.unsqueeze(-2)
        out.adj *= mask.unsqueeze(-1)
        out.mask = mask.bool()

        if self.cached:
            self._set_cache(out=out, adj=self._summarize(data), adj_kernel=adj_kernel, mask=mask)

        return out

    @staticmethod
    def add_pool(x, adj):
        return adj.transpose(-1, -2) @ x

    @staticmethod
    def mean_pool(x, adj):
        return DensePool.add_pool(x, adj)/adj.sum(1)

    @staticmethod
    def max_pool(x, adj):
        return torch.max(x.unsqueeze(-2).expand(*adj.size(), -1) * adj.unsqueeze(-1), dim=-2)[0]
