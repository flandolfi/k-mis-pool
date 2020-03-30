import torch
from abc import ABC
from torch_scatter import scatter_min
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Batch
from gpool import orderings, utils


class _Pool(ABC):
    def __init__(self, kernel_size=1, stride=None, ordering='random', aggr='add'):
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size + 1
        self.aggr = aggr
        self.ordering = self._get_ordering(ordering)

    def _get_ordering(self, ordering):
        if callable(ordering):
            return ordering

        if not isinstance(ordering, str):
            raise ValueError(f"Expected string or callable, got {ordering} instead.")

        opts = {'descending': False}

        if ordering[:4] in {'min-', 'max-'}:
            opts['descending'] = ordering.startswith('max-')
            ordering = ordering[4:]

        if ordering.endswith('paths'):
            if ordering == 'd-paths':
                return orderings.DPaths(**opts)
            if ordering == 'k-paths':
                opts['k'] = self.stride
            else:
                opts['k'] = int(ordering[:-6])

            return orderings.KPaths(**opts)

        return getattr(orderings, ordering.title())(**opts)


class SparsePool(MessagePassing, _Pool):
    def __init__(self, kernel_size=1, stride=None, ordering='random', aggr='add', *args, **kwargs):
        _Pool.__init__(self, kernel_size, stride, ordering, aggr)
        MessagePassing.__init__(self, 'add' if aggr == 'mean' else aggr, *args, **kwargs)

        self.__aggr__ = aggr

    def select(self, data: Batch):
        selected = torch.zeros(data.num_nodes, dtype=torch.bool)
        available = torch.ones_like(selected)
        frontier = available.clone()
        neighbors = selected.clone()

        order = self.ordering(data)
        index = torch.arange(data.num_nodes)
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

    def forward(self, data: Batch):
        mask = self.select(data)
        labels = torch.empty(data.num_nodes, dtype=torch.long)
        labels[mask] = torch.arange(mask.sum().item(), dtype=torch.long)
        out = data.clone()

        if out.edge_attr is None:
            out.edge_attr = torch.ones_like(out.edge_index[0], dtype=torch.float)

        if self.__aggr__ == 'mean':
            out.x = torch.cat([out.x, torch.ones(out.x.size(0), 1)], dim=-1)

        for _ in range(self.kernel_size):
            out.x = self.propagate(edge_index=out.edge_index, edge_attr=out.edge_attr, x=out.x)

        indices, values = utils.k_hop(out.edge_index, out.edge_attr,
                                      self.stride, out.num_nodes, mask)
        out.edge_index, out.edge_attr, out.num_nodes = labels[indices], values, mask.sum().item()

        for key, val in out('x', 'pos', 'norm', 'batch'):
            out[key] = val[mask]

        if 'y' in out and out.y.size(0) == data.num_nodes:
            out.y = out.y.mask

        if self.__aggr__ == 'mean':
            out.x = out.x[:, :-1]/out.x[:, -1:]

        return out

    def message(self, x_j, edge_attr):
        return x_j * edge_attr.view((-1, 1))


class DensePool(torch.nn.Module, _Pool):
    def __init__(self, kernel_size=1, stride=None, ordering='random', aggr='add'):
        _Pool.__init__(self, kernel_size, stride, ordering, aggr)
        torch.nn.Module.__init__(self)

        self.__aggr_op__ = getattr(self, aggr + '_pool')

    def forward(self, data: Batch):
        out = data.clone()

        if out.adj.dim() < 3:
            for key in out.keys:
                out[key].unsqueeze_(0)

        adj = out.adj
        adj_k = {1: out.adj.clone()}

        for p in range(2, max(self.stride, self.kernel_size) + 1):
            adj = adj @ adj

            if p in {self.kernel_size, self.stride - 1, self.stride}:
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

        out.x = self.__aggr_op__(out.x, adj_k[self.kernel_size])
        out.adj = adj_k[self.stride]
        adj_size = out.adj.size()
        out.adj = out.adj.gather(2, perm.unsqueeze(-2).expand(-1, adj_size[1], -1, *adj_size[3:]))
        out.adj *= mask.unsqueeze(-2)
        out.mask = mask.bool()

        for key, val in out:
            if key != 'mask' and val.size()[:2] == adj_size[:2]:
                if val.dim() == 2:
                    out[key] = val.gather(1, perm) * mask
                else:
                    out[key] = val.gather(1, perm.unsqueeze(-1).expand(-1, -1, *val.size()[2:]))
                    out[key] *= mask.unsqueeze(-1)

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
