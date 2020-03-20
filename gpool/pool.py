import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Batch
from gpool import orderings, utils


class SparsePool(MessagePassing):
    def __init__(self, kernel_size=1, stride=None, ordering='random', *args, **kwargs):
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size + 1

        if isinstance(ordering, str):
            opts = {'k': self.stride} if ordering.endswith('k-hop-degree') else {}
            ordering = getattr(orderings, ordering.title().replace('-', ''))(**opts)

        self.ordering = ordering

        super(SparsePool, self).__init__(*args, **kwargs)

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
                _, idx = utils.sparse_min(av_order, av_batch)

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

        for _ in range(self.kernel_size):
            out.x = self.propagate(edge_index=out.edge_index, edge_attr=out.edge_attr, x=out.x)

        indices, values = utils.k_hop(out.edge_index, out.edge_attr,
                                      self.stride, out.num_nodes, mask)
        out.edge_index, out.edge_attr, out.num_nodes = labels[indices], values, mask.sum().item()

        for key, val in data:
            if torch.is_tensor(val) and val.size(0) == data.num_nodes:
                out['key'] = val[mask]

        return out

    def message(self, x_j, edge_attr):
        return x_j * edge_attr.view((-1, 1))

