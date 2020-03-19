import torch
import torch_geometric as pyg
from torch_geometric.nn.conv import MessagePassing
from gpool import orderings, utils


class SparsePool(MessagePassing):
    def __init__(self, pool_size=1, stride=None, ordering='random', *args, **kwargs):
        self.pool_size = pool_size
        self.stride = stride if stride else pool_size + 1

        if isinstance(ordering, str):
            ordering = getattr(orderings, ordering.title().replace('-', ''))()

        self.ordering = ordering

        super(SparsePool, self).__init__(*args, **kwargs)

    def select(self, data: pyg.data.Data):
        selected = torch.zeros(data.num_nodes, dtype=torch.uint8)
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

    def aggregate(self, data: pyg.data.Batch, mask: torch.ByteTensor):
        pass











