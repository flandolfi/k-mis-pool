from abc import ABC, abstractmethod

import torch
import torch_sparse
import torch_scatter
from torch_geometric.data import Batch


class Ordering(ABC):
    def __init__(self, descending=False):
        self.descending = descending

    def __call__(self, data: Batch):
        out = self._compute(data)

        if self.descending:
            out *= -1

        if 'adj' in data and out.dim() != 2:
            out = out.view(-1, data.adj.size(1)).expand_as(data.mask).clone()
            out[~data.mask] = float('inf')

        return out

    @abstractmethod
    def _compute(self, data: Batch):
        raise NotImplementedError


class Canonical(Ordering):
    def _compute(self, data: Batch):
        size = data.adj.size(1) if 'adj' in data else data.num_nodes

        return torch.arange(size)


class InverseCanonical(Canonical):
    def __init__(self):
        super(InverseCanonical, self).__init__(True)


class Random(Ordering):
    def _compute(self, data: Batch):
        size = data.adj.size(1) if 'adj' in data else data.num_nodes

        return torch.randperm(size).float()


class KPaths(Ordering):
    def __init__(self, k=1, descending=False):
        super(KPaths, self).__init__(descending)
        self.k = k

    def _compute(self, data: Batch):
        if 'adj' in data:
            out = torch.ones_like(data.mask, dtype=torch.float)
            out.unsqueeze_(-1)

            for _ in range(self.k):
                out = data.adj @ out

            return out.squeeze(-1)

        out = torch.ones([data.num_nodes, 1], dtype=torch.float)
        ind, val, n = data.edge_index, data.edge_attr, data.num_nodes

        if val is None:
            val = torch.ones_like(ind[0], dtype=torch.float)

        for _ in range(self.k):
            out = torch_sparse.spmm(ind, val, n, n, out)

        return out.view(-1)


class Degree(KPaths):
    def __init__(self, descending=False):
        super(Degree, self).__init__(1, descending)


class DPaths(Ordering):
    def _compute(self, data: Batch):
        if 'adj' in data:
            adj = data.adj.clone()
            seen = adj.bool() + torch.eye(adj.size(1), dtype=torch.bool).unsqueeze(0)
            mask = torch.ones(adj.size(0), dtype=torch.bool)

            while mask.any():
                adj[mask] @= data.adj[mask]
                mask = (adj.bool() & ~seen).any(-1).any(-1)
                seen |= adj.bool()

            return adj.diagonal(dim1=-2, dim2=-1)

        if 'batch' in data:
            batch = data.batch
            num_graphs = data.num_graphs
            n = batch.size(0)
            count = torch.bincount(batch, minlength=num_graphs)
            max_nodes = count.max().item()
            limits = torch.zeros_like(count)
            limits[1:] = count.cumsum(0)[:-1]

            idx = torch.arange(n, dtype=torch.long) - limits[batch].long()
            out = torch.eye(max_nodes)[idx]
        else:
            n = data.num_nodes
            idx = torch.arange(n, dtype=torch.long)
            out = torch.eye(n, dtype=torch.float)
            batch = torch.zeros(n, dtype=torch.long)
            num_graphs = 1

        ind, val = data.edge_index, data.edge_attr
        seen = out.bool()
        mask = torch.ones(num_graphs, dtype=torch.bool)

        if val is None:
            val = torch.ones_like(ind[0], dtype=torch.float)

        while mask.any():
            out = torch_sparse.spmm(ind, val, n, n, out)
            changed = (out.bool() & ~seen).any(-1)
            seen |= out.bool()

            mask = torch_scatter.scatter_max(changed.int(), batch, dim_size=num_graphs)[0].bool()
            todo = mask[batch][ind[0]]
            ind = ind[:, todo]
            val = val[todo]

        return out.gather(1, idx.unsqueeze(1)).squeeze()
