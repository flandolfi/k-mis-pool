import torch

from torch_geometric.typing import Adj, Tensor, Tuple, OptTensor, Size, Union, Optional, OptPairTensor
from torch_sparse import SparseTensor
from torch_scatter import scatter_min

from kmis import orderings, sample, utils


def get_coarsening_matrix(adj: SparseTensor, k: int = 1, eps: float = 0.5,
                          rank: OptTensor = None) -> Tuple[SparseTensor, Tensor]:
    mis = sample.maximal_k_independent_set(adj, k, rank)
    rw = utils.normalize_dim(adj)
    
    if eps > 0.:
        val = (1 - eps) * rw.storage.value()
        rw = rw.set_value(val, layout="coo")
        diag = rw.get_diag() + eps
        rw = rw.set_diag(diag)
    
    c_mat = rw.eye(adj.size(0), device=adj.device())[mis]
    
    for _ in range(k):
        c_mat = c_mat @ rw
    
    return utils.normalize_dim(c_mat.t()), mis


def sample_partition_matrix(c_mat: SparseTensor) -> SparseTensor:
    cluster = sample.sample_multinomial(c_mat)
    n, s = c_mat.sparse_sizes()
    device = c_mat.device()
    return SparseTensor(row=torch.arange(n, dtype=torch.long, device=device),
                        col=cluster,
                        value=torch.ones_like(cluster, dtype=torch.float),
                        sparse_sizes=(n, s), is_sorted=True)


@torch.jit.script
def cluster_k_mis(adj: SparseTensor, k: int = 1, rank: OptTensor = None) -> Tuple[Tensor, Tensor]:
    n, device = adj.size(0), adj.device()
    row, col, val = adj.coo()

    if rank is None:
        rank = torch.arange(n, dtype=torch.long, device=device)

    mis = sample.maximal_k_independent_set(adj, k, rank)
    min_rank = torch.full((n,), fill_value=n, dtype=torch.long, device=device)
    min_rank[mis] = rank[mis]

    for _ in range(k):
        scatter_min(min_rank[row], col, out=min_rank)

    _, clusters = torch.unique(min_rank, return_inverse=True)
    return clusters, mis


class KMISCoarsening(torch.nn.Module):
    def __init__(self, k=1, ordering='random', eps=0.5, sample_partition=True):
        super(KMISCoarsening, self).__init__()

        self.k = k
        self.eps = eps
        self.ordering: orderings.Ordering = self._get_ordering(ordering)
        self.sample_partition = sample_partition

    def _get_ordering(self, ordering):
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

        if tokens[-1] in {'paths', 'curvature', 'walk'}:
            opts['k'] = self.k

        cls_name = ''.join(t.title() for t in tokens)

        if not hasattr(orderings, cls_name):
            return orderings.Lambda(getattr(torch, tokens[-1]), **opts)

        return getattr(orderings, cls_name)(**opts)

    @staticmethod
    def pool(l_mat: Union[Tensor, SparseTensor], *xs: OptTensor) -> Tuple[OptTensor, ...]:
        out = []

        for x in xs:
            if x is None:
                out.append(None)
            else:
                out.append(l_mat @ x)
                
        return tuple(out)

    def coarsen(self, c_mat: Union[Tensor, SparseTensor], adj: SparseTensor) -> Union[Tensor, SparseTensor]:
        out = c_mat.t() @ (adj @ c_mat)
        
        if self.sample_partition:
            return out

        diag = adj.get_diag()
        
        if torch.all(diag == 0.):
            return out
        
        n, s, device = adj.size(0), out.size(0), adj.device()
        sigma = c_mat.t() @ (SparseTensor.eye(n, device=device).set_diag(-diag) @ c_mat)
        sigma_diag = c_mat.t() @ diag.unsqueeze(-1)
        sigma_diag = SparseTensor.eye(s, device=device).set_diag(sigma_diag.squeeze(-1))
        
        rows, cols, values = tuple(zip(out.coo(), sigma.coo(), sigma_diag.coo()))
        return SparseTensor(row=torch.cat(rows),
                            col=torch.cat(cols),
                            value=torch.cat(values),
                            sparse_sizes=(s, s)).coalesce('sum')

    @staticmethod
    def _maybe_size(*xs: OptTensor) -> Optional[Size]:
        for x in xs:
            if x is not None:
                n = x.size(0)
                return n, n

        return None

    def _get_rank(self, x: OptTensor, adj: Adj) -> OptTensor:
        if self.ordering is None:
            return None

        return self.ordering(x, adj)

    def get_coarsening_matrices(self, adj: SparseTensor, x: OptTensor = None) \
            -> Tuple[SparseTensor, SparseTensor, Tensor]:
        rank = self._get_rank(x, adj)
        c_mat, mis = get_coarsening_matrix(adj, self.k, self.eps, rank)

        if self.sample_partition:
            c_mat = sample_partition_matrix(c_mat)
        
        l_mat = utils.normalize_dim(c_mat.t())

        return c_mat, l_mat, mis

    def forward(self, x: Union[OptTensor, OptPairTensor],
                edge_index: Adj, edge_attr: OptTensor = None,
                batch: OptTensor = None) -> Tuple[Union[OptTensor, OptPairTensor], SparseTensor,
                                                  OptTensor, Tensor, SparseTensor, SparseTensor]:
        if not isinstance(x, tuple):
            x: Tuple[OptTensor] = (x,)

        size = self._maybe_size(batch, *x)
        adj = edge_index

        if isinstance(adj, Tensor):
            adj = SparseTensor.from_edge_index(edge_index, edge_attr, size)

        c_mat, l_mat, mis = self.get_coarsening_matrices(adj, x[0])

        adj = self.coarsen(c_mat, adj)
        out = self.pool(l_mat, *x)

        if len(out) == 1:
            out = out[0]
        
        if batch is not None:
            batch = batch[mis]
        
        return out, adj, batch, mis, c_mat, l_mat
