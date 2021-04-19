import torch

from torch_geometric.typing import Adj, Tensor, Tuple, OptTensor, Size, Union, Optional
from torch_sparse import SparseTensor

from kmis import orderings, sample, utils


@torch.jit.script
def get_coarsening_matrix(adj: SparseTensor, k: int = 1, eps: float = 0.5,
                          rank: OptTensor = None) -> Tuple[SparseTensor, Tensor]:
    mis = sample.maximal_k_independent_set(adj, k, rank)
    rw = utils.normalize_dim(adj)
    
    if eps > 0.:
        diag = rw.get_diag() + eps
        rw = rw.set_value(rw.storage.value() * (1 - eps))
        rw = rw.set_diag(diag)
    
    c_mat = rw.eye(adj.size(0), device=adj.device())[:, mis]
    
    for _ in range(k):
        c_mat = rw @ c_mat
    
    return utils.normalize_dim(c_mat), mis


@torch.jit.script
def sample_partition_matrix(c_mat: SparseTensor) -> SparseTensor:
    cluster = utils.sample_multinomial(c_mat)
    n, s = c_mat.sparse_sizes()
    device = c_mat.device()
    return SparseTensor(row=torch.arange(n, dtype=torch.long, device=device),
                        col=cluster,
                        value=torch.ones_like(cluster, dtype=torch.float),
                        sparse_sizes=(n, s), is_sorted=True)


class KMISCoarsening(torch.nn.Module):
    def __init__(self, k=1, ordering='random', eps=0.5,
                 normalize=True, sample_partition=True):
        super(KMISCoarsening, self).__init__()

        self.k = k
        self.eps = eps
        self.ordering: orderings.Ordering = self._get_ordering(ordering)
        self.normalize = normalize
        self.sample_partition = sample_partition

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

    @staticmethod
    def pool(p_inv: Union[Tensor, SparseTensor], *xs: OptTensor) -> Tuple[OptTensor, ...]:
        out = []

        for x in xs:
            if x is None:
                out.append(None)
            else:
                out.append(p_inv @ x)
                
        return tuple(out)

    @staticmethod
    def coarsen(c_mat: SparseTensor, adj: SparseTensor) -> SparseTensor:
        return (c_mat.t() @ adj) @ c_mat

    @staticmethod
    def _maybe_size(*xs: OptTensor) -> Optional[Size]:
        for x in xs:
            if x is not None:
                n = x.size(0)
                return n, n

        return None

    def _get_rank(self, adj: Adj, *xs: OptTensor) -> OptTensor:
        if self.ordering is None:
            return None
        
        x = None

        for x in xs:
            if x is not None:
                break

        return self.ordering(x, adj)

    def forward(self, edge_index: Adj, edge_attr: OptTensor = None,
                *xs: OptTensor, batch: OptTensor = None) -> Tuple[Union[SparseTensor, OptTensor], ...]:
        size = self._maybe_size(batch, *xs)
        adj = edge_index

        if isinstance(adj, Tensor):
            adj = SparseTensor.from_edge_index(edge_index, edge_attr, size)

        rank = self._get_rank(adj, *xs)
        c_mat, mis = get_coarsening_matrix(adj, self.k, self.eps, rank)
        
        if self.sample_partition:
            c_mat = sample_partition_matrix(c_mat)
            p_inv = utils.normalize_dim(c_mat.t())
        else:
            p_inv = c_mat.to_dense()
            p_inv = torch.pinverse(p_inv)

        adj = self.coarsen(c_mat, adj)
        out = self.pool(p_inv, *xs)
        
        if batch is not None:
            batch = batch[mis]
        
        return adj, c_mat, p_inv, mis, *out, batch
