import torch

from torch_geometric.typing import Adj, Tensor, Tuple, OptTensor, Size, Union, Optional, OptPairTensor
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import SparseTensor
from torch_scatter import scatter_min

from kmis import orderings, sample, utils

OptPairSparseTensor = Union[SparseTensor, Tuple[SparseTensor, SparseTensor]]


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


class KMISCoarsening(MessagePassing):
    propagate_type = {'x': Tensor}
    
    def __init__(self, k=1, ordering='random', eps=0.5, 
                 sample_partition='on_train', 
                 sample_aggregate=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(KMISCoarsening, self).__init__(**kwargs)

        self.k = k
        self.eps = eps
        self.ordering: orderings.Ordering = self._get_ordering(ordering)
        self.sample_partition = sample_partition
        self.sample_aggregate = sample_aggregate

    @property
    def sample_partition(self):
        if self.training:
            return self._sample_on_train
        return self._sample_on_valid

    @sample_partition.setter
    def sample_partition(self, sample_partition):
        self._sample_on_train = sample_partition in {True, 'on_train'}
        self._sample_on_valid = sample_partition == True  # noqa
        self._return_pair = self._sample_on_train or self._sample_on_valid

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

    def pool(self, c_mat: SparseTensor, x: OptTensor) -> OptTensor:
        if x is None:
            return None
        
        l_mat = utils.normalize_dim(c_mat.t(), -1)
        return self.propagate(l_mat, x=x)

    def coarsen(self, c_mat: Union[Tensor, SparseTensor], adj: SparseTensor) -> Union[Tensor, SparseTensor]:
        out = c_mat.t() @ (adj @ c_mat)

        if self.sample_partition or not self._sample_on_train:
            return out

        diag = adj.get_diag().unsqueeze(-1)

        if torch.all(diag == 0.):
            return out

        n, s, device = adj.size(0), out.size(0), adj.device()
        sigma = c_mat.t() @ (utils.get_diagonal_matrix(-diag) @ c_mat)
        sigma_diag = utils.get_diagonal_matrix(c_mat.t() @ diag)

        rows, cols, values = tuple(zip(out.coo(), sigma.coo(), sigma_diag.coo()))
        return SparseTensor(row=torch.cat(rows),
                            col=torch.cat(cols),
                            value=torch.cat(values),
                            sparse_sizes=(s, s)).coalesce('sum')

    @staticmethod
    def _maybe_size(x: OptTensor = None, batch: OptTensor = None) -> Optional[Size]:
        for t in (x, batch):
            if t is not None:
                n = t.size(0)
                return n, n

        return None

    def _get_rank(self, x: OptTensor, adj: Adj) -> OptTensor:
        if self.ordering is None:
            return None

        return self.ordering(x, adj)

    def get_coarsening_matrix(self, adj: SparseTensor, x: OptTensor = None) \
            -> Tuple[OptPairSparseTensor, Tensor]:
        rank = self._get_rank(x, adj)
        c_mat, mis = get_coarsening_matrix(adj, self.k, self.eps, rank)

        if self.sample_partition:
            p_mat = sample_partition_matrix(c_mat)
            return (c_mat, p_mat), mis
        
        if self._return_pair:
            return (c_mat, c_mat), mis

        return c_mat, mis

    def forward(self, x: Union[OptTensor, OptPairTensor],
                edge_index: Adj, edge_attr: OptTensor = None,
                batch: OptTensor = None) -> Tuple[Union[OptTensor, OptPairTensor], SparseTensor,
                                                  OptTensor, Tensor, OptPairSparseTensor]:
        adj = edge_index
        rank, rank_cut = x, -1
        
        if isinstance(x, tuple):
            rank, x = x
            rank_cut = rank.size(1)
            x = torch.cat([rank, x])

        if isinstance(adj, Tensor):
            size = self._maybe_size(x, batch)
            adj = SparseTensor.from_edge_index(edge_index, edge_attr, size)

        out_mat, mis = self.get_coarsening_matrix(adj, rank)
        
        if self._return_pair:
            c_mat, p_mat = out_mat

            if self.sample_aggregate:
                c_mat = p_mat
        else:
            c_mat = p_mat = out_mat
            
        out_adj = self.coarsen(p_mat, adj)
        out_x = self.pool(c_mat, x)
        
        if batch is not None:
            batch = batch[mis]
            
        if rank_cut >= 0:
            out_x = (out_x[:, :rank_cut], out_x[:, rank_cut:])
        
        return out_x, out_adj, batch, mis, out_mat
    
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:  # noqa
        return adj_t.matmul(x, reduce=self.aggr)
