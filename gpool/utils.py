import torch
import torch_sparse


def sparse_sort(src: torch.Tensor, index: torch.Tensor, dim=0, descending=False, eps=1e-12):
    f_src = src.float()
    f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
    f_src = (f_src - f_min)/(f_max - f_min + eps) + index.float()*(-1)**int(descending)
    perm = f_src.argsort(dim=dim, descending=descending)

    return src[perm], perm


def sparse_topk(src: torch.Tensor, index: torch.Tensor, k=1, dim=0, descending=False, eps=1e-12):
    sort, perm = sparse_sort(src, index, dim, descending, eps)

    idx = index[perm]
    mask = torch.ones_like(idx, dtype=torch.uint8)
    mask[k:] = idx[k:] != idx[:-k]

    return sort[mask], perm[mask]


def sparse_min(src: torch.Tensor, index: torch.Tensor, dim=0, eps=1e-12):
    return sparse_topk(src, index, 1, dim, False, eps)


def sparse_max(src: torch.Tensor, index: torch.Tensor, dim=0, eps=1e-12):
    return sparse_topk(src, index, 1, dim, True, eps)


def k_hop(edge_index, edge_attr=None, k=2, num_nodes=None, mask=None):
    n = num_nodes if num_nodes else edge_index.max().item() + 1

    if edge_attr is None:
        edge_attr = torch.ones_like(edge_index[0], dtype=torch.float)

    adj_k = {1: (edge_index, edge_attr)}

    def adj_pow(exp, in_mask=None, out_mask=None):
        if exp in adj_k:
            idx, val = adj_k[exp]
            row, col = idx

            edge_mask = torch.ones_like(row, dtype=torch.uint8)

            if in_mask is not None:
                edge_mask &= in_mask[row]

            if out_mask is not None:
                edge_mask &= out_mask[col]

            return idx[:, edge_mask], val[edge_mask]

        l_exp = exp // 2
        l_ind, l_val = adj_pow(l_exp, in_mask, None)
        r_ind, r_val = adj_pow(exp - l_exp, None, out_mask)
        adj_k[exp] = torch_sparse.spspmm(l_ind, l_val, r_ind, r_val, n, n, n)

        return adj_k[exp]

    return adj_pow(k, mask, mask)
