import torch


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

