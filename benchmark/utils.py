import torch
import skorch
import sklearn
import numpy as np

from torch_geometric.data import Data, Dataset


def _to_tensor_wrapper(func):
    to_tensor = func

    def wrapper(X, device, allow_sparse=False):
        if isinstance(X, Data):
            return X.to(device)

        return to_tensor(X, device, allow_sparse)

    return wrapper


def _get_item_wrapper(func):
    wrapped = func

    def wrapper(dataset, idx):
        if isinstance(idx, np.int64):
            idx = int(idx)

        return wrapped(dataset, idx)

    return wrapper


def _unpack_data_wrapper(func):
    wrapped = func

    def wrapper(data):
        if isinstance(data, Data):
            return data, data.y

        return wrapped(data)

    return wrapper


def fix_skorch():
    torch.multiprocessing.set_sharing_strategy('file_system')
    skorch.net.unpack_data = _unpack_data_wrapper(skorch.net.unpack_data)
    skorch.net.to_tensor = _to_tensor_wrapper(skorch.net.to_tensor)
    Dataset.__getitem__ = _get_item_wrapper(Dataset.__getitem__)
    sklearn.utils.validation.check_consistent_length = lambda *arrays: None
