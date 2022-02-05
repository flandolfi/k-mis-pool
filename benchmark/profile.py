import time
import json
import logging
from typing import Union

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import torch
from torch_sparse import SparseTensor

from kmis import reduce, orderings, sample
from benchmark import datasets


def profile(name: str = 'luxembourg_osm',
            group: str = 'DIMACS10',
            root: str = 'datasets/',
            device: str = 'cuda',
            k: Union[list, int] = 1,
            runs: int = 10,
            output_json: str = None,
            logging_level: int = logging.INFO):
    ks = k if isinstance(k, list) else [k]
    adj = datasets.load_graph(name, group, root, logging_level=logging_level)
    results = dict(name=name, group=group, device=device,
                   n=adj.size(0), m=adj.nnz())
    
    # Add randomness for tie-splitting
    perm = torch.randperm(adj.size(0))
    row, col, _ = adj.coo()
    ptr, col, _ = SparseTensor(row=perm[row], col=perm[col],
                               sparse_sizes=adj.sparse_sizes()).csr()
    adj = SparseTensor(rowptr=ptr, col=col, sparse_sizes=adj.sparse_sizes())
    _ignored = torch.tensor([0], dtype=torch.float, device=device)

    if device == 'cpu':
        results['tx_time'] = 0
    else:
        logging.info(f"Moving adjacency matrix to {device}...")
        start = time.time()
        adj = adj.to(device=device)
        elapsed = time.time() - start
        results['tx_time'] = elapsed

    results['reductions'] = {}

    with logging_redirect_tqdm():
        for k in tqdm(ks):
            k_results = []

            for _ in tqdm(list(range(runs))):
                run_result = {}
                ordering = orderings.DivKSum(k)

                logging.info(f"Computing inverse {k}-paths ordering...")
                start = time.time()
                rank = ordering(torch.ones((adj.size(0),), dtype=torch.float, device=device), adj)
                elapsed = time.time() - start
                run_result['ord_time'] = elapsed

                logging.info(f"Computing {k}-MIS...")
                start = time.time()
                mis = sample.maximal_k_independent_set(adj, k, rank)
                elapsed = time.time() - start
                run_result['k_mis_time'] = elapsed

                logging.info(f"Reducing graph {k}-MIS...")
                start = time.time()
                adj_redux, cluster = reduce.sparse_reduce_adj(mis, adj, k, rank)
                elapsed = time.time() - start
                run_result['reduction_time'] = elapsed

                run_result.update(n=adj_redux.size(0), m=adj_redux.nnz())
                k_results.append(run_result)

            results['reductions'][k] = k_results

    if output_json:
        with open(output_json, 'w') as fd:
            json.dump(results, fd)
    else:
        print(results)
