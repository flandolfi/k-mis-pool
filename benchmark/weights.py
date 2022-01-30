import os
import logging

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import torch
from torch_geometric.seed import seed_everything

from benchmark.datasets import load_graph
from kmis import maximal_k_independent_set, orderings


def generate_random_weights(size, seed, highest=100):
    seed_everything(seed)
    return torch.randint(1, highest + 1, size=(size,), dtype=torch.short, device='cpu')


def generate_dimacs92_files(
        name: str = 'luxembourg_osm',
        group: str = 'DIMACS10',
        root: str = 'datasets/',
        k: int = 1,
        runs: int = 10,
        output_dir: str = None,
        logging_level: int = logging.INFO):
    if output_dir is None:
        output_dir = root
        
    adj = load_graph(name, group, root, 'cpu', logging_level=logging_level)

    logging.info(f"Computing the power-{k} graph")
    adj = adj.set_value(None, layout=None).fill_diag(fill_value=None)
    adj_pow = adj.clone()
    
    for _ in range(1, k):
        adj_pow @= adj
    
    del adj
    adj = adj_pow.remove_diag()
    row, col, _ = adj.coo()
    mask = row < col
    row = row[mask] + 1
    col = col[mask] + 1
    n, m = adj.size(0), row.size(0)
    del adj
    
    with logging_redirect_tqdm():
        for seed in tqdm(range(runs), total=runs):
            header = f"c seed {seed}\np edge {n} {m}\n"
            logging.info("Generating random weights...")
            x = generate_random_weights(n, seed=seed)

            logging.info("Writing dimacs92 file...")
            with open(os.path.join(output_dir, f"{name}_{group}_K{k}_S{seed}.graph"), 'w') as fd:
                fd.write(header)
                fd.write("".join(f"n {i} {w}\n" for i, w in enumerate(x, start=1)))
                
                for e in tqdm(range(m), total=m):
                    fd.write(f"e {row[e]} {col[e]}\n")


def weight(name: str = 'luxembourg_osm',
           group: str = 'DIMACS10',
           root: str = 'datasets/',
           k: int = 1,
           ordering: str = 'greedy',
           device: str = 'cpu',
           runs: int = 10,
           store_results: str = None,
           logging_level: int = logging.INFO):
    adj = load_graph(name, group, root, device, logging_level=logging_level)
    
    logging.info(f"Initializing ordering {ordering}")
    
    if ordering == 'greedy':
        ordering = orderings.Greedy()
    else:
        ordering_name = ''.join(map(str.title, ordering.split('-')))
        ordering: orderings.Ordering = getattr(orderings, ordering_name)(k=k)
    
    results = []
    
    with logging_redirect_tqdm():
        for seed in tqdm(range(runs), total=runs):
            logging.info("Generating random weights...")
            x = generate_random_weights(adj.size(0), seed=seed).to(dtype=torch.float, device=device)
            total_weight = x.sum().item()
    
            logging.info("Computing ordering...")
            rank = ordering(x, adj)
            
            logging.info(f"Computing {k}-MWIS...")
            mis = maximal_k_independent_set(adj, k, rank)
            
            mis_size = mis.sum().item()
            mis_weight = x[mis].sum().item()
            logging.info(f"Obtained a {k}-MWIS of size {mis_size} and weight {mis_weight}")
            
            results.append({
                'name': name,
                'group': group,
                'ordering': ordering.__class__.__name__,
                'k': k,
                'n': adj.size(0),
                'm': adj.nnz(),
                'mis_size': mis_size,
                'mis_weight': mis_weight,
                'total_weight': total_weight,
            })
        
    df = pd.DataFrame.from_records(results)
    print(df)
    
    if store_results is not None:
        df.to_json(store_results)
    