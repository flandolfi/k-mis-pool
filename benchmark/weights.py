import logging

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import torch
from torch_geometric.seed import seed_everything

from benchmark.datasets import load_graph
from kmis import maximal_k_independent_set, orderings


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
            seed_everything(seed)
            x = torch.rand((adj.size(0), 1), dtype=torch.float, device=device)
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
    