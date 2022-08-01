import logging

import fire

from benchmark.train import train, grid_search
from benchmark.datasets import info
from benchmark.weights import weight, generate_dimacs92_files
from benchmark.plots import plot_reductions, plot_mnist
from benchmark.profiling import profile

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    level=logging.INFO)


if __name__ == "__main__":
    fire.Fire({
        'train': train,
        'grid_search': grid_search,
        'info': info,
        'weight': weight,
        'export': generate_dimacs92_files,
        'dist': plot_reductions,
        'mnist': plot_mnist,
        'profile': profile,
    })
