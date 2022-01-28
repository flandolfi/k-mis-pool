import fire

from benchmark.train import train, grid_search
from benchmark.datasets import info
from benchmark.weights import weight


if __name__ == "__main__":
    fire.Fire({
        'train': train,
        'grid_search': grid_search,
        'info': info,
        'weight': weight,
    })
