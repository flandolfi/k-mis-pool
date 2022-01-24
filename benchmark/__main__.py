import fire

from benchmark.train import train, grid_search


if __name__ == "__main__":
    fire.Fire({
        'train': train,
        'grid_search': grid_search,
    })
