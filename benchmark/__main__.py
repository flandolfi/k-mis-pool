import fire

from benchmark.spectrum import approx
from benchmark.grids import draw_mnist
from benchmark.sample import sample


if __name__ == "__main__":
    fire.Fire({
        'approx': approx,
        'mnist': draw_mnist,
        'sample': sample,
    })
