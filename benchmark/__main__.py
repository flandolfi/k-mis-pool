import fire

from benchmark.spectrum import approx, spectrum_approximation
from benchmark.grids import draw_mnist
from benchmark.sample import sample


if __name__ == "__main__":
    fire.Fire({
        'approx': spectrum_approximation,
        'mnist': draw_mnist,
        'sample': sample,
    })
