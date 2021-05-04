import fire

from benchmark.approx import spectrum_approximation
from benchmark.grids import draw_mnist
from benchmark.sample import sample


if __name__ == "__main__":
    fire.Fire({
        'approx': lambda *args, **kwargs: str(spectrum_approximation(*args, **kwargs)),
        'mnist': draw_mnist,
        'sample': sample,
    })
