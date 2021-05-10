import fire

from benchmark.approx import spectrum_approximation
from benchmark.grids import draw_mnist
from benchmark.sample import sample
from benchmark.models import count_params
from benchmark.train import train, score


if __name__ == "__main__":
    fire.Fire({
        'approx': lambda *args, **kwargs: str(spectrum_approximation(*args, **kwargs)),
        'mnist': draw_mnist,
        'sample': sample,
        'count_params': count_params,
        'train': train,
        'score': score,
    })
