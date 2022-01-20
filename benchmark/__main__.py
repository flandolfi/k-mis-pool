import fire

from benchmark.train import train, score


if __name__ == "__main__":
    fire.Fire({
        'train': train,
        'score': score,
    })
