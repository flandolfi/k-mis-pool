import torch
from sklearn import datasets
import matplotlib.pyplot as plt

from kmis.reduce import KMISIsomap


def isomap(samples=1500, size=12, **kwargs):
    X, color = datasets.make_swiss_roll(n_samples=samples)
    k_mis = KMISIsomap(components=2, **kwargs)
    
    fig = plt.figure(figsize=(2*size, size))

    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    ax.set_axis_off()
    
    Y = k_mis(torch.from_numpy(X).float()).cpu().numpy()
    ax = fig.add_subplot(122)
    ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    ax.set_axis_off()
    plt.axis('tight')
    plt.show()
