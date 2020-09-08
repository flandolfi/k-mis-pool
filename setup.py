from setuptools import setup, find_packages

__version__ = '0.0.1'
url = 'https://github.com/flandolfi/graph-pooling'

dependency_links = [
    'https://pytorch-geometric.com/whl/torch-1.6.0.html',
    'https://download.pytorch.org/whl/torch_stable.html'
]

install_requires = [
    'torch',
    'torch_sparse',
    'torch_scatter',
    'torch_cluster',
    'torch_geometric'
]

setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']

setup(
    name='mis-pool',
    version=__version__,
    description='Graph Coarsening via Maximal Independent Set Selection',
    author='Francesco Landolfi',
    author_email='francesco.landolfi@phd.unipi.it',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=['pytorch', 'pooling', 'geometric-deep-learning', 'graph'],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=find_packages(),
)
