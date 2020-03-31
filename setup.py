from setuptools import setup, find_packages

__version__ = '0.0.1'
url = 'https://github.com/flandolfi/graph-pooling'

install_requires = [
    'numpy',
    'torch',
    'torch_sparse',
    'torch_scatter',
    'torch_cluster',
    'torch_geometric',
    'skorch',
    'scikit-learn',
    'pandas',
    'tqdm'
]
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']

setup(
    name='gpool',
    version=__version__,
    description='Pooling for Graph Neural Networks',
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