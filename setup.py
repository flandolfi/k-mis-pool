from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import CppExtension, BuildExtension

extra_compile_args = []
# extra_compile_args = ['-g', '-O0', '-DDEBUG']

cmdclass = {'build_ext': BuildExtension}
ext_modules = [
    CppExtension('gpool.select_cpu', 
                 ['cpu/select.cpp'], 
                 extra_compile_args=extra_compile_args),
]

__version__ = '0.0.1'
url = 'https://github.com/flandolfi/graph-pooling'

install_requires = [
    'numpy',
    'torch', 
    'torch_geometric'
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
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
)