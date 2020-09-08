import math
from abc import abstractmethod, ABC

import torch


class Kernel(ABC):
    @abstractmethod
    def __call__(self, values: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError


class Gaussian(Kernel):
    def __call__(self, values):
        return values.square().neg_().div_(2).exp_()


class UnitKernel(Kernel):
    def __call__(self, values):
        return (values.abs() <= 1).float()


class Parabolic(UnitKernel):
    def __call__(self, values):
        return 0.75*(1. - values.square())*super(Parabolic, self).__call__(values)


class Biweight(UnitKernel):
    def __call__(self, values):
        return 15.*(1. - values.square()).square()*super(Biweight, self).__call__(values)/16.


class Triweight(UnitKernel):
    def __call__(self, values):
        return 35.*(1. - values.square()).pow(3)*super(Triweight, self).__call__(values)/32.


class Tricube(UnitKernel):
    def __call__(self, values):
        return 70.*(1. - values.pow(3)).pow(3)*super(Tricube, self).__call__(values)/81.


class Cosine(UnitKernel):
    def __call__(self, values):
        return 0.25*math.pi*torch.cos(0.5*math.pi*values)*super(Cosine, self).__call__(values)


# Aliases
RBF = Normal = RadialBasisFunction = Gaussian
