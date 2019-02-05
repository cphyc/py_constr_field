import attr
import numpy as np

from .utils import WTH


@attr.s(frozen=True)
class Filter(object):
    radius = attr.ib(converter=float)

    def W(self, k):
        '''The smoothing function in Fourier space.

        Parameter
        ---------
        k : float or ndarray

        Returns
        -------
        W : float or ndarray'''
        raise NotImplementedError


@attr.s(frozen=True)
class GaussianFilter(Filter):
    def W(self, k):
        '''The smoothing function in Fourier space.

        Parameter
        ---------
        k : float or ndarray

        Returns
        -------
        W : float or ndarray'''
        return np.exp(-(k*self.radius)**2/2)


@attr.s(frozen=True)
class TopHatFilter(Filter):
    def W(self, k):
        '''The smoothing function in Fourier space.

        Parameter
        ---------
        k : float or ndarray

        Returns
        -------
        W : float or ndarray'''
        return WTH(k*self.radius)