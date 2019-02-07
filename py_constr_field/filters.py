import attr
import numpy as np
import numexpr as ne

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
        radius = self.radius
        return ne.evaluate('exp(-(k*radius)**2/2)',
         local_dict={'radius': radius, 'k': k})


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
