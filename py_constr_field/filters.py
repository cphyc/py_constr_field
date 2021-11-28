import attr
import numexpr as ne
import numpy as np
from scipy.special import j0

from .utils import WTH


@attr.s(frozen=True)
class Filter:
    radius = attr.ib(converter=float)

    def W(self, k):
        """The smoothing function in Fourier space.

        Parameter
        ---------
        k : float or ndarray

        Returns
        -------
        W : float or ndarray"""
        raise NotImplementedError


@attr.s(frozen=True)
class GaussianFilter(Filter):
    def W(self, k):
        """The smoothing function in Fourier space.

        Parameter
        ---------
        k : float or ndarray

        Returns
        -------
        W : float or ndarray"""
        radius = self.radius
        return ne.evaluate(
            "exp(-(k*radius)**2/2)", local_dict={"radius": radius, "k": k}
        )


@attr.s(frozen=True)
class TopHatFilter(Filter):
    def W(self, k):
        """The smoothing function in Fourier space.

        Parameter
        ---------
        k : float or ndarray

        Returns
        -------
        W : float or ndarray"""
        return WTH(k * self.radius)


@attr.s(frozen=True)
class TopHatFilterND(Filter):
    Ndim = attr.ib(default=2)

    def W(self, k):
        """The smoothing function in Fourier space.

        Parameter
        ---------
        k : float or ndarray

        Returns
        -------
        W : float or ndarray

        Notes
        -----
        See Desjacques 2016, Eq. A15-17.
        """
        if self.Ndim == 3:
            return WTH(k * self.radius)
        if self.Ndim == 2:
            return j0(k * self.radius)
        elif self.Ndim == 1:
            return np.cos(k * self.radius)
