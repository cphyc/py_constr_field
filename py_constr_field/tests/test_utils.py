from numpy.testing import assert_allclose, assert_almost_equal
from scipy.special import spherical_jn
import numpy as np

from py_constr_field.utils import trapz, WTH


def test_trapz():
    x = np.linspace(0, 100)
    y = x**2

    vref = np.trapz(y, x)
    vtest = trapz(y, x)

    assert_almost_equal(vref, vtest)


def test_WTH():
    x = np.linspace(0, 2, 1000)
    vref = np.where(x == 0, 1, 3*spherical_jn(1, x) / x)
    vtest = WTH(x)

    assert_allclose(vref, vtest)
