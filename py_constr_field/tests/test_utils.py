import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from scipy.special import spherical_jn

from py_constr_field.utils import WTH, trapz


def test_trapz():
    x = np.linspace(0, 100)
    y = x**2

    vref = np.trapz(y, x)
    vtest = trapz(y, x)

    assert_almost_equal(vref, vtest)


def test_WTH():
    x = np.linspace(0, 2, 1000)
    # This will show a warning because of a divide by 0
    vref = np.where(x == 0, 1, 3 * spherical_jn(1, x) / x)
    vtest = WTH(x)

    assert_allclose(vref, vtest)
