from colossus.cosmology import cosmology
from numpy.testing import assert_allclose
import numpy as np

from py_constr_field import filters
from py_constr_field.field import FieldHandler

cosmo = cosmology.setCosmology('planck18')
k = np.geomspace(1e-4, 1e4, 2000)
Pk = cosmo.matterPowerSpectrum(k)

def test_sigma():
    filt = filters.GaussianFilter(radius=5)
    fh = FieldHandler(Ndim=3, Lbox=50, dimensions=16, Pk=(k, Pk), filter=filt)
    
    def test_sigma_n(i):
        got = fh.sigma(i)
        expected = np.sqrt(np.trapz(k**(2+2*i) * Pk * filt.W(k)**2, k) / (2*np.pi**2))

        assert_allclose(got, expected)

    for i in range(-5, 5):
        yield test_sigma_n, i

def test_std():
    # TODO: test that the variance of the field is 0.81 at 8Mpc
    pass