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
    
    for i in (0, 1, 2):
        got = fh.sigma(i)
        expected = np.sqrt(np.trapz(k**(2+i) * Pk / (2*np.pi**2), k))

        assert_allclose(got, expected)

def test_std():
    # TODO: test that the variance of the field is 0.81 at 8Mpc
    pass