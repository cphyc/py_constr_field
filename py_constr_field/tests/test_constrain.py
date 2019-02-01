from py_constr_field.constrain import Constrain, FieldHandler, DensityConstrain, compute_covariance
from py_constr_field import filters

from colossus.cosmology import cosmology
import numpy as np

cosmo = cosmology.setCosmology('planck18')
k = np.geomspace(1e-4, 1e4, 1000)
Pk = cosmo.matterPowerSpectrum(k)


def test_constrain_initialisation():
    filter = filters.GaussianFilter(radius=5)
    c = Constrain(position=[0, 0, 0], filter=filter, value=1.69,
                  field_handler=None)
    c


def test_constrain_correlation():
    fh = FieldHandler(Ndim=3, Lbox=50, dimensions=16, Pk=(k, Pk))
    filt = filters.TopHatFilter(radius=8)

    c = DensityConstrain([0, 0, 0], filt, fh, 1.69)
    v = compute_covariance(c, c)
