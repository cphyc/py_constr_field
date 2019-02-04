from colossus.cosmology import cosmology
import numpy as np
from numpy.testing import assert_allclose, assert_array_less

from py_constr_field import constrain as C
from py_constr_field import filters
from py_constr_field.field import FieldHandler

cosmo = cosmology.setCosmology('planck18')
k = np.geomspace(1e-4, 1e4, 2000)
Pk = cosmo.matterPowerSpectrum(k)


def test_constrain_initialisation():
    filter = filters.GaussianFilter(radius=5)
    c = C.Constrain(position=[0, 0, 0], filter=filter, value=1.69,
                    field_handler=None)
    c


def test_constrain_correlation_lag():
    fh = FieldHandler(Ndim=3, Lbox=50, dimensions=16, Pk=(k, Pk))
    fh.precompute()
    filt = filters.GaussianFilter(radius=5)

    X1 = np.array([0, 0, 0])
    X2 = np.array([1, 2, 3])
    X3 = X2 * 2
    X4 = X2 * 3

    c1 = C.DensityConstrain(X1, filter=filt, value=1.69, field_handler=fh)
    c2 = C.GradientConstrain(X2, filter=filt, value=[0, 0, 0], field_handler=fh)
    c3 = C.HessianConstrain(X3, filter=filt, value=[0, 0, 0, 0, 0, 0], field_handler=fh)
    c4 = C.ThirdDerivativeConstrain(X4, filter=filt, value=list(range(10)), field_handler=fh)

    fh.add_constrain(c1)
    fh.add_constrain(c2)
    fh.add_constrain(c3)
    fh.add_constrain(c4)

    cov = fh.compute_covariance()

    # Check positive definite
    evals = np.linalg.eigvalsh(cov)
    assert_array_less(np.zeros_like(evals), evals)

    # Check symmetry
    assert_allclose(cov, cov.T)


def test_constrain_correlation_nolag():
    fh = FieldHandler(Ndim=3, Lbox=50, dimensions=16, Pk=(k, Pk))
    fh.precompute()
    filt = filters.GaussianFilter(radius=5)

    X1 = np.array([0, 0, 0])

    c1 = C.DensityConstrain(X1, filter=filt, value=1.69, field_handler=fh)
    c2 = C.GradientConstrain(X1, filter=filt, value=[0, 0, 0], field_handler=fh)
    c3 = C.HessianConstrain(X1, filter=filt, value=[0, 0, 0, 0, 0, 0], field_handler=fh)
    c4 = C.ThirdDerivativeConstrain(X1, filter=filt, value=list(range(10)), field_handler=fh)

    fh.add_constrain(c1)
    fh.add_constrain(c2)
    fh.add_constrain(c3)
    fh.add_constrain(c4)

    cov = fh.compute_covariance()

    # Check positive definite
    evals = np.linalg.eigvalsh(cov)
    assert_array_less(np.zeros_like(evals), evals)

    # Check symmetry
    assert_allclose(cov, cov.T)
