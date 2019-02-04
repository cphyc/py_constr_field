from colossus.cosmology import cosmology
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
from correlations.correlations import Correlator
from correlations.utils import Utils

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
    f1 = filters.GaussianFilter(radius=5)
    f2 = filters.GaussianFilter(radius=6)
    f3 = filters.GaussianFilter(radius=7)
    f4 = filters.GaussianFilter(radius=8)

    X1 = np.array([0, 0, 0])

    c1 = C.DensityConstrain(X1, filter=f1, value=1.69, field_handler=fh)
    c2 = C.GradientConstrain(X1, filter=f2, value=[0, 0, 0], field_handler=fh)
    c3 = C.HessianConstrain(X1, filter=f3, value=[0, 0, 0, 0, 0, 0], field_handler=fh)
    c4 = C.ThirdDerivativeConstrain(X1, filter=f4, value=list(range(10)), field_handler=fh)

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

def test_full_correlation():
    def test_it(use_cache):
        f1 = filters.GaussianFilter(radius=5)
        f2 = filters.GaussianFilter(radius=6)
        f3 = filters.GaussianFilter(radius=7)

        X1 = np.array([0, 0, 0])
        X2 = np.array([1, 2, 3])
        X3 = X2 * 2

        # Reference
        c = Correlator(quiet=True)
        c.add_point(X1, ['density'], f1.radius)
        c.add_point(X2, ['grad_delta'], f2.radius)
        c.add_point(X3, ['hessian'], f3.radius)
        U = Utils(c.k, c.Pk)

        # Note: here we use the old package's k and Pk so that their result agree can be
        # be compared
        fh = FieldHandler(Ndim=3, Lbox=50, dimensions=16, Pk=(c.k, c.Pk), use_covariance_cache=use_cache)

        c1 = C.DensityConstrain(X1, filter=f1, value=1.69, field_handler=fh)
        c2 = C.GradientConstrain(X2, filter=f2, value=[0, 0, 0], field_handler=fh)
        c3 = C.HessianConstrain(X3, filter=f3, value=[0, 0, 0, 0, 0, 0], field_handler=fh)
        fh.add_constrain(c1)
        fh.add_constrain(c2)
        fh.add_constrain(c3)

        # Compute matrix of sigma
        sigma = np.array(
            [U.sigma(0, f1.radius)] + 
            [U.sigma(1, f2.radius)]*3 +
            [U.sigma(2, f3.radius)]*6)
        S = sigma[:, None] * sigma[None, :]

        ref = c.cov
        new = fh.compute_covariance() / S

        # Check closeness (note: the order may be different so only check det and eigenvalues)
        det_ref = np.linalg.det(ref)
        det_new = np.linalg.det(new)

        assert_allclose(det_ref, det_new)

        eval_ref = np.linalg.eigvalsh(ref)
        eval_new = np.linalg.eigvalsh(new)
        
        assert_allclose(eval_ref, eval_new)

    for use_cache in (True, False):
        yield test_it, use_cache
