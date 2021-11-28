import numpy as np
from colossus.cosmology import cosmology
from numpy.testing import assert_allclose

from py_constr_field import filters
from py_constr_field.field import FieldHandler

cosmo = cosmology.setCosmology("planck18")
k = np.geomspace(1e-4, 1e4, 2000)
Pk = cosmo.matterPowerSpectrum(k)


def test_sigma():
    filt = filters.GaussianFilter(radius=5)
    fh = FieldHandler(Ndim=3, Lbox=50, dimensions=16, Pk=(k, Pk), filter=filt)

    def test_sigma_n(i):
        got = fh.sigma(i)
        expected = np.sqrt(
            np.trapz(k ** (2 + 2 * i) * Pk * filt.W(k) ** 2, k) / (2 * np.pi ** 2)
        )

        assert_allclose(got, expected)

    for i in range(-5, 5):
        test_sigma_n(i)


def test_field_sigma():
    fh = FieldHandler(Ndim=3, Lbox=100, dimensions=128, Pk=(k, Pk))

    smoothing_scales = np.geomspace(5, 10, 20)
    sigma_th = [fh.sigma(0, filters.TopHatFilter(R)) for R in smoothing_scales]
    sigma_exp = [
        fh.get_smoothed(filters.TopHatFilter(R)).std() for R in smoothing_scales
    ]

    # Check the field variance
    assert_allclose(sigma_th, sigma_exp, rtol=5e-2)


def test_filtering():
    N = 64
    v0 = 10
    fh = FieldHandler(Ndim=3, Lbox=N, dimensions=N, Pk=(k, Pk))
    fh.white_noise[:] = 0
    fh.white_noise[N // 2, N // 2, N // 2] = v0
    fh.white_noise_fft = np.fft.rfftn(fh.white_noise)

    R = 10
    field = fh.get_smoothed(R)
    x, y, z = (
        np.array(np.meshgrid(*[np.linspace(0, N - 1, N)] * 3, indexing="ij")) - N // 2
    )
    ref = (
        1
        / (2 * np.pi * R ** 2) ** (3 / 2)
        * np.exp(-(x ** 2 + y ** 2 + z ** 2) / 2 / R ** 2)
        * v0
    )

    assert_allclose(field, ref, atol=1e-5)
