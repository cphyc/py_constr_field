import numpy as np
from numba import njit, guvectorize


# some useful functions
@njit
def trapz(A, x):
    return np.sum((A[1:] + A[:-1]) / 2 * np.diff(x))


@guvectorize(['void(float64[:],float64[:])'],
             '(N)->(N)')
def WTH(X, out):
    for i in range(X.shape[0]):
        x = X[i]
        x2 = X[i]**2
        if x < 1e-4:
            out[i] = 1 - x2/10
        else:
            out[i] = 3 * (np.sin(x)/x2/x - np.cos(x)/x2)


@njit
def integrand(phi, theta, ikx, iky, ikz, ikk, d, k, k2Pk_W1_W2):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Compute parity
    ii = ikx + iky + ikz - ikk

    kx = k * sin_theta * np.cos(phi)
    ky = k * sin_theta * np.sin(phi)
    kz = k * cos_theta

    exppart = np.cos(kz * d - ii*np.pi/2)

    intgd = (
        k2Pk_W1_W2 * sin_theta * exppart
    )

    if ikx != 0:
        intgd *= kx**ikx
    if iky != 0:
        intgd *= ky**iky
    if ikz != 0:
        intgd *= kz**ikz
    if ikk != 0:
        intgd /= k**ikk

    return np.trapz(intgd, k)
