import numpy as np
from numba import njit, guvectorize
from itertools import combinations_with_replacement, permutations
from opt_einsum import contract as einsum


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

    exppart = np.cos(kx * d - ii*np.pi/2)

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

    return trapz(intgd, k)


def build_index(Nfreedom):
    ii = 0
    index = np.zeros([3]*Nfreedom, dtype=int)
    combin = combinations_with_replacement(range(3), Nfreedom)
    for icount, ii in enumerate(combin):
        for jj in permutations(ii):
            index[jj] = icount
    return np.atleast_1d(index)


def rotate_covariance(c1, c2, cov):
    '''Rotate a covariance matrix from the separation frame to the cartesian frame

    Parameter
    ---------
    c1, c2 : Constrain object or array.
    cov : ndarray (N, M)
        The covariance matrix in the separation frame.

    Returns
    -------
    cov : ndarray (N, M)
        The covariance matrix in the cartesian frame.

    '''
    X1 = c1._cfd.position.astype(np.float64)
    X2 = c2._cfd.position.astype(np.float64)
    Nf1 = c1._cfd.N[:, :-1].sum(axis=1)[0]
    Nf2 = c2._cfd.N[:, :-1].sum(axis=1)[0]

    # Nothing to do for scalars or at zero separation
    if (Nf1 == 0 and Nf2 == 0) or np.all(X1 == X2):
        return cov

    r = X2 - X1
    e1 = r
    e2 = np.array([-e1[1]-e1[2], e1[0]-e1[2], e1[0]+e1[1]])
    e3 = np.cross(e1, e2)

    e1 /= np.linalg.norm(e1)
    e2 /= np.linalg.norm(e2)
    e3 /= np.linalg.norm(e3)
    E = np.array([e1, e2, e3])

    B = np.diag([1, 1, 1])
    R1 = np.array([[np.dot(b, e) for b in B] for e in E])
    R2 = np.array([[np.dot(b, e) for b in B] for e in E])

    index1 = build_index(Nf1)
    index2 = build_index(Nf2)

    # Unpack covariance into large tensor of shape (3, 3, ...)
    extended_cov = cov[:, index2][index1, ...]

    # Drop useless dimensions
    if Nf1 == 0:
        extended_cov = extended_cov[0, ...]
    if Nf2 == 0:
        extended_cov = extended_cov[..., 0]

    Ndim = extended_cov.ndim
    ii = 0

    # Build the rotation using the einstein notation
    args = [extended_cov, list(range(Ndim))]
    for i in range(Nf1):
        args.extend([R1, [ii, ii+Ndim]])
        ii += 1
    for i in range(Nf2):
        args.extend([R2, [ii, ii+Ndim]])
        ii += 1

    extended_cov = einsum(*args)

    # Re-add missing dimensions
    if Nf1 == 0:
        extended_cov = extended_cov[None, ...]
    if Nf2 == 0:
        extended_cov = extended_cov[..., None]

    # Reproject into dense space
    new_cov = np.zeros_like(cov)
    for a, ii in enumerate(combinations_with_replacement(range(3), Nf1)):
        if Nf1 == 0:
            ii = [0]
        for b, jj in enumerate(combinations_with_replacement(range(3), Nf2)):
            if Nf2 == 0:
                jj = [0]
            ipos = list(ii) + list(jj)
            new_cov[a, b] = extended_cov.item(*ipos)

    return np.asarray(new_cov)

def rotate_xi(c, positions, xi0):
    '''Rotate the xi function from separation to cartesian frame.
    
    Parameters
    ----------
    c : Constrain
        The constrain object
    positions : ndarray (Npt, Ndim)
        The positions at which the xi function is evaluated.
    xi0 : ndarray (Npt, 1, N1, ..., Nn)
        The value of the correlation function between the field and the constrain.
        
    Returns
    -------
    xi : ndarray, (Npt, 1, N1, ..., Nn)
        The value of correlation function in the cartesian frame.
    '''
    shape = positions.shape[:-1]
    
    Nf1 = 0
    Nf2 = c._cfd.N[:, :-1].sum(axis=1)[0]
    X1 = c._cfd.position
    X2 = positions
    d = np.linalg.norm(X2 - X1, axis=-1)
    e1 = X2 - X1
    e1[d == 0, :] = [1, 0, 0]  # Special care at zero-separation
    e2 = np.array([-e1[..., 1]-e1[..., 2], e1[..., 0]-e1[..., 2], e1[..., 0]+e1[..., 1]]).T
    e3 = np.cross(e1, e2)

    e1 /= np.linalg.norm(e1, axis=-1)[..., None]
    e2 /= np.linalg.norm(e2, axis=-1)[..., None]
    e3 /= np.linalg.norm(e3, axis=-1)[..., None]

    E = np.empty((*shape, 3, 3))
    E[..., 0, :] = e1
    E[..., 1, :] = e2
    E[..., 2, :] = e3

    B = np.diag([1, 1, 1])
    R1 = np.array([[np.dot(b, e) for b in B] for e in E])
    R2 = np.array([[np.dot(b, e) for b in B] for e in E])

    index1 = build_index(Nf1)
    index2 = build_index(Nf2)

    # Unpack xi into large tensor of shape (..., 3, 3, 3)
    extended_xi = xi0[..., index1, :][..., index2]

    # Drop useless dimensions
    tmp = list(extended_xi.shape)
    if Nf1 == 0:
        tmp.pop(len(shape))
    if Nf2 == 0:
        tmp.pop(-1)
    extended_xi = extended_xi.reshape(tmp)

    Ndim = extended_xi.ndim - len(shape)
    ii = 0

    # Build the rotation using the einstein notation
    args = [extended_xi, [...] + list(range(Ndim))]
    for i in range(Nf1):
        args.extend([R1, [..., ii, ii+Ndim]])
        ii += 1
    for i in range(Nf2):
        args.extend([R2, [..., ii, ii+Ndim]])
        ii += 1

    extended_xi = einsum(*args)

    # Re-add missing dimensions
    tmp = list(extended_xi.shape)
    if Nf1 == 0:
        tmp.insert(len(shape), 1)
    if Nf2 == 0:
        tmp.append(1)
    extended_xi = extended_xi.reshape(tmp)

    # Reproject into dense space
    xi = np.zeros_like(xi0)
    for a, ii in enumerate(combinations_with_replacement(range(3), Nf1)):
        if Nf1 == 0:
            ii = [0]
        for b, jj in enumerate(combinations_with_replacement(range(3), Nf2)):
            if Nf2 == 0:
                jj = [0]
            ipos = list(ii) + list(jj)
            xi[..., a, b] = extended_xi[[...] + ipos]

    return xi
