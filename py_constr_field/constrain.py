'''Classes to represent linear constrains.'''
import attr
import numpy as np
from scipy.integrate import dblquad
from itertools import combinations_with_replacement
from opt_einsum import contract as einsum
import numexpr
from scipy.interpolate import interp1d

from .field import FieldHandler
from .filters import Filter
from .utils import integrand, build_index


def rotate_covariance(c1, c2, cov):
    '''Given a covariance matrix between two constrain, rotate it back
    to the original frame.
    
    Parameter
    ---------
    c1, c2 : Constrain object or array.
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
    ext_cov = cov[:, index2][index1, ...]

    # Drop useless dimensions
    if Nf1 == 0:
        ext_cov = ext_cov[0, ...]
    if Nf2 == 0:
        ext_cov = ext_cov[..., 0]

    Ndim = ext_cov.ndim
    ii = 0

    # Build the rotation using the einstein notation
    args = [ext_cov, list(range(Ndim))]
    for i in range(Nf1):
        args.extend([R1, [ii, ii+Ndim]])
        ii += 1
    for i in range(Nf2):
        args.extend([R2, [ii, ii+Ndim]])
        ii += 1

    ext_cov = einsum(*args)

    # Re-add missing dimensions
    if Nf1 == 0:
        ext_cov = ext_cov[None, ...]
    if Nf2 == 0:
        ext_cov = ext_cov[..., None]

    # Reproject into dense space
    new_cov = np.zeros_like(cov)
    for a, ii in enumerate(combinations_with_replacement(range(3), Nf1)):
        if Nf1 == 0:
            ii = [0]
        for b, jj in enumerate(combinations_with_replacement(range(3), Nf2)):
            if Nf2 == 0:
                jj = [0]
            ipos = list(ii) + list(jj)
            new_cov[a, b] = ext_cov.item(*ipos)

    return np.asarray(new_cov)


def compute_covariance(c1, c2, frame):
    '''Compute the covariance between two constrains.

    Arguments
    ---------
    c1, c2 : constrain objects
    frame : ["original" | "separation"]
       The frame to compute covariance.

    Returns
    -------
    cov : ndarray
       The covariance matrix.

    Notes
    -----
    If the frame is "original", the covariance is expressed in the
    frame set by the constrains. If the frame is "separation", it is
    expressed in the separation frame where the third dimension is the
    separation one.
    '''
    X1 = c1._cfd.position
    X2 = c2._cfd.position
    r = X2 - X1
    d = np.linalg.norm(r)
    cache = c1._fh.covariance_cache
    use_cache = c1._fh.use_covariance_cache

    window1 = c1._filter.W
    window2 = c2._filter.W

    Pk_gen = c1._fh.Pk

    k = Pk_gen.x
    Pk = Pk_gen.y
    k2Pk = k**2 * Pk

    k2Pk_W1_W2 = k2Pk * window1(k) * window2(k)

    eightpi3 = 8*np.pi**3

    N1 = c1._cfd.N
    N2 = c2._cfd.N
    cov = np.zeros((N1.shape[0], N2.shape[0]))

    for i, (ikx, iky, ikz, ikk) in enumerate(N1):
        for j, (jkx, jky, jkz, jkk) in enumerate(N2):
            lkx = ikx + jkx
            lky = iky + jky
            lkz = ikz + jkz
            lkk = ikk + jkk

            sign = (-1)**(jkx+jky+jkz)

            key = tuple([lkx] +
                        list(sorted((lky, lkz))) +
                        [lkk, d, c1._filter, c2._filter, sign])

            # Key already computed, use data from cache
            if use_cache and key in cache:
                val = cache[key]
            elif lky % 2 == 1 or lkz % 2 == 1:
                val = 0
            else:    
                integral, _ = dblquad(
                    integrand, 0, np.pi,
                    lambda theta: 0, lambda theta: 2*np.pi,
                    epsrel=1e-5, epsabs=1e-5,
                    args=(lkx, lky, lkz, lkk, d, k, k2Pk_W1_W2))

                val = sign * integral / eightpi3
            
            # Store value in cache
            cov[i, j] = val
            if use_cache: 
                cache[key] = val

    if frame == 'original':
        return rotate_covariance(c1, c2, cov)
    elif frame == 'separation':
        return cov


@attr.s(frozen=True)
class ConstrainFixedData(object):
    position = attr.ib(converter=lambda e: np.array(e, dtype=np.float64))
    filter = attr.ib(validator=[attr.validators.instance_of(Filter)])
    field_handler = attr.ib(type=FieldHandler)
    N = attr.ib(converter=np.atleast_2d)


class Constrain(object):
    value = None
    N = None
    frame = np.diag([1, 1, 1])
    _ipos = None
    _xi = None

    def __init__(self, position, filter, field_handler, value):
        N = np.atleast_2d(self.N)
        self._cfd = ConstrainFixedData(position, filter, field_handler,
                                       N=N)
        self.value = value
        self._filter = filter
        self._fh = field_handler
        self._ipos = self._compute_ipos()

    def _compute_ipos(self):
        grid = self._fh.get_grid()

        new_shape = [-1] + [1] * (grid.ndim-1)
        pos = self._cfd.position.reshape(*new_shape)
        d = numexpr.evaluate('sum((grid-pos)**2, axis=0)')
        ipos = np.unravel_index(np.argmin(d), grid.shape[1:])
        return ipos

    def _precompute_xi(self):
        Rmin = self._fh.Lbox / self._fh.dimensions / 2
        Rmax = self._fh.Lbox * np.sqrt(3)
        distances = np.arange(Rmin, Rmax, Rmin)
        window1 = self._filter.W
        window2 = self._fh.filter.W

        Pk_gen = self._fh.Pk

        k = Pk_gen.x
        Pk = Pk_gen.y
        k2Pk = k**2 * Pk

        k2Pk_W1_W2 = k2Pk * window1(k) * window2(k)

        eightpi3 = 8*np.pi**3

        N1 = self._cfd.N
        xi = np.zeros((len(N1), distances.shape))

        for i, (ikx, iky, ikz, ikk) in enumerate(N1):
            for j, d in distances:
                if iky % 2 == 1 or ikz % 2 == 1:
                    val = 0
                else:    
                    integral, _ = dblquad(
                        integrand, 0, np.pi,
                        lambda theta: 0, lambda theta: 2*np.pi,
                        epsrel=1e-5, epsabs=1e-5,
                        args=(ikx, iky, ikz, ikk, d, k, k2Pk_W1_W2))

                    val = integral / eightpi3
                xi[i, j] = val

        # Build interpolators *in the frame of the separation*
        self._xi = [interp1d(distances, xi[i, :], kind='quadratic') for i in range(len(N1))]

    def __repr__(self):
        return '<Constrain: %s, v=%s>' % (self.__class__.__name__, self.value)

    def measure(self):
        '''Measure the value of the field for the given constrain.'''
        raise NotImplementedError()

    def compute_covariance(self, other, frame):
        '''Compute the covariance matrice between two functional constrains.'''
        return compute_covariance(self, other, frame)

    def compute_xi(self, positions):
        '''Compute the correlation function between the current functionals
        and the density field at a given postion.'''
        pass        


class DensityConstrain(Constrain):
    N = [0, 0, 0, 0]
    sign = 1

    def measure(self):
        field = self._fh.get_smoothed(self._filter)
        grid = self._fh.get_grid()
        ipos = self._ipos

        return field[ipos]


class GradientConstrain(Constrain):
    N = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0]]

    def measure(self):
        field = self._fh.get_smoothed(self._filter)
        grid = self._fh.get_grid()
        ipos = self._ipos

        N = grid.shape[-1]

        indices = tuple(np.meshgrid(*(np.arange(i-1, i+2) % N for i in ipos), indexing='ij'))
        tmp = field[indices]
        return np.array(np.gradient(tmp))[:, 1, 1, 1]


class HessianConstrain(Constrain):
    N = [[2, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 0, 1, 0],
         [0, 2, 0, 0],
         [0, 1, 1, 0],
         [0, 0, 2, 0]]

    def measure(self):
        field = self._fh.get_smoothed(self._filter)
        grid = self._fh.get_grid()
        ipos = self._ipos

        N = grid.shape[-1]

        indices = tuple(np.meshgrid(*(np.arange(i-2, i+3) % N for i in ipos), indexing='ij'))
        tmp = field[indices]

        gradients = np.array(np.gradient(np.gradient(tmp, axis=(-3, -2, -1)), axis=(-3, -2, -1)))
        return gradients[:, :, 2, 2, 2]


class ThirdDerivativeConstrain(Constrain):
    N = [[3, 0, 0, 0],
         [2, 1, 0, 0],
         [2, 0, 1, 0],
         [1, 2, 0, 0],
         [1, 1, 1, 0],
         [1, 0, 2, 0],
         [0, 3, 0, 0],
         [0, 2, 1, 0],
         [0, 1, 2, 0],
         [0, 0, 3, 0]]
