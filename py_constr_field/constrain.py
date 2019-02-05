'''Classes to represent linear constrains.'''
import attr
import numpy as np
from scipy.integrate import dblquad
from itertools import combinations_with_replacement
import numexpr
from scipy.interpolate import interp1d
from tqdm import tqdm

from .field import FieldHandler
from .filters import Filter
from .utils import integrand, build_index, rotate_covariance, rotate_xi

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
        self.N = np.atleast_2d(self.N)

    def _compute_ipos(self):
        grid = self._fh.get_grid()

        new_shape = [-1] + [1] * (grid.ndim-1)
        pos = self._cfd.position.reshape(*new_shape)
        grid = self._fh.get_grid()
        d = numexpr.evaluate('sum((grid-pos)**2, axis=0)',
         local_dict=dict(pos=pos, grid=grid))
        ipos = np.unravel_index(np.argmin(d), grid.shape[1:])
        return ipos

    def _precompute_xi(self, rtol=1e-2):
        '''Precompute the value of the correlation function on a fixed grid'''
        Rmin = min(self._filter.radius, self._fh.filter.radius)

        window1 = self._filter.W
        window2 = self._fh.filter.W

        Pk_gen = self._fh.Pk

        k = Pk_gen.x
        Pk = Pk_gen.y
        k2Pk = k**2 * Pk

        k2Pk_W1_W2 = k2Pk * window1(k) * window2(k)

        eightpi3 = 8*np.pi**3

        N1 = self._cfd.N
        self._xi = {}

        @np.vectorize
        def compute_xi(d, ikx, iky, ikz, ikk):
            integral, _ = dblquad(
                integrand, 0, np.pi,
                lambda theta: 0, lambda theta: 2*np.pi,
                epsrel=1e-5, epsabs=1e-5,
                args=(ikx, iky, ikz, ikk, d, k, k2Pk_W1_W2))

            val = integral / eightpi3
            return val

        for i, (ikx, iky, ikz, ikk) in enumerate(N1):
            if iky % 2 == 1 or ikz % 2 == 1:
                self._xi[i] = interp1d([0, 1e5], [0, 0])
                continue

            N = ikx + iky + ikz - ikk
            sigma2 = self._fh.sigma(N/2)**2

            distances = [0, self._fh.Lbox * np.sqrt(3) / 2]
            y = list(compute_xi(distances, ikx, iky, ikz, ikk))

            # Find distance at which the correlation vanishes
            norm = sigma2
            diff = np.abs(y[-1] / norm)
            while diff > rtol:
                new_dmax = distances[-1]*1.5
                print(f'Increasing dmax={new_dmax:.3f} (diff={diff:.3f}, '
                      f'ylast={y[-1]:.3f}, sigma²={sigma2:.3f})')
                distances.append(new_dmax)
                y.append(compute_xi(new_dmax, ikx, iky, ikz, ikk))
                norm = min(np.abs(np.ptp(y)), sigma2)
                diff = np.abs(y[-1] / norm)

            order = np.argsort(distances)
            distances = np.asarray(distances)[order]
            y = np.asarray(y)[order]

            # Refine until dx and the variation between two steps are small enough
            dx = np.diff(distances)
            norm = min(np.abs(y.ptp()), sigma2)
            dy_o_sigma = np.abs(np.diff(y) / norm)
            mask = (dx > Rmin / 2) | (dy_o_sigma > rtol)
            while mask.sum() > 0 and len(distances) < 1000:
                i0 = np.argwhere(mask).flatten()
                i1 = i0 + 1

                new_d = (distances[i0] + distances[i1]) / 2
                new_y = compute_xi(new_d, ikx, iky, ikz, ikk)

                distances = np.concatenate((distances, new_d))
                order = np.argsort(distances)

                # Sort arrays
                y = np.concatenate((y, new_y))[order]
                distances = distances[order]

                dx = np.diff(distances)
                norm = min(np.abs(y.ptp()), sigma2)
                dy_o_sigma = np.abs(np.diff(y) / norm)
                mask = (dx > Rmin / 2) | (dy_o_sigma > rtol)
                print(f'Computing {mask.sum()}/{mask.shape[0]} new point '
                      f'(norm={norm:.3f}, sigma²={sigma2:.3f})')

            self._xi[i] = interp1d(distances, y, kind='quadratic', bounds_error=False, fill_value=0)

    def __repr__(self):
        return '<Constrain: %s, v=%s>' % (self.__class__.__name__, self.value)

    def compute_covariance(self, other, frame):
        '''Compute the covariance matrice between two functional constrains.'''
        return compute_covariance(self, other, frame)

    @property
    def xi(self):
        '''Compute the correlation function between the current functional
        and the density field at given positions.'''
        if self._xi is None:
            self._precompute_xi()

        X0 = self._cfd.position
        def loc(positions):
            tmp = np.atleast_2d((positions))
            assert tmp.shape[-1] == 3
            positions = tmp.reshape(-1, 3)

            d = np.linalg.norm(X0 - positions, axis=-1)

            xi = np.array([xi(d)[..., None, :] for xi in self._xi.values()]).T
            return rotate_xi(self, positions, xi)

        return loc

    def measure_compact(self):
        '''Measure the value of the field for the given constrain and return
        its independant component.'''
        val = self.measure()
        Nfreedom = self.N[0, :-1].sum()
        out = []
        for i0 in combinations_with_replacement(range(3), Nfreedom):
            out.append(val[i0])
        return np.array(out)

    def measure(self):
        '''Measure the value of the field for the given constrain.'''
        raise NotImplementedError()


class DensityConstrain(Constrain):
    N = [0, 0, 0, 0]
    sign = 1

    def measure(self):
        field = self._fh.get_smoothed(self._filter)
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

        indices = tuple(np.meshgrid(*(np.arange(i-1, i+2) %
                                      N for i in ipos), indexing='ij'))
        tmp = field[indices]
        dx = self._fh.Lbox / self._fh.dimensions
        return np.array(np.gradient(tmp, dx))[..., 1, 1, 1]


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

        indices = tuple(np.meshgrid(*(np.arange(i-2, i+3) %
                                      N for i in ipos), indexing='ij'))
        tmp = field[indices]
        dx = self._fh.Lbox / self._fh.dimensions
        gradients = np.array(np.gradient(np.gradient(
            tmp, dx, axis=(-3, -2, -1)), dx, axis=(-3, -2, -1)))
        return gradients[..., 2, 2, 2]


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

    def measure(self):
        field = self._fh.get_smoothed(self._filter)
        grid = self._fh.get_grid()
        ipos = self._ipos

        N = grid.shape[-1]

        indices = tuple(np.meshgrid(*(np.arange(i-3, i+4) %
                                      N for i in ipos), indexing='ij'))
        tmp = field[indices]
        dx = self._fh.Lbox / self._fh.dimensions
        gradients = np.array(np.gradient(np.gradient(np.gradient(
                    tmp, dx, axis=(-3, -2, -1)),
                dx, axis=(-3, -2, -1)),
            dx, axis=(-3, -2, -1)))
        return gradients[..., 3, 3, 3]
