'''Classes to represent a Gaussian Random field.'''
import numpy as np
import attr
import pyfftw.interfaces.numpy_fft as fft
from scipy.interpolate import interp1d
from collections import namedtuple
from opt_einsum import contract_path, contract
import numexpr as ne


def build_1dinterpolator(arg):
    '''Given a tuple of two arguments, return a 1d interpolation
    function. If the argument is already callable, just pass it.

    Arguments
    ---------
    arg : tuple or callable
       If a callable, return it. Otherwise, return a 1d interpolation function.

    Returns
    -------
    interpolation : callable
       An interpolation function.
       '''
    if callable(arg):
        return arg
    else:
        k, Pk = arg
        return interp1d(k, Pk, kind='quadratic')


@attr.s
class FieldHandler(object):
    Ndim = attr.ib(converter=int)
    Lbox = attr.ib(converter=float)
    dimensions = attr.ib(converter=int)
    Pk = attr.ib(converter=build_1dinterpolator)
    filter = attr.ib()
    sigma8 = attr.ib(converter=float)
    seed = attr.ib(converter=int, factory=lambda: np.random.randint(1<<32))

    constrains = attr.ib(factory=list)

    white_noise = attr.ib()
    white_noise_fft = None

    _smoothed = attr.ib(factory=dict)
    _grid = attr.ib(default=None)
    _xi = attr.ib(default=None)
    _covariance = attr.ib(default=None)

    covariance_cache = attr.ib(factory=dict)
    use_covariance_cache = attr.ib(default=True)

    _k2Pk = None

    def __attrs_post_init__(self):
        '''Precompute some data.'''

        k = self.Pk.x
        Pk = self.Pk(k)
        L = self.Lbox
        N = 1j * self.dimensions

        self._k2Pk = k**2 * Pk
        self._grid = np.mgrid[0:L:N, 0:L:N, 0:L:N]
        np.random.seed(self.seed)

    @white_noise.default
    def generate_white_noise(self):
        '''Generate a white noise with the relevant power spectrum.

        Returns
        -------
        white_noise : np.ndarray
        '''
        # Compute the k grid
        d = self.Lbox / self.dimensions / (2*np.pi)
        all_k = ([fft.fftfreq(self.dimensions, d=d)] * (self.Ndim-1) +
                 [fft.rfftfreq(self.dimensions, d=d)])

        self.kgrid = kgrid = np.array(np.meshgrid(*all_k, indexing='ij'))
        self.knorm = knorm = np.sqrt(ne.evaluate('sum(kgrid**2, axis=0)'))

        # Compute Pk
        Pk = np.zeros_like(knorm)
        mask = knorm > 0
        Pk[mask] = self.Pk(knorm[mask])

        # Compute white noise (in Fourier space)
        norm = np.sqrt(Pk/2) / self.Lbox
        deltak_real = np.random.normal(size=knorm.shape) * norm
        deltak_imag = np.random.normal(size=knorm.shape) * norm

        deltak = deltak_real + 1j*deltak_imag

        # Compute field in real space
        white_noise = fft.irfftn(deltak)

        # Normalize variance
        deltak_smoothed = deltak * self.filter.W(knorm)
        field = fft.irfftn(deltak_smoothed)
        std = field.std()

        self.white_noise_fft = deltak * self.sigma8 / std
        self.white_noise = white_noise * self.sigma8 / std

        return self.white_noise

    def get_smoothed(self, filter):
        if filter in self._smoothed:
            return self._smoothed[filter]
        knorm = self.knorm
        fft_field = self.white_noise_fft * filter.W(knorm)
        field = fft.irfftn(fft_field)
        self._smoothed[filter] = field
        return field

    def get_grid(self):
        return self._grid

    def add_constrain(self, constrain):
        '''Add a constrain to the GRF.

        Parameters
        ----------
        constrain : Constrain object
           The object describing the constrain.'''
        self._covariance = None
        self.constrains.append(constrain)

    def compute_covariance(self, frame='original'):
        '''Compute the covariance between the constrain points.'''
        if self._covariance is not None:
            return self._covariance

        data = dict()
        for i1, c1 in enumerate(self.constrains):
            for i2, c2 in enumerate(self.constrains):
                if i2 < i1:
                    continue
                tmp = c1.compute_covariance(c2, frame)
                data[i1, i2] = tmp.T
                data[i2, i1] = tmp

        Nc = len(self.constrains)
        self._covariance = np.block([[data[i1, i2] for i1 in range(Nc)] for i2 in range(Nc)])
        return self._covariance

    def compute_xi(self):
        '''Compute the correlation function between the constrain and the field'''
        if self._xi is not None:
            return self._xi
        xi = []
        pos = np.swapaxes(self.get_grid(), 0, -1).copy()
        for c in self.constrains:
            xi.append(c.xi(pos))
        self._xi = np.concatenate(xi, axis=-1)[:, 0, :].copy()
        return self._xi

    def get_constrained(self, filter, std_target):
        self.normalize(filter, std_target)
        xi = self.compute_xi()
        xij = self.compute_covariance()
        xij_inv = np.linalg.inv(xij)

        # Get measured value at constrain location
        ctilde = []
        c0 = []
        for c in self.constrains:
            val = c.measure_compact()
            ctilde.append(val)
            c0.append(np.atleast_1d(c.value))

        ctilde = np.concatenate(ctilde)
        c0 = np.concatenate(c0)

        # Maks nan values in constrain
        mask = ~np.isnan(c0)

        xi = xi[:, mask]
        xij = xij[:, mask][mask, :]
        xij_inv = np.linalg.inv(xij)
        c0 = c0[mask]
        ctilde = ctilde[mask]

        print(f'Applying {mask.sum()} constrains')

        # Compute constrained field
        ftilde = self.get_smoothed(self.filter).flatten()

        path, _ = contract_path('ai,ij,bj->ba', xi, xij_inv, [ctilde, ctilde], optimize='optimal')
        ftilde_bar, f_bar = contract('ai,ij,bj->ba', xi, xij_inv, [ctilde, c0], optimize=path)

        f = ftilde + f_bar - ftilde_bar

        shape = (128, 128, 128)
        f = f.reshape(shape)
        ftilde = ftilde.reshape(shape)
        ftilde_bar = ftilde_bar.reshape(shape)
        f_bar = f_bar.reshape(shape)

        ConstrainedField = namedtuple('ConstrainedField', ['f', 'ftilde', 'f_bar', 'ftilde_bar', 'ctilde'])

        return ConstrainedField(f=f, ftilde=ftilde, f_bar=f_bar, ftilde_bar=ftilde_bar,
                                ctilde=ctilde)

    def sigma(self, N, filter=None):
        k = self.Pk.x
        Pk = self.Pk.y

        if filter is None:
            filter = self.filter

        kpower = 2 + 2*N
        return np.sqrt(np.trapz(k**kpower * Pk * filter.W(k)**2, k) / (2*np.pi**2))
