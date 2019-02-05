'''Classes to represent a Gaussian Random field.'''
import numpy as np
import attr
import pyfftw.interfaces.numpy_fft as fft
from scipy.interpolate import interp1d


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

    constrains = attr.ib(factory=list)

    white_noise = attr.ib()
    white_noise_fft = None

    _smoothed = attr.ib(factory=dict)
    _grid = attr.ib(default=None)

    covariance = attr.ib(default=None)
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

    @white_noise.default
    def generate_white_noise(self):
        '''Generate a white noise with the relevant power spectrum.

        Returns
        -------
        white_noise : np.ndarray
        '''
        # Compute the k grid
        D = self.Lbox / self.dimensions / (2*np.pi)
        all_k = ([fft.fftfreq(self.dimensions, d=D)
                 for _ in range(self.Ndim-1)] +
                 [fft.rfftfreq(self.dimensions, d=D)])

        self.kgrid = kgrid = np.array(np.meshgrid(*all_k, indexing='ij'))
        self.knorm = knorm = np.sqrt(np.sum(kgrid**2, axis=0))

        # Compute Pk
        sigmas = np.zeros_like(knorm)
        mask = knorm > 0
        sigmas[mask] = np.sqrt(self.Pk(knorm[mask]))

        # Compute white noise (in Fourier space)
        norm = np.random.rayleigh(sigmas)
        phi = 2*np.pi*np.random.rand(*sigmas.shape)

        z = norm * np.exp(1j*phi)

        # Compute field in real space
        white_noise = fft.irfftn(z)

        # Compute field variance
        k2Pk = self.Pk.x**2 * self.Pk.y
        std = white_noise.std()
        std_target = np.sqrt(np.trapz(k2Pk / (2*np.pi**2), self.Pk.x))

        factor = std_target / std

        self.white_noise_fft = z * factor
        self.white_noise = white_noise * factor
        return white_noise

    def normalize(self, filter, std_target):
        '''Normalize the white noise using the filter to a given value
        
        Params
        ------
        filter : Filter object
        std_target : float
            The variance of the smoothed field.'''
        std_measured = self.get_smoothed(filter).std()
        factor = std_target / std_measured
        self.white_noise_fft *= factor
        self.white_noise *= factor
        
        self._smoothed = {}

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
        self.constrains.append(constrain)

    def compute_covariance(self, frame='original'):
        '''Compute the covariance between the constrain points.'''
        data = dict()
        for i1, c1 in enumerate(self.constrains):
            for i2, c2 in enumerate(self.constrains):
                if i2 < i1:
                    continue
                tmp = c1.compute_covariance(c2, frame)
                data[i1, i2] = tmp.T
                data[i2, i1] = tmp

        Nc = len(self.constrains)
        return np.block([[data[i1, i2] for i1 in range(Nc)] for i2 in range(Nc)])

    def sigma(self, N):
        k2Pk = self._k2Pk
        k = self.Pk.x
        Pk = self.Pk.y

        kpower = 2 + 2*N
        return np.sqrt(np.trapz(k**kpower * Pk * self.filter.W(k)**2, k) / (2*np.pi**2))
