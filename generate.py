'''Generate and handle Gaussian Random Fields. This makes use of the
algorithm described in Hoffman & Ribak 1991.

Corentin Cadiou -- 2019'''
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class Constrain(object):
    '''A class to represent a linear constrain acting on a GRF'''

    value = 0  # Value of the constrain
    position = None  # Position of the constrain

    def __init__(self, pos, val):
        self.position = pos
        self.value = val

    def measure(self, pos, field, field_fft, crf):
        '''Measure the value of the constrain in the field.

        Parameters
        ----------
        points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
           The points defining the regular grid in n dimensions.pos : ndarray
        field : ndarray
           The value of the field.
        field_fft : ndarray, complex
           The discrete Fourier transform of the field
        crf : ConstrainedRandomField
           A constrained random field object.
        '''
        raise NotImplementedError

    def xi(self, pos):
        '''Evaluate the correlation function between the field and its
        functional at the location.'''
        raise NotImplementedError


def compute_gradient(field, field_fft, crf):
    kgrid = crf.kgrid
    Ndim = len(kgrid)

    grad_fft = np.zeros([Ndim] + field_fft.shape, dtype='complex')
    for idim in range(Ndim):
        grad_fft[idim, ...] = field_fft * (1j) * kgrid[idim]

    grad = np.fft.irfftn(grad_fft, axes=range(1, Ndim+1))
    crf.data['gradient'] = grad


class FieldConstrain(Constrain):
    '''Constrain the field to a given value.'''
    def measure(self, pos, field, field_fft, crf):
        field_interp = RegularGridInterpolator(pos, field, method='linear')

        return field_interp(self.position)


class GradientFieldConstrain(Constrain):
    '''Constrain the gradient of the field to a given value.'''
    def measure(self, pos, field, field_fft, crf):
        if 'gradient' in crf.data:
            grad = crf.data['gradient']
        else:
            kgrid = crf.kgrid
            Ndim = len(kgrid)

            grad_fft = np.zeros([Ndim] + field_fft.shape, dtype='complex')
            for idim in range(Ndim):
                grad_fft[idim, ...] = field_fft * (1j) * kgrid[idim]

            grad = np.fft.irfftn(grad_fft, axes=range(1, Ndim+1))
            crf.data['gradient'] = grad

        grad = np.swapaxes(grad, 0, -1)
        grad_field_interp = RegularGridInterpolator(pos, grad, method='linear')
        return grad_field_interp(self.position)


class ConstrainedRandomField(object):
    '''A class to represent gaussian random fields subject to constrains.'''

    k = None      # k-grid [(Mpc/h)^-1]
    Pk = None     # Power-spectrum
    boxlen = None # Size of box in physical units

    constrains = []  # A list of Constrain objects
    pos_grid = None  # A grid of positions where the field will be evaluated

    def __init__(self, boxlen, k, Pk, Npt=128):
        self.boxlen = boxlen
        self.k = k
        self.Pk = Pk
        self.Npt = Npt

        x = np.linspace(0, boxlen, Npt)
        self.pos_grid = np.array(np.meshgrid(x, x, x, indexing='ij'))

    def add_constrain(self, constrain):
        self.constrains.append(constrain)

    def compute_constrain_covariance():
        raise NotImplementedError

    def draw(self, return_mean=False):
        '''Draw a random field given the constrains.

        Parameters
        ----------
        return_mean : boolean
           If True, return the mean value.

        Returns
        -------
        f : ndarray
           A realisation of the Gaussian random field on the grid
        fmean : ndarray
           Only returned if return_mean is True. The mean value of the
           field on the grid.
        '''
        # Setup
        constrains = self.constrains
        xij = self.compute_constrain_covariance()
        xij_inv = np.linalg.inv(xij)
        c = np.array([c.value for c in constrains])

        # 1. Generate a GRF with given power spectrum -- no constrain
        ftilde = None
        fft_ftilde = np.fft.fftn(ftilde)

        # 2. Measure value of constrains in the field
        ctilde = np.zeros(len(constrains))
        xi = [None] * len(constrains)
        for i, constrain in enumerate(constrains):
            ctilde[i] = constrain.measure(ftilde, fft_ftilde)
            xi[i] = constrain.xi(self.pos_grid)

        # (Nconstrain, Npt, ..., Npt) -> (Npt, ..., Npt, Nconstrain)
        xi = np.swapaxes(xi, 0, -1)

        xi_xij_inv = np.einsum('...i,ij->j', xi, xij_inv)

        # 3. Compute ftilde_mean
        ftilde_mean = np.dot(xi_xij_inv, ctilde)

        # 5. Compute field under Gamma constrains
        f_mean = np.dot(xi_xij_inv, c)
        f = ftilde + f_mean - ftilde_mean

        if return_mean:
            return f, f_mean
        else:
            return f
