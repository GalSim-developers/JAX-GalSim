import jax
import jax.numpy as jnp
import numpy as np
import math

from jax._src.numpy.util import _wraps
from jax.tree_util import register_pytree_node_class

import galsim as _galsim

from jax_galsim.image import Image, ImageD

_wraps(_galsim.noise.addNoise)


def addNoise(self, noise):
    noise.applyTo(self)


_wraps(_galsim.noise.addNoiseSNR)


def addNoiseSNR(self, noise, snr, preserve_flux=False):
    noise_var = noise.getVariance()
    sumsq = jnp.sum(self.array**2)
    if preserve_flux:
        new_noise_var = sumsq / snr / snr
        noise = noise.withVariance(new_noise_var)
        self.addNoise(noise)
        return new_noise_var
    else:
        sn_meas = jnp.sqrt(sumsq / noise_var)
        flux = snr / sn_meas
        self *= flux
        self.addNoise(noise)
        return noise_var


Image.addNoise = addNoise
Image.addNoiseSNR = addNoiseSNR


@register_pytree_node_class
class BaseNoise:
    _wraps(_galsim.BaseNoise)

    def __init__(self, rng=None):
        from jax_galsim.random import BaseDeviate

        if rng is None:
            self._rng = BaseDeviate()
        else:
            if not isinstance(rng, BaseDeviate):
                raise TypeError("rng must be a galsim.BaseDeviate instance.")
            self._rng = rng

    @property
    def rng(self):
        """The `BaseDeviate` of this noise object."""
        return self._rng

    def getVariance(self):
        """Get variance in current noise model."""
        return self._getVariance()

    def _getVariance(self):
        raise NotImplementedError("Cannot call getVariance on a pure BaseNoise object")

    def withVariance(self, variance):
        """Return a new noise object (of the same type as the current one) with the specified
        variance.

        Parameters:
            variance:   The desired variance in the noise.

        Returns:
            a new Noise object with the given variance.
        """
        return self._withVariance(variance)

    def _withVariance(self, variance):
        raise NotImplementedError("Cannot call withVariance on a pure BaseNoise object")

    def withScaledVariance(self, variance_ratio):
        """Return a new noise object with the variance scaled up by the specified factor.

        This is equivalent to noise * variance_ratio.

        Parameters:
            variance_ratio: The factor by which to scale the variance of the correlation
                            function profile.

        Returns:
            a new Noise object whose variance has been scaled by the given amount.
        """
        return self._withScaledVariance(variance_ratio)

    def _withScaledVariance(self, variance_ratio):
        raise NotImplementedError(
            "Cannot call withScaledVariance on a pure BaseNoise object"
        )

    def __mul__(self, variance_ratio):
        """Multiply the variance of the noise by ``variance_ratio``.

        Parameters:
            variance_ratio: The factor by which to scale the variance of the correlation
                            function profile.

        Returns:
            a new Noise object whose variance has been scaled by the given amount.
        """
        return self.withScaledVariance(variance_ratio)

    def __div__(self, variance_ratio):
        """Equivalent to self * (1/variance_ratio)"""
        return self.withScaledVariance(1.0 / variance_ratio)

    __rmul__ = __mul__
    __truediv__ = __div__

    def applyTo(self, image):
        """Add noise to an input `Image`.

        e.g.::

            >>> noise.applyTo(image)

        On output the `Image` instance ``image`` will have been given additional noise according
        to the current noise model.

        Note: This is equivalent to the alternate syntax::

            >>> image.addNoise(noise)

        which may be more convenient or clearer.
        """
        if not isinstance(image, Image):
            raise TypeError("Provided image must be a galsim.Image")
        return self._applyTo(image)

    def _applyTo(self, image):
        raise NotImplementedError("Cannot call applyTo on a pure BaseNoise object")

    def __eq__(self, other):
        # Quick and dirty.  Just check reprs are equal.
        return self is other or repr(self) == repr(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = None

    def tree_flatten(self):
        """Flatten the noise object."""
        children = (self._rng,)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        del aux_data
        obj = object.__new__(cls)
        obj._rng = children[0]
        return obj


@register_pytree_node_class
class GaussianNoise(BaseNoise):
    @_wraps(_galsim.GaussianNoise)
    def __init__(self, rng=None, sigma=1.0):
        from jax_galsim.random import GaussianDeviate

        BaseNoise.__init__(self, rng)
        self._sigma = sigma
        self._gd = GaussianDeviate(self.rng, sigma=sigma)

    @property
    def sigma(self):
        """The input sigma value."""
        return self._sigma

    def _applyTo(self, image):
        # Extract next seed
        image._array = self._gd.add_generate(image._array)
        return image

    def _getVariance(self):
        return self.sigma**2

    def _withVariance(self, variance):
        return GaussianNoise(self.rng, math.sqrt(variance))

    def _withScaledVariance(self, variance_ratio):
        return GaussianNoise(self.rng, self.sigma * math.sqrt(variance_ratio))

    def copy(self, rng=None):
        """Returns a copy of the Gaussian noise model.

        By default, the copy will share the `BaseDeviate` random number generator with the parent
        instance.  However, you can provide a new rng to use in the copy if you want with::

            >>> noise_copy = noise.copy(rng=new_rng)
        """
        if rng is None:
            rng = self.rng
        return GaussianNoise(rng, self.sigma)

    def __repr__(self):
        return "galsim.GaussianNoise(rng=%r, sigma=%r)" % (self.rng, self.sigma)

    def __str__(self):
        return "galsim.GaussianNoise(sigma=%s)" % (self.sigma)

    def tree_flatten(self):
        """Flatten the noise object."""
        children = (self._rng, self._sigma)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        del aux_data
        obj = object.__new__(cls)
        obj._rng = children[0]
        obj._sigma = children[1]
        return obj
