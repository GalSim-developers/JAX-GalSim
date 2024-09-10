import galsim as _galsim
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from jax_galsim.core.utils import cast_to_float, ensure_hashable, implements
from jax_galsim.errors import GalSimError, GalSimIncompatibleValuesError
from jax_galsim.image import Image, ImageD
from jax_galsim.random import BaseDeviate, GaussianDeviate, PoissonDeviate


@implements(_galsim.noise.addNoise)
def addNoise(self, noise):
    # This will be inserted into the Image class as a method.  So self = image.
    noise.applyTo(self)


@implements(_galsim.noise.addNoiseSNR)
def addNoiseSNR(self, noise, snr, preserve_flux=False):
    # This will be inserted into the Image class as a method.  So self = image.
    noise_var = noise.getVariance()
    sumsq = jnp.sum(self.array**2, dtype=float)
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


@implements(_galsim.noise.BaseNoise)
@register_pytree_node_class
class BaseNoise:
    def __init__(self, rng=None):
        if rng is None:
            self._rng = BaseDeviate()
        else:
            if not isinstance(rng, BaseDeviate):
                raise TypeError("rng must be a galsim.BaseDeviate instance.")
            # we link the noise fields to the RNG state
            self._rng = rng

    @property
    @implements(_galsim.noise.BaseNoise.rng)
    def rng(self):
        return self._rng

    @implements(_galsim.noise.BaseNoise.getVariance)
    def getVariance(self):
        return self._getVariance()

    def _getVariance(self):
        raise NotImplementedError("Cannot call getVariance on a pure BaseNoise object")

    @implements(_galsim.noise.BaseNoise.withVariance)
    def withVariance(self, variance):
        return self._withVariance(variance)

    def _withVariance(self, variance):
        raise NotImplementedError("Cannot call withVariance on a pure BaseNoise object")

    @implements(_galsim.noise.BaseNoise.withScaledVariance)
    def withScaledVariance(self, variance_ratio):
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

    @implements(_galsim.noise.BaseNoise.applyTo)
    def applyTo(self, image):
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
        """This function flattens the BaseNoise into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        children = (self._rng,)
        # Define auxiliary static data that doesn’t need to be traced
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        return cls(rng=children[0])


@implements(_galsim.noise.GaussianNoise)
@register_pytree_node_class
class GaussianNoise(BaseNoise):
    def __init__(self, rng=None, sigma=1.0):
        super().__init__(GaussianDeviate(rng, sigma=sigma))
        self._sigma = cast_to_float(sigma)

    @property
    @implements(_galsim.noise.GaussianNoise.sigma)
    def sigma(self):
        return self._sigma

    def _applyTo(self, image):
        image._array = (image._array + self._rng.generate(image._array)).astype(
            image.dtype
        )

    def _getVariance(self):
        return self.sigma**2

    def _withVariance(self, variance):
        return GaussianNoise(self.rng, jnp.sqrt(variance))

    def _withScaledVariance(self, variance_ratio):
        return GaussianNoise(self.rng, self.sigma * jnp.sqrt(variance_ratio))

    @implements(
        _galsim.noise.GaussianNoise.copy,
        lax_description="JAX-GalSim RNGs cannot be shared so a copy is made if None is given.",
    )
    def copy(self, rng=None):
        if rng is None:
            rng = self.rng
        return GaussianNoise(rng=rng, sigma=self.sigma)

    def __repr__(self):
        return "galsim.GaussianNoise(rng=%r, sigma=%r)" % (
            self.rng,
            ensure_hashable(self.sigma),
        )

    def __str__(self):
        return "galsim.GaussianNoise(sigma=%s)" % (ensure_hashable(self.sigma))

    def tree_flatten(self):
        """This function flattens the GaussianNoise into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        children = (self._sigma, self._rng)
        # Define auxiliary static data that doesn’t need to be traced
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        return cls(sigma=children[0], rng=children[1])


@implements(_galsim.noise.PoissonNoise)
@register_pytree_node_class
class PoissonNoise(BaseNoise):
    def __init__(self, rng=None, sky_level=0.0):
        super().__init__(PoissonDeviate(rng))
        self._sky_level = cast_to_float(sky_level)

    @property
    @implements(_galsim.noise.PoissonNoise.sky_level)
    def sky_level(self):
        return self._sky_level

    def _applyTo(self, image):
        noise_array = image.array.copy().astype(float)

        # Minor subtlety for integer images.  It's a bit more consistent to convert to an
        # integer with the sky still added and then subtract off the sky.  But this isn't quite
        # right if the sky has a fractional part.  So only subtract off the integer part of the
        # sky at the end.  For float images, you get the same answer either way, so it doesn't
        # matter.
        frac_sky = self.sky_level - image.dtype(self.sky_level)
        int_sky = self.sky_level - frac_sky

        noise_array = jax.lax.cond(
            self.sky_level != 0.0,
            lambda na, sl: na + sl,
            lambda na, sl: na,
            noise_array,
            self.sky_level,
        )
        # Make sure no negative values
        noise_array = jnp.clip(noise_array, 0.0)
        # The noise_image now has the expectation values for each pixel with the sky added.
        noise_array = self._rng.generate_from_expectation(noise_array)
        # Subtract off the sky, since we don't want it in the final image.
        noise_array = jax.lax.cond(
            frac_sky != 0.0,
            lambda na, fs: na - fs,
            lambda na, fs: na,
            noise_array,
            frac_sky,
        )
        # Noise array is now the correct value for each pixel.
        image._array = noise_array.astype(image.dtype)
        image._array = jax.lax.cond(
            int_sky != 0.0,
            lambda na, ints: (na - ints).astype(float),
            lambda na, ints: na.astype(float),
            image._array,
            int_sky,
        ).astype(image.dtype)

    def _getVariance(self):
        return self.sky_level

    def _withVariance(self, variance):
        return PoissonNoise(self.rng, variance)

    def _withScaledVariance(self, variance_ratio):
        return PoissonNoise(self.rng, self.sky_level * variance_ratio)

    @implements(
        _galsim.noise.PoissonNoise.copy,
        lax_description="JAX-GalSim RNGs cannot be shared so a copy is made if None is given.",
    )
    def copy(self, rng=None):
        if rng is None:
            rng = self.rng
        return PoissonNoise(rng=rng, sky_level=self.sky_level)

    def __repr__(self):
        return "galsim.PoissonNoise(rng=%r, sky_level=%r)" % (self.rng, self.sky_level)

    def __str__(self):
        return "galsim.PoissonNoise(sky_level=%s)" % (self.sky_level)

    def tree_flatten(self):
        """This function flattens the PoissonNoise into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        children = (self._sky_level, self._rng)
        # Define auxiliary static data that doesn’t need to be traced
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        return cls(sky_level=children[0], rng=children[1])


@implements(_galsim.noise.CCDNoise)
@register_pytree_node_class
class CCDNoise(BaseNoise):
    def __init__(self, rng=None, sky_level=0.0, gain=1.0, read_noise=0.0):
        super().__init__(rng)
        self._sky_level = cast_to_float(sky_level)
        self._gain = cast_to_float(gain)
        self._read_noise = cast_to_float(read_noise)

    @property
    def _pd(self):
        return PoissonDeviate(self.rng)

    @property
    def _gd(self):
        return jax.lax.cond(
            self.gain > 0.0,
            lambda rng, read_noise, gain: GaussianDeviate(rng, sigma=read_noise / gain),
            lambda rng, read_noise, gain: GaussianDeviate(rng, sigma=read_noise),
            self.rng,
            self.read_noise,
            self.gain,
        )

    @property
    @implements(_galsim.noise.CCDNoise.sky_level)
    def sky_level(self):
        return self._sky_level

    @property
    @implements(_galsim.noise.CCDNoise.gain)
    def gain(self):
        return self._gain

    @property
    @implements(_galsim.noise.CCDNoise.read_noise)
    def read_noise(self):
        return self._read_noise

    def _applyTo(self, image):
        noise_array = image.array.copy().astype(float)

        # cf. PoissonNoise._applyTo function
        frac_sky = self.sky_level - image.dtype(self.sky_level)  # 0 if dtype = float
        int_sky = self.sky_level - frac_sky

        noise_array = jax.lax.cond(
            self.sky_level != 0.0,
            lambda na, sl: na + sl,
            lambda na, sl: na,
            noise_array,
            self.sky_level,
        )

        # First add the poisson noise from the signal + sky:
        noise_array = jax.lax.cond(
            self.gain > 0.0,
            lambda pd, na, gain: (
                pd.generate_from_expectation(jnp.clip(na * gain, 0.0)) / gain
            ),
            lambda pd, na, gain: na,
            self._pd,
            noise_array,
            self.gain,
        )

        # Now add the read noise:
        noise_array = jax.lax.cond(
            self.read_noise > 0.0,
            lambda na, gd: na + gd.generate(na),
            lambda na, gd: na,
            noise_array,
            self._gd,
        )

        noise_array = jax.lax.cond(
            frac_sky != 0.0,
            lambda na, fs: na - fs,
            lambda na, fs: na,
            noise_array,
            frac_sky,
        )
        # Noise array is now the correct value for each pixel.
        image._array = noise_array.astype(image.dtype)
        image._array = jax.lax.cond(
            int_sky != 0.0,
            lambda na, ints: (na - ints).astype(float),
            lambda na, ints: na.astype(float),
            image._array,
            int_sky,
        ).astype(image.dtype)

    def _getVariance(self):
        return jax.lax.cond(
            self.gain > 0.0,
            lambda gain, sky_level, read_noise: sky_level / gain
            + (read_noise / gain) ** 2,
            lambda gain, sky_level, read_noise: read_noise**2,
            self.gain,
            self.sky_level,
            self.read_noise,
        )

    def _withVariance(self, variance):
        current_var = self._getVariance()
        return jax.lax.cond(
            current_var > 0.0,
            lambda variance, current_var: self._withScaledVariance(
                variance / current_var
            ),
            lambda variance, current_var: CCDNoise(self.rng, sky_level=variance),
            variance,
            current_var,
        )

    def _withScaledVariance(self, variance_ratio):
        return CCDNoise(
            self.rng,
            gain=self.gain,
            sky_level=self.sky_level * variance_ratio,
            read_noise=self.read_noise * jnp.sqrt(variance_ratio),
        )

    @implements(
        _galsim.noise.CCDNoise.copy,
        lax_description="JAX-GalSim RNGs cannot be shared so a copy is made if None is given.",
    )
    def copy(self, rng=None):
        if rng is None:
            rng = self.rng
        return CCDNoise(rng, self.sky_level, self.gain, self.read_noise)

    def __repr__(self):
        return "galsim.CCDNoise(rng=%r, sky_level=%r, gain=%r, read_noise=%r)" % (
            self.rng,
            self.sky_level,
            self.gain,
            self.read_noise,
        )

    def __str__(self):
        return "galsim.CCDNoise(sky_level=%r, gain=%r, read_noise=%r)" % (
            self.sky_level,
            self.gain,
            self.read_noise,
        )

    def tree_flatten(self):
        """This function flattens the CCDNoise into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        children = (self.rng, self.sky_level, self.gain, self.read_noise)
        # Define auxiliary static data that doesn’t need to be traced
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        return cls(
            rng=children[0],
            sky_level=children[1],
            gain=children[2],
            read_noise=children[3],
        )


@implements(_galsim.noise.DeviateNoise)
@register_pytree_node_class
class DeviateNoise(BaseNoise):
    def __init__(self, dev):
        super().__init__(dev)

    def _applyTo(self, image):
        image._array = (image._array + self._rng.generate(image._array)).astype(
            image.dtype
        )

    def _getVariance(self):
        raise GalSimError("No single variance value for DeviateNoise")

    def _withVariance(self, variance):
        raise GalSimError("Changing the variance is not allowed for DeviateNoise")

    def _withScaledVariance(self, variance):
        raise GalSimError("Changing the variance is not allowed for DeviateNoise")

    @implements(
        _galsim.noise.DeviateNoise.copy,
        lax_description="JAX-GalSim RNGs cannot be shared so a copy is made if None is given.",
    )
    def copy(self, rng=None):
        if rng is None:
            dev = self.rng
        else:
            # Slightly different this time, since we want to make sure that we keep the same
            # kind of deviate, but just reset it to follow the given rng.
            dev = self.rng.duplicate()
            dev.reset(rng)
        return DeviateNoise(dev)

    def __repr__(self):
        return "galsim.DeviateNoise(dev=%r)" % self.rng

    def __str__(self):
        return "galsim.DeviateNoise(dev=%s)" % self.rng

    def tree_flatten(self):
        """This function flattens the DeviateNoise into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        children = (self._rng,)
        # Define auxiliary static data that doesn’t need to be traced
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        return cls(rng=children[0])


@implements(_galsim.noise.VariableGaussianNoise)
@register_pytree_node_class
class VariableGaussianNoise(BaseNoise):
    def __init__(self, rng, var_image):
        super().__init__(GaussianDeviate(rng))

        # Make sure var_image is an ImageD, converting dtype if necessary
        self._var_image = ImageD(var_image)

    @property
    @implements(_galsim.noise.VariableGaussianNoise.var_image)
    def var_image(self):
        return self._var_image

    # Repeat this here, since we want to add an extra sanity check, which should go in the
    # non-underscore version.
    @implements(_galsim.noise.VariableGaussianNoise.applyTo)
    def applyTo(self, image):
        if not isinstance(image, Image):
            raise TypeError("Provided image must be a galsim.Image")
        if image.array.shape != self.var_image.array.shape:
            raise GalSimIncompatibleValuesError(
                "Provided image shape does not match the shape of var_image",
                image=image,
                var_image=self.var_image,
            )
        return self._applyTo(image)

    def _applyTo(self, image):
        # jax galsim never fills an image so this is safe
        noise_array = self._rng.generate_from_variance(self.var_image.array)
        image._array = image._array + noise_array.astype(image.dtype)

    @implements(
        _galsim.noise.VariableGaussianNoise.copy,
        lax_description="JAX-GalSim RNGs cannot be shared so a copy is made if None is given.",
    )
    def copy(self, rng=None):
        if rng is None:
            rng = self.rng
        return VariableGaussianNoise(rng, self.var_image)

    def _getVariance(self):
        raise GalSimError("No single variance value for VariableGaussianNoise")

    def _withVariance(self, variance):
        raise GalSimError(
            "Changing the variance is not allowed for VariableGaussianNoise"
        )

    def _withScaledVariance(self, variance):
        # This one isn't undefined like withVariance, but it's inefficient.  Better to
        # scale the values in the image before constructing VariableGaussianNoise.
        raise GalSimError(
            "Changing the variance is not allowed for VariableGaussianNoise"
        )

    def __repr__(self):
        return "galsim.VariableGaussianNoise(rng=%r, var_image=%r)" % (
            self.rng,
            self.var_image,
        )

    def __str__(self):
        return "galsim.VariableGaussianNoise(var_image=%s)" % (self.var_image)

    def tree_flatten(self):
        """This function flattens the VariableGaussianNoise into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        children = (self._rng, self._var_image)
        # Define auxiliary static data that doesn’t need to be traced
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        return cls(children[0], children[1])
