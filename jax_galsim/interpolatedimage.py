import textwrap
import jax.numpy as jnp
from jax._src.numpy.util import _wraps
from jax.tree_util import register_pytree_node_class
import math

import galsim as _galsim
from galsim.utilities import doc_inherit
from galsim.errors import GalSimRangeError, GalSimValueError, GalSimUndefinedBoundsError
from galsim.errors import GalSimIncompatibleValuesError

from jax_galsim.gsobject import GSObject
from jax_galsim.gsparams import GSParams
from jax_galsim.image import Image
from jax_galsim.position import PositionD
from jax_galsim.interpolant import Quintic
from jax_galsim.utilities import convert_interpolant
from jax_galsim.bounds import BoundsI


@_wraps(
    _galsim.InterpolatedImage,
    lax_description=textwrap.dedent(
        """The JAX equivalent of galsim.InterpolatedImage does not support

            - noise padding
            - depixelize
            - reading images from FITS files

        """
    ),
)
@register_pytree_node_class
class InterpolatedImage(GSObject):
    _req_params = {'image': str}
    _opt_params = {
        'x_interpolant': str,
        'k_interpolant': str,
        'normalization': str,
        'scale': float,
        'flux': float,
        'pad_factor': float,
        'noise_pad_size': float,
        'noise_pad': str,
        'pad_image': str,
        'calculate_stepk': bool,
        'calculate_maxk': bool,
        'use_true_center': bool,
        'depixelize': bool,
        'offset': PositionD,
        'hdu': int
    }
    _takes_rng = True
    _cache_noise_pad = {}

    _has_hard_edges = False
    _is_axisymmetric = False
    _is_analytic_x = True
    _is_analytic_k = True

    def __init__(self, image, x_interpolant=None, k_interpolant=None, normalization='flux',
                 scale=None, wcs=None, flux=None, pad_factor=4., noise_pad_size=0, noise_pad=0.,
                 rng=None, pad_image=None, calculate_stepk=True, calculate_maxk=True,
                 use_cache=True, use_true_center=True, depixelize=False, offset=None,
                 gsparams=None, _force_stepk=0., _force_maxk=0., hdu=None):

        from .wcs import BaseWCS, PixelScale
        # FIXME: no BaseDeviate in jax_galsim
        # from .random import BaseDeviate

        # If the "image" is not actually an image, try to read the image as a file.
        if isinstance(image, str):
            # FIXME: no FITSIO in jax_galsim
            # image = fits.read(image, hdu=hdu)
            raise NotImplementedError(
                "Reading InterpolatedImages from FITS files is not implemented in jax_galsim."
            )
        elif not isinstance(image, Image):
            raise TypeError("Supplied image must be an Image or file name")

        # it must have well-defined bounds, otherwise seg fault in SBInterpolatedImage constructor
        if not image.bounds.isDefined():
            raise GalSimUndefinedBoundsError("Supplied image does not have bounds defined.")

        # check what normalization was specified for the image: is it an image of surface
        # brightness, or flux?
        if normalization.lower() not in ("flux", "f", "surface brightness", "sb"):
            raise GalSimValueError("Invalid normalization requested.", normalization,
                                   ('flux', 'f', 'surface brightness', 'sb'))

        # Set up the interpolants if none was provided by user, or check that the user-provided ones
        # are of a valid type
        self._gsparams = GSParams.check(gsparams)
        if x_interpolant is None:
            self._x_interpolant = Quintic(gsparams=self._gsparams)
        else:
            self._x_interpolant = convert_interpolant(x_interpolant).withGSParams(self._gsparams)
        if k_interpolant is None:
            self._k_interpolant = Quintic(gsparams=self._gsparams)
        else:
            self._k_interpolant = convert_interpolant(k_interpolant).withGSParams(self._gsparams)

        # Store the image as an attribute and make sure we don't change the original image
        # in anything we do here.  (e.g. set scale, etc.)
        if depixelize:
            # FIXME: no depixelize in jax_galsim
            # self._image = image.view(dtype=np.float64).depixelize(self._x_interpolant)
            raise NotImplementedError("InterpolatedImages do not support 'depixelize' in jax_galsim.")
        else:
            self._image = image.view(dtype=jnp.float64, contiguous=True)
        self._image.setCenter(0, 0)

        # Set the wcs if necessary
        if scale is not None:
            if wcs is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot provide both scale and wcs to InterpolatedImage", scale=scale, wcs=wcs)
            self._image.wcs = PixelScale(scale)
        elif wcs is not None:
            if not isinstance(wcs, BaseWCS):
                raise TypeError("wcs parameter is not a galsim.BaseWCS instance")
            self._image.wcs = wcs
        elif self._image.wcs is None:
            raise GalSimIncompatibleValuesError(
                "No information given with Image or keywords about pixel scale!",
                scale=scale, wcs=wcs, image=image)

        # Figure out the offset to apply based on the original image (not the padded one).
        # We will apply this below in _sbp.
        offset = self._parse_offset(offset)
        self._offset = self._adjust_offset(self._image.bounds, offset, None, use_true_center)

        im_cen = image.true_center if use_true_center else image.center
        self._wcs = self._image.wcs.local(image_pos=im_cen)

        # Build the fully padded real-space image according to the various pad options.
        self._buildRealImage(pad_factor, pad_image, noise_pad_size, noise_pad, rng, use_cache)
        self._image_flux = jnp.sum(self._image.array, dtype=jnp.float64)

        # I think the only things that will mess up if flux == 0 are the
        # calculateStepK and calculateMaxK functions, and rescaling the flux to some value.
        if (calculate_stepk or calculate_maxk or flux is not None) and self._image_flux == 0.:
            raise GalSimValueError("This input image has zero total flux. It does not define a "
                                   "valid surface brightness profile.", image)

        # Process the different options for flux, stepk, maxk
        self._flux = self._getFlux(flux, normalization)
        self._calculate_stepk = calculate_stepk
        self._calculate_maxk = calculate_maxk
        self._stepk = self._getStepK(calculate_stepk, _force_stepk)
        self._maxk = self._getMaxK(calculate_maxk, _force_maxk)

    @doc_inherit
    def withGSParams(self, gsparams=None, **kwargs):
        if gsparams == self.gsparams:
            return self
        # Checking gsparams
        gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        # Flattening the representation to instantiate a clean new object
        children, aux_data = self.tree_flatten()
        aux_data["gsparams"] = gsparams
        ret = self.tree_unflatten(aux_data, children)

        ret._x_interpolant = self._x_interpolant.withGSParams(ret._gsparams, **kwargs)
        ret._k_interpolant = self._k_interpolant.withGSParams(ret._gsparams, **kwargs)
        if ret._gsparams.folding_threshold != self._gsparams.folding_threshold:
            ret._stepk = ret._getStepK(self._calculate_stepk, 0.)
        if ret._gsparams.maxk_threshold != self._gsparams.maxk_threshold:
            ret._maxk = ret._getMaxK(self._calculate_maxk, 0.)
        return ret

    def tree_flatten(self):
        """This function flattens the InterpolatedImage into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        children = (self.params,)
        # Define auxiliary static data that doesn’t need to be traced
        aux_data = {"gsparams": self.gsparams}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        return cls(**(children[0]), **aux_data)

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, InterpolatedImage) and
                 self._xim == other._xim and
                 self.x_interpolant == other.x_interpolant and
                 self.k_interpolant == other.k_interpolant and
                 self.flux == other.flux and
                 self._offset == other._offset and
                 self.gsparams == other.gsparams and
                 self._stepk == other._stepk and
                 self._maxk == other._maxk))

    # TODO: do this in JAX OFC
    # @lazy_property
    # def _sbp(self):
    #     min_scale = self._wcs._minScale()
    #     max_scale = self._wcs._maxScale()
    #     self._sbii = _galsim.SBInterpolatedImage(
    #             self._xim._image, self._image.bounds._b, self._pad_image.bounds._b,
    #             self._x_interpolant._i, self._k_interpolant._i,
    #             self._stepk*min_scale,
    #             self._maxk*max_scale,
    #             self.gsparams._gsp)

    #     self._sbp = self._sbii  # Temporary.  Will overwrite this with the return value.

    #     # Apply the offset
    #     prof = self
    #     if self._offset != _PositionD(0,0):
    #         # Opposite direction of what drawImage does.
    #         prof = prof._shift(-self._offset.x, -self._offset.y)

    #     # If the user specified a flux, then set to that flux value.
    #     if self._flux != self._image_flux:
    #         flux_ratio = self._flux / self._image_flux
    #     else:
    #         flux_ratio = 1.

    #     # Bring the profile from image coordinates into world coordinates
    #     # Note: offset needs to happen first before the transformation, so can't bundle it here.
    #     prof = self._wcs._profileToWorld(prof, flux_ratio, _PositionD(0,0))

    #     return prof._sbp

    def _buildRealImage(self, pad_factor, pad_image, noise_pad_size, noise_pad, rng, use_cache):
        # Check that given pad_image is valid:
        if pad_image is not None:
            if isinstance(pad_image, str):
                # FIXME: no FITSIO in jax_galsim
                # pad_image = fits.read(pad_image).view(dtype=np.float64)
                raise NotImplementedError(
                    "Reading padding images for InterpolatedImages from FITS files "
                    "is not implemented in jax_galsim."
                )
            elif isinstance(pad_image, Image):
                pad_image = pad_image.view(dtype=jnp.float64, contiguous=True)
            else:
                raise TypeError("Supplied pad_image must be an Image.", pad_image)

        if pad_factor <= 0.:
            raise GalSimRangeError("Invalid pad_factor <= 0 in InterpolatedImage", pad_factor, 0.)

        # Convert noise_pad_size from arcsec to pixels according to the local wcs.
        # Use the minimum scale, since we want to make sure noise_pad_size is
        # as large as we need in any direction.
        if noise_pad_size:
            # FIXME: no BaseDeviate in jax_galsim so no noise padding
            # if noise_pad_size < 0:
            #     raise GalSimValueError("noise_pad_size may not be negative", noise_pad_size)
            # if not noise_pad:
            #     raise GalSimIncompatibleValuesError(
            #             "Must provide noise_pad if noise_pad_size > 0",
            #             noise_pad=noise_pad, noise_pad_size=noise_pad_size)
            # noise_pad_size = int(math.ceil(noise_pad_size / self._wcs._minScale()))
            # noise_pad_size = Image.good_fft_size(noise_pad_size)
            raise NotImplementedError("InterpolatedImages do not support noise padding in jax_galsim.")
        else:
            if noise_pad:
                # FIXME: no BaseDeviate in jax_galsim so no noise padding
                # raise GalSimIncompatibleValuesError(
                #         "Must provide noise_pad_size if noise_pad != 0",
                #         noise_pad=noise_pad, noise_pad_size=noise_pad_size)
                raise NotImplementedError("InterpolatedImages do not support noise padding in jax_galsim.")

        # The size of the final padded image is the largest of the various size specifications
        pad_size = max(self._image.array.shape)
        if pad_factor > 1.:
            pad_size = int(math.ceil(pad_factor * pad_size))
        if noise_pad_size:
            pad_size = max(pad_size, noise_pad_size)
        if pad_image:
            pad_image.setCenter(0, 0)
            pad_size = max(pad_size, *pad_image.array.shape)
        # And round up to a good fft size
        pad_size = Image.good_fft_size(pad_size)

        self._xim = Image(pad_size, pad_size, dtype=jnp.float64, wcs=self._wcs)
        self._xim.setCenter(0, 0)

        # If requested, fill (some of) this image with noise padding.
        nz_bounds = self._image.bounds
        # FIXME: no BaseDeviate in jax_galsim so no noise padding
        # if noise_pad:
        #     # This is a bit involved, so pass this off to another helper function.
        #     b = self._buildNoisePadImage(noise_pad_size, noise_pad, rng, use_cache)
        #     nz_bounds += b

        # The the user gives us a pad image to use, fill the relevant portion with that.
        if pad_image:
            # assert self._xim.bounds.includes(pad_image.bounds)
            self._xim[pad_image.bounds] = pad_image
            nz_bounds += pad_image.bounds

        # Now place the given image in the center of the padding image:
        # assert self._xim.bounds.includes(self._image.bounds)
        self._xim[self._image.bounds] = self._image
        self._xim.wcs = self._wcs

        # And update the _image to be that portion of the full real image rather than the
        # input image.
        self._image = self._xim[self._image.bounds]

        # These next two allow for easy pickling/repring.  We don't need to serialize all the
        # zeros around the edge.  But we do need to keep any non-zero padding as a pad_image.
        self._pad_image = self._xim[nz_bounds]
        # self._pad_factor = (max(self._xim.array.shape)-1.e-6) / max(self._image.array.shape)
        self._pad_factor = pad_factor

    # FIXME: no BaseDeviate in jax_galsim so no noise padding
    # def _buildNoisePadImage(self, noise_pad_size, noise_pad, rng, use_cache):
    #     """A helper function that builds the ``pad_image`` from the given ``noise_pad``
    #     specification.
    #     """
    #     from .random import BaseDeviate
    #     from .noise import GaussianNoise
    #     from .correlatednoise import BaseCorrelatedNoise, CorrelatedNoise

    #     # Make sure we make rng a BaseDeviate if rng is None
    #     rng1 = BaseDeviate(rng)

    #     # Figure out what kind of noise to apply to the image
    #     try:
    #         noise_pad = float(noise_pad)
    #     except (TypeError, ValueError):
    #         if isinstance(noise_pad, BaseCorrelatedNoise):
    #             noise = noise_pad.copy(rng=rng1)
    #         elif isinstance(noise_pad, Image):
    #             noise = CorrelatedNoise(noise_pad, rng1)
    #         elif use_cache and noise_pad in InterpolatedImage._cache_noise_pad:
    #             noise = InterpolatedImage._cache_noise_pad[noise_pad]
    #             if rng:
    #                 # Make sure that we are using a specified RNG by resetting that in this cached
    #                 # CorrelatedNoise instance, otherwise preserve the cached RNG
    #                 noise = noise.copy(rng=rng1)
    #         elif isinstance(noise_pad, basestring):
    #             noise = CorrelatedNoise(fits.read(noise_pad), rng1)
    #             if use_cache:
    #                 InterpolatedImage._cache_noise_pad[noise_pad] = noise
    #         else:
    #             raise GalSimValueError(
    #                 "Input noise_pad must be a float/int, a CorrelatedNoise, Image, or filename "
    #                 "containing an image to use to make a CorrelatedNoise.", noise_pad)

    #     else:
    #         if noise_pad < 0.:
    #             raise GalSimRangeError("Noise variance may not be negative.", noise_pad, 0.)
    #         noise = GaussianNoise(rng1, sigma = np.sqrt(noise_pad))

    #     # Find the portion of xim to fill with noise.
    #     # It's allowed for the noise padding to not cover the whole pad image
    #     half_size = noise_pad_size // 2
    #     b = _BoundsI(-half_size, -half_size + noise_pad_size-1,
    #                  -half_size, -half_size + noise_pad_size-1)
    #     #assert self._xim.bounds.includes(b)
    #     noise_image = self._xim[b]
    #     # Add the noise
    #     noise_image.addNoise(noise)
    #     return b

    def _getFlux(self, flux, normalization):
        # If the user specified a surface brightness normalization for the input Image, then
        # need to rescale flux by the pixel area to get proper normalization.
        if flux is None:
            flux = self._image_flux
            if normalization.lower() in ('surface brightness', 'sb'):
                flux *= self._wcs.pixelArea()
        return flux

    def _getStepK(self, calculate_stepk, _force_stepk):
        # GalSim cannot automatically know what stepK and maxK are appropriate for the
        # input image.  So it is usually worth it to do a manual calculation (below).
        #
        # However, there is also a hidden option to force it to use specific values of stepK and
        # maxK (caveat user!).  The values of _force_stepk and _force_maxk should be provided in
        # terms of physical scale, e.g., for images that have a scale length of 0.1 arcsec, the
        # stepK and maxK should be provided in units of 1/arcsec.  Then we convert to the 1/pixel
        # units required by the C++ layer below.  Also note that profile recentering for even-sized
        # images (see the ._adjust_offset step below) leads to automatic reduction of stepK slightly
        # below what is provided here, while maxK is preserved.
        if _force_stepk > 0.:
            return _force_stepk
        elif calculate_stepk:
            if calculate_stepk is True:
                im = self._image
            else:
                # If not a bool, then value is max_stepk
                R = (jnp.ceil(jnp.pi / calculate_stepk)).astype(int)
                b = BoundsI(-R, R, -R, R)
                b = self._image.bounds & b
                im = self._image[b]
            thresh = (1.0 - self.gsparams.folding_threshold) * self._image_flux
            # this line appears buggy in galsim - I expect they meant to use im
            R = _galsim.CalculateSizeContainingFlux(im._image, thresh)
        else:
            R = jnp.max(self._image.array.shape) / 2. - 0.5
        return self._getSimpleStepK(R)

    def _getSimpleStepK(self, R):
        min_scale = self._wcs._minScale()
        # Add xInterp range in quadrature just like convolution:
        R2 = self._x_interpolant.xrange
        R = jnp.hypot(R, R2)
        stepk = jnp.pi / (R * min_scale)
        return stepk

    def _getMaxK(self, calculate_maxk, _force_maxk):
        max_scale = self._wcs._maxScale()
        if _force_maxk > 0.:
            return _force_maxk
        elif calculate_maxk:
            self._maxk = 0.
            self._sbp
            if calculate_maxk is True:
                self._sbii.calculateMaxK(0.)
            else:
                # If not a bool, then value is max_maxk
                self._sbii.calculateMaxK(float(calculate_maxk))
            self.__dict__.pop('_sbp')  # Need to remake it.
            return self._sbii.maxK() / max_scale
        else:
            return self._x_interpolant.krange / max_scale

    def __hash__(self):
        # Definitely want to cache this, since the size of the image could be large.
        if not hasattr(self, '_hash'):
            self._hash = hash(("galsim.InterpolatedImage", self.x_interpolant, self.k_interpolant))
            self._hash ^= hash((self.flux, self._stepk, self._maxk, self._pad_factor))
            self._hash ^= hash((self._xim.bounds, self._image.bounds, self._pad_image.bounds))
            # A common offset is 0.5,0.5, and *sometimes* this produces the same hash as 0,0
            # (which is also common).  I guess because they are only different in 2 bits.
            # This mucking of the numbers seems to help make the hash more reliably different for
            # these two cases.  Note: "sometiems" because of this:
            # https://stackoverflow.com/questions/27522626/hash-function-in-python-3-3-returns-different-results-between-sessions
            self._hash ^= hash((self._offset.x * 1.234, self._offset.y * 0.23424))
            self._hash ^= hash(self._gsparams)
            self._hash ^= hash(self._xim.wcs)
            # Just hash the diagonal.  Much faster, and usually is unique enough.
            # (Let python handle collisions as needed if multiple similar IIs are used as keys.)
            self._hash ^= hash(tuple(jnp.diag(self._pad_image.array)))
        return self._hash

    def __repr__(self):
        s = 'galsim.InterpolatedImage(%r, %r, %r' % (
            self._image, self.x_interpolant, self.k_interpolant
        )
        # Most things we keep even if not required, but the pad_image is large, so skip it
        # if it's really just the same as the main image.
        if self._pad_image.bounds != self._image.bounds:
            s += ', pad_image=%r' % (self._pad_image)
        s += ', pad_factor=%f, flux=%r, offset=%r' % (self._pad_factor, self.flux, self._offset)
        s += ', use_true_center=False, gsparams=%r, _force_stepk=%r, _force_maxk=%r)' % (
            self.gsparams, self._stepk, self._maxk
        )
        return s

    def __str__(self):
        return 'galsim.InterpolatedImage(image=%s, flux=%s)' % (self.image, self.flux)

    def __getstate__(self):
        d = self.__dict__.copy()
        # TODO - probably remove these pops for things we don't have
        d.pop('_sbii', None)
        d.pop('_sbp', None)
        # Only pickle _pad_image.  Not _xim or _image
        d['_xim_bounds'] = self._xim.bounds
        d['_image_bounds'] = self._image.bounds
        d.pop('_xim', None)
        d.pop('_image', None)
        return d

    def __setstate__(self, d):
        xim_bounds = d.pop('_xim_bounds')
        image_bounds = d.pop('_image_bounds')
        self.__dict__ = d
        if self._pad_image.bounds == xim_bounds:
            self._xim = self._pad_image
        else:
            self._xim = Image(xim_bounds, wcs=self._wcs, dtype=jnp.float64)
            self._xim[self._pad_image.bounds] = self._pad_image
        self._image = self._xim[image_bounds]

    @property
    def x_interpolant(self):
        """The real-space `Interpolant` for this profile.
        """
        return self._x_interpolant

    @property
    def k_interpolant(self):
        """The Fourier-space `Interpolant` for this profile.
        """
        return self._k_interpolant

    @property
    def image(self):
        """The underlying `Image` being interpolated.
        """
        return self._image

    @property
    def _centroid(self):
        return PositionD(self._sbp.centroid())

    @property
    def _positive_flux(self):
        return self._sbp.getPositiveFlux()

    @property
    def _negative_flux(self):
        return self._sbp.getNegativeFlux()

    # @lazy_property
    def _flux_per_photon(self):
        # FIXME: jax_galsim does not photon shoot
        # return self._calculate_flux_per_photon()
        raise NotImplementedError("Photon shooting not implemented.")

    @property
    def _max_sb(self):
        return self._sbp.maxSB()

    def _xValue(self, pos):
        return self._sbp.xValue(pos._p)

    def _kValue(self, kpos):
        return self._sbp.kValue(kpos._p)

    def _shoot(self, photons, rng):
        raise NotImplementedError("Photon shooting not implemented.")

    def _drawReal(self, image, jac=None, offset=(0., 0.), flux_scaling=1.):
        dx, dy = offset
        _jac = 0 if jac is None else jac.__array_interface__['data'][0]
        self._sbp.draw(image._image, image.scale, _jac, dx, dy, flux_scaling)

    def _drawKImage(self, image, jac=None):
        _jac = 0 if jac is None else jac.__array_interface__['data'][0]
        self._sbp.drawK(image._image, image.scale, _jac)
