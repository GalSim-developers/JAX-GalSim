import math
import textwrap
from functools import partial

import galsim as _galsim
import jax
import jax.numpy as jnp
from galsim.errors import (
    GalSimIncompatibleValuesError,
    GalSimRangeError,
    GalSimUndefinedBoundsError,
    GalSimValueError,
)
from galsim.utilities import doc_inherit
from jax._src.numpy.util import _wraps
from jax.tree_util import register_pytree_node_class

from jax_galsim import fits
from jax_galsim.bounds import BoundsI
from jax_galsim.core.draw import draw_by_kValue, draw_by_xValue
from jax_galsim.gsobject import GSObject
from jax_galsim.gsparams import GSParams
from jax_galsim.image import Image
from jax_galsim.interpolant import Quintic
from jax_galsim.position import PositionD
from jax_galsim.transform import Transformation
from jax_galsim.utilities import convert_interpolant
from jax_galsim.wcs import PixelScale


@_wraps(
    _galsim.InterpolatedImage,
    lax_description=textwrap.dedent(
        """The JAX equivalent of galsim.InterpolatedImage does not support

            - noise padding
            - depixelize

        Further, it always computes the FFT of the image as opposed to galsim
        where this is done as needed. One almost always needs the FFT and JAX
        generally works best with pure functions that do not modify state.
        """
    ),
)
@register_pytree_node_class
class InterpolatedImage(Transformation):
    _req_params = {"image": str}
    _opt_params = {
        "x_interpolant": str,
        "k_interpolant": str,
        "normalization": str,
        "scale": float,
        "flux": float,
        "pad_factor": float,
        "noise_pad_size": float,
        "noise_pad": str,
        "pad_image": str,
        "calculate_stepk": bool,
        "calculate_maxk": bool,
        "use_true_center": bool,
        "depixelize": bool,
        "offset": PositionD,
        "hdu": int,
    }
    _takes_rng = True

    def __init__(
        self,
        image,
        x_interpolant=None,
        k_interpolant=None,
        normalization="flux",
        scale=None,
        wcs=None,
        flux=None,
        pad_factor=4.0,
        noise_pad_size=0,
        noise_pad=0.0,
        rng=None,
        pad_image=None,
        calculate_stepk=True,
        calculate_maxk=True,
        use_cache=True,
        use_true_center=True,
        depixelize=False,
        offset=None,
        gsparams=None,
        _force_stepk=0.0,
        _force_maxk=0.0,
        hdu=None,
    ):
        obj = InterpolatedImageImpl(
            image,
            x_interpolant=x_interpolant,
            k_interpolant=k_interpolant,
            normalization=normalization,
            scale=scale,
            wcs=wcs,
            flux=flux,
            pad_factor=pad_factor,
            noise_pad_size=noise_pad_size,
            noise_pad=noise_pad,
            rng=rng,
            pad_image=pad_image,
            calculate_stepk=calculate_stepk,
            calculate_maxk=calculate_maxk,
            use_cache=use_cache,
            use_true_center=use_true_center,
            depixelize=depixelize,
            offset=offset,
            gsparams=gsparams,
            _force_stepk=_force_stepk,
            _force_maxk=_force_maxk,
            hdu=hdu,
        )
        super().__init__(
            obj,
            jac=obj._jac_arr,
            flux_ratio=obj._flux_ratio / obj._wcs.pixelArea(),
            offset=PositionD(0.0, 0.0),
        )

    @property
    def x_interpolant(self):
        """The real-space `Interpolant` for this profile."""
        return self._original._x_interpolant

    @property
    def k_interpolant(self):
        """The Fourier-space `Interpolant` for this profile."""
        return self._original._k_interpolant

    @property
    def image(self):
        """The underlying `Image` being interpolated."""
        return self._original._image

    def __hash__(self):
        return hash(self._original)

    def __repr__(self):
        return repr(self._original)

    def __str__(self):
        return str(self._original)


@register_pytree_node_class
class InterpolatedImageImpl(GSObject):
    _cache_noise_pad = {}

    _has_hard_edges = False
    _is_axisymmetric = False
    _is_analytic_x = True
    _is_analytic_k = True

    def __init__(
        self,
        image,
        x_interpolant=None,
        k_interpolant=None,
        normalization="flux",
        scale=None,
        wcs=None,
        flux=None,
        pad_factor=4.0,
        noise_pad_size=0,
        noise_pad=0.0,
        rng=None,
        pad_image=None,
        calculate_stepk=True,
        calculate_maxk=True,
        use_cache=True,
        use_true_center=True,
        depixelize=False,
        offset=None,
        gsparams=None,
        _force_stepk=0.0,
        _force_maxk=0.0,
        hdu=None,
    ):
        # this class does a ton of munging of the inputs that I don't want to reconstruct when
        # flattening and unflattening the class.
        # thus I am going to make some refs here so we have it when we need it
        self._jax_children = (
            image,
            dict(
                scale=scale,
                wcs=wcs,
                flux=flux,
                pad_image=pad_image,
                offset=offset,
            ),
        )
        self._jax_aux_data = dict(
            x_interpolant=x_interpolant,
            k_interpolant=k_interpolant,
            normalization=normalization,
            pad_factor=pad_factor,
            noise_pad_size=noise_pad_size,
            noise_pad=noise_pad,
            rng=rng,
            calculate_stepk=calculate_stepk,
            calculate_maxk=calculate_maxk,
            use_cache=use_cache,
            use_true_center=use_true_center,
            depixelize=depixelize,
            gsparams=gsparams,
            _force_stepk=_force_stepk,
            _force_maxk=_force_maxk,
            hdu=hdu,
        )
        self._params = {}

        from .wcs import BaseWCS, PixelScale

        # FIXME: no BaseDeviate in jax_galsim
        # from .random import BaseDeviate
        # If the "image" is not actually an image, try to read the image as a file.
        if isinstance(image, str):
            image = fits.read(image, hdu=hdu)
        elif not isinstance(image, Image):
            raise TypeError("Supplied image must be an Image or file name")

        # it must have well-defined bounds, otherwise seg fault in SBInterpolatedImage constructor
        if not image.bounds.isDefined():
            raise GalSimUndefinedBoundsError(
                "Supplied image does not have bounds defined."
            )

        # check what normalization was specified for the image: is it an image of surface
        # brightness, or flux?
        if normalization.lower() not in ("flux", "f", "surface brightness", "sb"):
            raise GalSimValueError(
                "Invalid normalization requested.",
                normalization,
                ("flux", "f", "surface brightness", "sb"),
            )

        # Set up the interpolants if none was provided by user, or check that the user-provided ones
        # are of a valid type
        self._gsparams = GSParams.check(gsparams)
        if x_interpolant is None:
            self._x_interpolant = Quintic(gsparams=self._gsparams)
        else:
            self._x_interpolant = convert_interpolant(x_interpolant).withGSParams(
                self._gsparams
            )
        if k_interpolant is None:
            self._k_interpolant = Quintic(gsparams=self._gsparams)
        else:
            self._k_interpolant = convert_interpolant(k_interpolant).withGSParams(
                self._gsparams
            )

        # Store the image as an attribute and make sure we don't change the original image
        # in anything we do here.  (e.g. set scale, etc.)
        if depixelize:
            # FIXME: no depixelize in jax_galsim
            # self._image = image.view(dtype=np.float64).depixelize(self._x_interpolant)
            raise NotImplementedError(
                "InterpolatedImages do not support 'depixelize' in jax_galsim."
            )
        else:
            self._image = image.view(dtype=jnp.float64, contiguous=True)
        self._image.setCenter(0, 0)

        # Set the wcs if necessary
        if scale is not None:
            if wcs is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot provide both scale and wcs to InterpolatedImage",
                    scale=scale,
                    wcs=wcs,
                )
            self._image.wcs = PixelScale(scale)
        elif wcs is not None:
            if not isinstance(wcs, BaseWCS):
                raise TypeError("wcs parameter is not a galsim.BaseWCS instance")
            self._image.wcs = wcs
        elif self._image.wcs is None:
            raise GalSimIncompatibleValuesError(
                "No information given with Image or keywords about pixel scale!",
                scale=scale,
                wcs=wcs,
                image=image,
            )

        # Figure out the offset to apply based on the original image (not the padded one).
        # We will apply this below in _sbp.
        offset = self._parse_offset(offset)
        self._offset = self._adjust_offset(
            self._image.bounds, offset, None, use_true_center
        )

        im_cen = image.true_center if use_true_center else image.center
        self._jac_arr = self._image.wcs.jacobian(image_pos=im_cen).getMatrix().ravel()
        self._wcs = self._image.wcs.local(image_pos=im_cen)

        # Build the fully padded real-space image according to the various pad options.
        self._buildImages(
            pad_factor,
            pad_image,
            noise_pad_size,
            noise_pad,
            rng,
            use_cache,
            flux,
            normalization,
        )

        # I think the only things that will mess up if flux == 0 are the
        # calculateStepK and calculateMaxK functions, and rescaling the flux to some value.
        if (
            calculate_stepk or calculate_maxk or flux is not None
        ) and self._image_flux == 0.0:
            raise GalSimValueError(
                "This input image has zero total flux. It does not define a "
                "valid surface brightness profile.",
                image,
            )

        # Process the different options for flux, stepk, maxk
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
            ret._stepk = ret._getStepK(self._calculate_stepk, 0.0)
        if ret._gsparams.maxk_threshold != self._gsparams.maxk_threshold:
            ret._maxk = ret._getMaxK(self._calculate_maxk, 0.0)
        return ret

    def tree_flatten(self):
        """This function flattens the InterpolatedImage into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        return (self._jax_children, self._jax_aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        val = {}
        val.update(aux_data)
        val.update(children[1])
        return cls(children[0], **val)

    def _buildImages(
        self,
        pad_factor,
        pad_image,
        noise_pad_size,
        noise_pad,
        rng,
        use_cache,
        flux,
        normalization,
    ):
        # If the user specified a surface brightness normalization for the input Image, then
        # need to rescale flux by the pixel area to get proper normalization.
        self._image_flux = jnp.sum(self._image.array, dtype=float)
        if flux is None:
            flux = self._image_flux
            if normalization.lower() in ("surface brightness", "sb"):
                flux *= self._wcs.pixelArea()
        self._flux = flux

        # If the user specified a flux, then set the flux ratio for the transform that wraps
        # this class
        if self._flux != self._image_flux:
            self._flux_ratio = self._flux / self._image_flux
        else:
            self._flux_ratio = 1.0

        # Check that given pad_image is valid:
        if pad_image is not None:
            if isinstance(pad_image, str):
                pad_image = fits.read(pad_image).view(dtype=jnp.float64)
            elif isinstance(pad_image, Image):
                pad_image = pad_image.view(dtype=jnp.float64, contiguous=True)
            else:
                raise TypeError("Supplied pad_image must be an Image.", pad_image)

        if pad_factor <= 0.0:
            raise GalSimRangeError(
                "Invalid pad_factor <= 0 in InterpolatedImage", pad_factor, 0.0
            )

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
            raise NotImplementedError(
                "InterpolatedImages do not support noise padding in jax_galsim."
            )
        else:
            if noise_pad:
                # FIXME: no BaseDeviate in jax_galsim so no noise padding
                # raise GalSimIncompatibleValuesError(
                #         "Must provide noise_pad_size if noise_pad != 0",
                #         noise_pad=noise_pad, noise_pad_size=noise_pad_size)
                raise NotImplementedError(
                    "InterpolatedImages do not support noise padding in jax_galsim."
                )

        # The size of the final padded image is the largest of the various size specifications
        pad_size = max(self._image.array.shape)
        if pad_factor > 1.0:
            pad_size = int(math.ceil(pad_factor * pad_size))
        if noise_pad_size:
            pad_size = max(pad_size, noise_pad_size)
        if pad_image:
            pad_image.setCenter(0, 0)
            pad_size = max(pad_size, *pad_image.array.shape)
        # And round up to a good fft size
        pad_size = Image.good_fft_size(pad_size)

        self._xim = Image(pad_size, pad_size, dtype=jnp.float64, wcs=PixelScale(1.0))
        self._xim.setCenter(0, 0)
        self._image.wcs = PixelScale(1.0)

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

        # And update the _image to be that portion of the full real image rather than the
        # input image.
        self._image = self._xim[self._image.bounds]

        # These next two allow for easy pickling/repring.  We don't need to serialize all the
        # zeros around the edge.  But we do need to keep any non-zero padding as a pad_image.
        self._pad_image = self._xim[nz_bounds]
        # self._pad_factor = (max(self._xim.array.shape)-1.e-6) / max(self._image.array.shape)
        self._pad_factor = pad_factor

        # we always make this
        self._kim = self._xim.calculate_fft()

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
        if _force_stepk > 0.0:
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
            R = _calculate_size_containing_flux(im, thresh)
        else:
            R = max(*self._image.array.shape) / 2.0 - 0.5
        return self._getSimpleStepK(R)

    def _getSimpleStepK(self, R):
        min_scale = 1.0
        # Add xInterp range in quadrature just like convolution:
        R2 = self._x_interpolant.xrange
        R = jnp.hypot(R, R2)
        stepk = jnp.pi / (R * min_scale)
        return stepk

    def _getMaxK(self, calculate_maxk, _force_maxk):
        max_scale = 1.0
        if _force_maxk > 0.0:
            return _force_maxk
        elif calculate_maxk:
            _uscale = 1 / (2 * jnp.pi)
            self._maxk = self._x_interpolant.urange() / _uscale / max_scale

            if calculate_maxk is True:
                maxk = _find_maxk(
                    self._kim, self._maxk, self._gsparams.maxk_threshold * self.flux
                )
            else:
                maxk = _find_maxk(
                    self._kim, calculate_maxk, self._gsparams.maxk_threshold * self.flux
                )

            return maxk / max_scale
        else:
            return self._x_interpolant.krange / max_scale

    def __hash__(self):
        # Definitely want to cache this, since the size of the image could be large.
        if not hasattr(self, "_hash"):
            self._hash = hash(
                ("galsim.InterpolatedImage", self.x_interpolant, self.k_interpolant)
            )
            self._hash ^= hash(
                (
                    self.flux.item(),
                    self._stepk.item(),
                    self._maxk.item(),
                    self._pad_factor,
                )
            )
            self._hash ^= hash(
                (self._xim.bounds, self._image.bounds, self._pad_image.bounds)
            )
            # A common offset is 0.5,0.5, and *sometimes* this produces the same hash as 0,0
            # (which is also common).  I guess because they are only different in 2 bits.
            # This mucking of the numbers seems to help make the hash more reliably different for
            # these two cases.  Note: "sometiems" because of this:
            # https://stackoverflow.com/questions/27522626/hash-function-in-python-3-3-returns-different-results-between-sessions
            self._hash ^= hash((self._offset.x * 1.234, self._offset.y * 0.23424))
            self._hash ^= hash(self._gsparams)
            self._hash ^= hash(self._wcs)
            # Just hash the diagonal.  Much faster, and usually is unique enough.
            # (Let python handle collisions as needed if multiple similar IIs are used as keys.)
            self._hash ^= hash(tuple(jnp.diag(self._pad_image.array).tolist()))
        return self._hash

    def __repr__(self):
        s = "galsim.InterpolatedImage(%r, %r, %r" % (
            self._image,
            self.x_interpolant,
            self.k_interpolant,
        )
        # Most things we keep even if not required, but the pad_image is large, so skip it
        # if it's really just the same as the main image.
        if self._pad_image.bounds != self._image.bounds:
            s += ", pad_image=%r" % (self._pad_image)
        s += ", wcs=%s" % self._wcs
        s += ", pad_factor=%f, flux=%r, offset=%r" % (
            self._pad_factor,
            self.flux,
            self._offset,
        )
        s += (
            ", use_true_center=False, gsparams=%r, _force_stepk=%r, _force_maxk=%r)"
            % (self.gsparams, self._stepk, self._maxk)
        )
        return s

    def __str__(self):
        return "galsim.InterpolatedImage(image=%s, flux=%s)" % (self.image, self.flux)

    def __getstate__(self):
        d = self.__dict__.copy()
        # Only pickle _pad_image.  Not _xim or _image
        d["_xim_bounds"] = self._xim.bounds
        d["_image_bounds"] = self._image.bounds
        d.pop("_xim", None)
        d.pop("_image", None)
        return d

    def __setstate__(self, d):
        xim_bounds = d.pop("_xim_bounds")
        image_bounds = d.pop("_image_bounds")
        self.__dict__ = d
        if self._pad_image.bounds == xim_bounds:
            self._xim = self._pad_image
        else:
            self._xim = Image(xim_bounds, wcs=PixelScale(1.0), dtype=jnp.float64)
            self._xim[self._pad_image.bounds] = self._pad_image
        self._image = self._xim[image_bounds]

    @property
    def x_interpolant(self):
        """The real-space `Interpolant` for this profile."""
        return self._x_interpolant

    @property
    def k_interpolant(self):
        """The Fourier-space `Interpolant` for this profile."""
        return self._k_interpolant

    @property
    def image(self):
        """The underlying `Image` being interpolated."""
        return self._image

    @property
    def _flux(self):
        """By default, the flux is contained in the parameters dictionay."""
        return self._params["flux"]

    @_flux.setter
    def _flux(self, value):
        self._params["flux"] = value

    @property
    def _centroid(self):
        raise NotImplementedError("WIP interp - centroid")

    @property
    def _positive_flux(self):
        raise NotImplementedError("WIP interp - positive_flux")

    @property
    def _negative_flux(self):
        raise NotImplementedError("WIP interp - negative_flux")

    @property
    def _max_sb(self):
        return jnp.max(jnp.abs(self._pad_image.array))

    # @lazy_property
    def _flux_per_photon(self):
        # FIXME: jax_galsim does not photon shoot
        # return self._calculate_flux_per_photon()
        raise NotImplementedError("Photon shooting not implemented.")

    def _xValue(self, pos):
        pos += self._offset
        vals = _draw_with_interpolant_xval(
            jnp.array([pos.x], dtype=float),
            jnp.array([pos.y], dtype=float),
            self._pad_image.bounds.xmin,
            self._pad_image.bounds.ymin,
            self._pad_image.array,
            self._x_interpolant,
        )
        return vals[0]

    def _kValue(self, kpos):
        # phase factor due to offset
        # not we shift by -offset which explains the signs
        # in pkx, pky
        pkx = kpos.x * 1j * self._offset.x
        pky = kpos.y * 1j * self._offset.y
        pkx += pky
        pfac = jnp.exp(pkx)

        kx = jnp.array([kpos.x / self._kim.scale], dtype=float)
        ky = jnp.array([kpos.y / self._kim.scale], dtype=float)

        _uscale = 1.0 / (2.0 * jnp.pi)
        _maxk_xint = self._x_interpolant.urange() / _uscale / self._kim.scale

        val = _draw_with_interpolant_kval(
            kx,
            ky,
            self._kim.bounds.ymin,
            self._kim.bounds.ymin,
            self._kim.array,
            self._k_interpolant,
        )

        msk = (jnp.abs(kx) <= _maxk_xint) & (jnp.abs(ky) <= _maxk_xint)
        xint_val = self._x_interpolant._kval_noraise(
            kx * self._kim.scale
        ) * self._x_interpolant._kval_noraise(ky * self._kim.scale)
        return jnp.where(msk, val * xint_val * pfac, 0.0)[0]

    def _shoot(self, photons, rng):
        raise NotImplementedError("Photon shooting not implemented.")

    def _drawReal(self, image, jac=None, offset=(0.0, 0.0), flux_scaling=1.0):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_xValue(self, image, _jac, jnp.asarray(offset), flux_scaling)

    def _drawKImage(self, image, jac=None):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_kValue(self, image, _jac)


@partial(jax.jit, static_argnums=(5,))
def _draw_with_interpolant_xval(x, y, xmin, ymin, zp, interp):
    orig_shape = x.shape
    x = x.ravel()
    xi = jnp.floor(x - xmin).astype(jnp.int32)
    xp = xi + xmin
    nx = zp.shape[1]

    y = y.ravel()
    yi = jnp.floor(y - ymin).astype(jnp.int32)
    yp = yi + ymin
    ny = zp.shape[0]

    def _body_1d(i, args):
        z, wy, msky, yind, xi, xp, zp = args

        xind = xi + i
        mskx = (xind >= 0) & (xind < nx)
        _x = x - (xp + i)
        wx = interp._xval_noraise(_x)

        w = wx * wy
        msk = msky & mskx
        z += jnp.where(msk, zp[yind, xind] * w, 0)

        return [z, wy, msky, yind, xi, xp, zp]

    def _body(i, args):
        z, xi, yi, xp, yp, zp = args
        yind = yi + i
        msk = (yind >= 0) & (yind < ny)
        _y = y - (yp + i)
        wy = interp._xval_noraise(_y)
        z = jax.lax.fori_loop(
            -interp.xrange, interp.xrange + 1, _body_1d, [z, wy, msk, yind, xi, xp, zp]
        )[0]
        return [z, xi, yi, xp, yp, zp]

    z = jax.lax.fori_loop(
        -interp.xrange,
        interp.xrange + 1,
        _body,
        [jnp.zeros(x.shape, dtype=zp.dtype), xi, yi, xp, yp, zp],
    )[0]
    return z.reshape(orig_shape)


@partial(jax.jit, static_argnums=(5,))
def _draw_with_interpolant_kval(kx, ky, kxmin, kymin, zp, interp):
    orig_shape = kx.shape
    kx = kx.ravel()
    kxi = jnp.floor(kx - kxmin).astype(jnp.int32)
    kxp = kxi + kxmin
    nkx_2 = zp.shape[1] - 1
    nkx = nkx_2 * 2 + 1

    ky = ky.ravel()
    kyi = jnp.floor(ky - kymin).astype(jnp.int32)
    kyp = kyi + kymin
    nky = zp.shape[0]

    def _body_1d(i, args):
        z, wky, kyind, kxi, nkx, nkx_2, kxp, zp = args

        kxind = (kxi + i) % nkx
        _kx = kx - (kxp + i)
        wkx = interp._xval_noraise(_kx)

        val = jnp.where(
            kxind < nkx_2,
            zp[nky - 1 - kyind, nkx - 1 - kxind + nkx_2].conjugate(),
            zp[kyind, kxind - nkx_2],
        )
        z += val * wkx * wky

        return [z, wky, kyind, kxi, nkx, nkx_2, kxp, zp]

    def _body(i, args):
        z, kxi, kyi, nky, nkx, nkx_2, kxp, kyp, zp = args
        kyind = (kyi + i) % nky
        _ky = ky - (kyp + i)
        wky = interp._xval_noraise(_ky)
        z = jax.lax.fori_loop(
            -interp.xrange,
            interp.xrange + 1,
            _body_1d,
            [z, wky, kyind, kxi, nkx, nkx_2, kxp, zp],
        )[0]
        return [z, kxi, kyi, nky, nkx, nkx_2, kxp, kyp, zp]

    z = jax.lax.fori_loop(
        -interp.xrange,
        interp.xrange + 1,
        _body,
        [jnp.zeros(kx.shape, dtype=zp.dtype), kxi, kyi, nky, nkx, nkx_2, kxp, kyp, zp],
    )[0]
    return z.reshape(orig_shape)


@jax.jit
def _flux_frac(a, x, y, cenx, ceny):
    def _body(d, args):
        res, a, dx, dy, cenx, ceny = args
        msk = (jnp.abs(dx) <= d) & (jnp.abs(dx) <= d)

        res = res.at[d].set(
            jnp.sum(
                jnp.where(
                    msk,
                    a,
                    0.0,
                )
            )
        )

        return [res, a, dx, dy, cenx, ceny]

    res = jnp.zeros(a.shape[0], dtype=float) - jnp.inf
    return jax.lax.fori_loop(
        0, a.shape[0], _body, [res, a, x - cenx, y - ceny, cenx, ceny]
    )[0]


def _calculate_size_containing_flux(image, thresh):
    cenx, ceny = image.center.x, image.center.y
    x, y = image.get_pixel_centers()
    fluxes = _flux_frac(image.array, x, y, cenx, ceny)
    msk = fluxes >= -jnp.inf
    fluxes = jnp.where(msk, fluxes, jnp.max(fluxes))
    d = jnp.arange(image.array.shape[0]) + 1.0
    expfac = 4.0
    dint = jnp.arange(image.array.shape[0] * expfac) / expfac + 1.0
    fluxes = jnp.interp(dint, d, fluxes)
    msk = fluxes <= thresh
    return (
        jnp.argmax(
            jnp.where(
                msk,
                dint,
                -jnp.inf,
            )
        )
        / expfac
        + 1.0
    )


@jax.jit
def _inner_comp_find_maxk(arr, thresh, kx, ky):
    msk = arr * arr.conjugate() > thresh * thresh
    max_kx = jnp.max(
        jnp.where(
            msk,
            jnp.abs(kx),
            -jnp.inf,
        )
    )
    max_ky = jnp.max(
        jnp.where(
            msk,
            jnp.abs(ky),
            -jnp.inf,
        )
    )
    return jnp.maximum(max_kx, max_ky)


def _find_maxk(kim, max_maxk, thresh):
    kx, ky = kim.get_pixel_centers()
    kx *= kim.scale
    ky *= kim.scale
    return _inner_comp_find_maxk(kim.array, thresh, kx, ky) * 1.15
