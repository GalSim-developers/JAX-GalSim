import copy
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
from jax_galsim.core.utils import compute_major_minor_from_jacobian, ensure_hashable
from jax_galsim.gsobject import GSObject
from jax_galsim.gsparams import GSParams
from jax_galsim.image import Image
from jax_galsim.interpolant import Quintic
from jax_galsim.position import PositionD
from jax_galsim.transform import Transformation
from jax_galsim.utilities import convert_interpolant
from jax_galsim.wcs import BaseWCS, PixelScale

# These keys are removed from the public API of
# InterpolatedImage so that it matches the galsim
# one.
# The DirMeta class does this along with the changes to
# __getattribute__ and __dir__ below.
_KEYS_TO_REMOVE = [
    "flux_ratio",
    "jac",
    "offset",
    "original",
]


# magic from https://stackoverflow.com/questions/46120462/how-to-override-the-dir-method-for-a-class
class DirMeta(type):
    def __dir__(cls):
        keys = set(list(cls.__dict__.keys()) + dir(cls.__base__))
        keys -= set(_KEYS_TO_REMOVE)
        return list(keys)


@_wraps(
    _galsim.InterpolatedImage,
    lax_description=textwrap.dedent(
        """The JAX equivalent of galsim.InterpolatedImage does not support

            - noise padding
            - depixelize
            - most of the type checks and dtype casts done by galsim

        Further, it always computes the FFT of the image as opposed to galsim
        where this is done as needed. One almost always needs the FFT and JAX
        generally works best with pure functions that do not modify state.
        """
    ),
)
@register_pytree_node_class
class InterpolatedImage(Transformation, metaclass=DirMeta):
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
        _recenter_image=True,  # this option is used by _InterpolatedImage below
        hdu=None,
        _obj=None,
    ):
        # If the "image" is not actually an image, try to read the image as a file.
        if isinstance(image, str):
            image = fits.read(image, hdu=hdu)
        elif not isinstance(image, Image):
            raise TypeError("Supplied image must be an Image or file name")

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
            gsparams=GSParams.check(gsparams),
            _force_stepk=_force_stepk,
            _force_maxk=_force_maxk,
            _recenter_image=_recenter_image,
            hdu=hdu,
        )

        if _obj is not None:
            obj = _obj
        else:
            obj = _InterpolatedImageImpl(
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
                gsparams=GSParams.check(gsparams),
                _force_stepk=_force_stepk,
                _force_maxk=_force_maxk,
                hdu=hdu,
                _recenter_image=_recenter_image,
            )

        # we don't use the parent init but instead set things by hand to
        # avoid computations upon init
        self._gsparams = GSParams.check(gsparams, obj.gsparams)
        self._propagate_gsparams = True
        if self._propagate_gsparams:
            obj = obj.withGSParams(self._gsparams)
        self._original = obj
        self._params = {
            "offset": PositionD(0.0, 0.0),
        }
        self._jax_children[1]["_obj"] = obj

    @property
    def _flux_ratio(self):
        return self._original._flux_ratio / self._original._wcs.pixelArea()

    @property
    def _jac(self):
        return self._original._jac_arr.reshape((2, 2))

    def __getattribute__(self, name):
        if name in _KEYS_TO_REMOVE:
            raise AttributeError(f"{self.__class__} has no attribute '{name}'")
        return super().__getattribute__(name)

    def __dir__(self):
        allattrs = set(self.__dict__.keys() + dir(self.__class__))
        allattrs -= set(_KEYS_TO_REMOVE)
        return list(allattrs)

    # the galsim tests use this internal attribute
    # so we add it here
    @property
    def _xim(self):
        return self._original._xim

    @property
    def _maxk(self):
        if self._jax_aux_data["_force_maxk"] > 0:
            return self._jax_aux_data["_force_maxk"]
        else:
            return super()._maxk

    @property
    def _stepk(self):
        if self._jax_aux_data["_force_stepk"] > 0:
            return self._jax_aux_data["_force_stepk"]
        else:
            return super()._stepk

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
        # Definitely want to cache this, since the size of the image could be large.
        if not hasattr(self, "_hash"):
            self._hash = hash(
                ("galsim.InterpolatedImage", self.x_interpolant, self.k_interpolant)
            )
            self._hash ^= hash(
                (
                    ensure_hashable(self.flux),
                    ensure_hashable(self._stepk),
                    ensure_hashable(self._maxk),
                    ensure_hashable(self._original._jax_aux_data["pad_factor"]),
                )
            )
            self._hash ^= hash(
                (
                    self._original._xim.bounds,
                    self._original._image.bounds,
                    self._original._pad_image.bounds,
                )
            )
            # A common offset is 0.5,0.5, and *sometimes* this produces the same hash as 0,0
            # (which is also common).  I guess because they are only different in 2 bits.
            # This mucking of the numbers seems to help make the hash more reliably different for
            # these two cases.  Note: "sometiems" because of this:
            # https://stackoverflow.com/questions/27522626/hash-function-in-python-3-3-returns-different-results-between-sessions
            self._hash ^= hash(
                (
                    ensure_hashable(self._original._offset.x * 1.234),
                    ensure_hashable(self._original._offset.y * 0.23424),
                )
            )
            self._hash ^= hash(self.gsparams)
            self._hash ^= hash(self._original._wcs)
            # Just hash the diagonal.  Much faster, and usually is unique enough.
            # (Let python handle collisions as needed if multiple similar IIs are used as keys.)
            self._hash ^= hash(ensure_hashable(self._original._pad_image.array))
        return self._hash

    def __repr__(self):
        s = "galsim.InterpolatedImage(%r, %r, %r, wcs=%r" % (
            self._original.image,
            self.x_interpolant,
            self.k_interpolant,
            self._original._wcs,
        )
        # Most things we keep even if not required, but the pad_image is large, so skip it
        # if it's really just the same as the main image.
        if self._original._pad_image.bounds != self._original.image.bounds:
            s += ", pad_image=%r" % (self._pad_image)
        s += ", pad_factor=%f, flux=%r, offset=%r" % (
            ensure_hashable(self._original._jax_aux_data["pad_factor"]),
            ensure_hashable(self.flux),
            self._original._offset,
        )
        s += (
            ", use_true_center=False, gsparams=%r, _force_stepk=%r, _force_maxk=%r)"
            % (
                self.gsparams,
                ensure_hashable(self._stepk),
                ensure_hashable(self._maxk),
            )
        )
        return s

    def __str__(self):
        return "galsim.InterpolatedImage(image=%s, flux=%s)" % (self.image, self.flux)

    def __eq__(self, other):
        return self is other or (
            isinstance(other, InterpolatedImage)
            and self._xim == other._xim
            and self.x_interpolant == other.x_interpolant
            and self.k_interpolant == other.k_interpolant
            and self.flux == other.flux
            and self._original._offset == other._original._offset
            and self.gsparams == other.gsparams
            and self._stepk == other._stepk
            and self._maxk == other._maxk
        )

    def tree_flatten(self):
        """This function flattens the InterpolatedImage into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        return (self._jax_children, copy.copy(self._jax_aux_data))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        val = {}
        val.update(aux_data)
        val.update(children[1])
        return cls(children[0], **val)

    @_wraps(_galsim.InterpolatedImage.withGSParams)
    def withGSParams(self, gsparams=None, **kwargs):
        if gsparams == self.gsparams:
            return self
        # Checking gsparams
        gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        # Flattening the representation to instantiate a clean new object
        children, aux_data = self.tree_flatten()
        aux_data["gsparams"] = gsparams
        ret = self.tree_unflatten(aux_data, children)

        return ret


@partial(jax.jit, static_argnums=(1,))
def _zeropad_image(arr, npad):
    return jnp.pad(arr, npad, mode="constant", constant_values=0.0)


@register_pytree_node_class
class _InterpolatedImageImpl(GSObject):
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
        _recenter_image=True,
    ):
        # this class does a ton of munging of the inputs that I don't want to reconstruct when
        # flattening and unflattening the class.
        # thus I am going to make some refs here so we have it when we need it
        self._cached_comps = {}
        self._jax_children = (
            image,
            dict(
                scale=scale,
                wcs=wcs,
                flux=flux,
                pad_image=pad_image,
                offset=offset,
            ),
            self._cached_comps,
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
            _recenter_image=_recenter_image,
            hdu=hdu,
        )

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

        if pad_image is not None:
            raise NotImplementedError("pad_image not implemented in jax_galsim.")

        if pad_factor <= 0.0:
            raise GalSimRangeError(
                "Invalid pad_factor <= 0 in InterpolatedImage", pad_factor, 0.0
            )

        if noise_pad_size:
            raise NotImplementedError(
                "InterpolatedImages do not support noise padding in jax_galsim."
            )
        else:
            if noise_pad:
                raise NotImplementedError(
                    "InterpolatedImages do not support noise padding in jax_galsim."
                )

        if scale is not None:
            if wcs is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot provide both scale and wcs to InterpolatedImage",
                    scale=self._jax_children[1]["scale"],
                    wcs=self._jax_children[1]["wcs"],
                )
        elif wcs is not None:
            if not isinstance(wcs, BaseWCS):
                raise TypeError("wcs parameter is not a galsim.BaseWCS instance")
        else:
            if self._jax_children[0].wcs is None:
                raise GalSimIncompatibleValuesError(
                    "No information given with Image or keywords about pixel scale!",
                    scale=self._jax_children[1]["scale"],
                    wcs=self._jax_children[1]["wcs"],
                    image=self._jax_children[0],
                )

    @property
    def _flux_ratio(self):
        if self._jax_children[1]["flux"] is None:
            flux = self._image_flux
            if self._jax_aux_data["normalization"].lower() in (
                "surface brightness",
                "sb",
            ):
                flux *= self._wcs.pixelArea()
        else:
            flux = self._jax_children[1]["flux"]

        # If the user specified a flux, then set the flux ratio for the transform that wraps
        # this class
        return flux / self._image_flux

    @property
    def _image_flux(self):
        return jnp.sum(self._image.array, dtype=float)

    @property
    def _offset(self):
        # Figure out the offset to apply based on the original image (not the padded one).
        # We will apply this below in _sbp.
        offset = self._parse_offset(self._jax_children[1]["offset"])
        return self._adjust_offset(
            self._image.bounds, offset, None, self._jax_aux_data["use_true_center"]
        )

    @property
    def _image(self):
        # Store the image as an attribute and make sure we don't change the original image
        # in anything we do here.  (e.g. set scale, etc.)
        if self._jax_aux_data["depixelize"]:
            # FIXME: no depixelize in jax_galsim
            # self._image = image.view(dtype=np.float64).depixelize(self._x_interpolant)
            raise NotImplementedError(
                "InterpolatedImages do not support 'depixelize' in jax_galsim."
            )
        else:
            image = self._jax_children[0].view(dtype=float)

        if self._jax_aux_data["_recenter_image"]:
            image.setCenter(0, 0)

        return image

    @property
    def _wcs(self):
        im_cen = (
            self._jax_children[0].true_center
            if self._jax_aux_data["use_true_center"]
            else self._jax_children[0].center
        )

        # error checking was done on init
        if self._jax_children[1]["scale"] is not None:
            wcs = PixelScale(self._jax_children[1]["scale"])
        elif self._jax_children[1]["wcs"] is not None:
            wcs = self._jax_children[1]["wcs"]
        else:
            wcs = self._jax_children[0].wcs

        return wcs.local(image_pos=im_cen)

    @property
    def _jac_arr(self):
        image = self._jax_children[0]
        im_cen = (
            image.true_center if self._jax_aux_data["use_true_center"] else image.center
        )
        return self._wcs.jacobian(image_pos=im_cen).getMatrix().ravel()

    @property
    def _maxk(self):
        if self._jax_aux_data["_force_maxk"]:
            major, minor = compute_major_minor_from_jacobian(
                self._jac_arr.reshape((2, 2))
            )
            return self._jax_aux_data["_force_maxk"] * minor
        else:
            if "_maxk" not in self._cached_comps:
                self._cached_comps["_maxk"] = self._getMaxK(
                    self._jax_aux_data["calculate_maxk"]
                )
            return self._cached_comps["_maxk"]

    @property
    def _stepk(self):
        if self._jax_aux_data["_force_stepk"]:
            major, minor = compute_major_minor_from_jacobian(
                self._jac_arr.reshape((2, 2))
            )
            return self._jax_aux_data["_force_stepk"] * minor
        else:
            if "_stepk" not in self._cached_comps:
                self._cached_comps["_stepk"] = self._getStepK(
                    self._jax_aux_data["calculate_stepk"]
                )
            return self._cached_comps["_stepk"]

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

        return ret

    def tree_flatten(self):
        """This function flattens the InterpolatedImage into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        return (self._jax_children, copy.copy(self._jax_aux_data))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        val = {}
        val.update(aux_data)
        val.update(children[1])
        ret = cls(children[0], **val)
        ret._cached_comps.update(children[2])
        return ret

    @property
    def _xim(self):
        if "_xim" not in self._cached_comps:
            pad_factor = self._jax_aux_data["pad_factor"]

            # The size of the final padded image is the largest of the various size specifications
            pad_size = max(self._image.array.shape)
            if pad_factor > 1.0:
                pad_size = int(math.ceil(pad_factor * pad_size))
            # And round up to a good fft size
            pad_size = Image.good_fft_size(pad_size)

            xim = Image(
                _zeropad_image(
                    self._image.array, (pad_size - max(self._image.array.shape)) // 2
                ),
                wcs=PixelScale(1.0),
            )
            xim.setCenter(0, 0)
            xim.wcs = PixelScale(1.0)

            # Now place the given image in the center of the padding image:
            # assert self._xim.bounds.includes(self._image.bounds)
            xim[self._image.bounds] = self._image

            self._cached_comps["_xim"] = xim

        return self._cached_comps["_xim"]

    @property
    def _pad_image(self):
        # These next two allow for easy pickling/repring.  We don't need to serialize all the
        # zeros around the edge.  But we do need to keep any non-zero padding as a pad_image.
        xim = self._xim
        nz_bounds = self._image.bounds
        return xim[nz_bounds]

    @property
    def _kim(self):
        if "_kim" in self._cached_comps:
            return self._cached_comps["_kim"]
        else:
            kim = self._xim.calculate_fft()
            self._cached_comps["_kim"] = kim
            return kim

    @property
    def _pos_neg_fluxes(self):
        # record pos and neg fluxes now too
        pflux = jnp.sum(jnp.where(self._pad_image.array > 0, self._pad_image.array, 0))
        nflux = jnp.abs(
            jnp.sum(jnp.where(self._pad_image.array < 0, self._pad_image.array, 0))
        )
        pint = self._x_interpolant.positive_flux
        nint = self._x_interpolant.negative_flux
        pint2d = pint * pint + nint * nint
        nint2d = 2 * pint * nint
        return [
            pint2d * pflux + nint2d * nflux,
            pint2d * nflux + nint2d * pflux,
        ]

    def _getStepK(self, calculate_stepk):
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
        if calculate_stepk:
            if calculate_stepk is True:
                im = self.image
            else:
                # If not a bool, then value is max_stepk
                R = (jnp.ceil(jnp.pi / calculate_stepk)).astype(int)
                b = BoundsI(-R, R, -R, R)
                b = self.image.bounds & b
                im = self.image[b]
            thresh = (1.0 - self.gsparams.folding_threshold) * self._image_flux
            # this line appears buggy in galsim - I expect they meant to use im
            R = _calculate_size_containing_flux(im, thresh)
        else:
            R = max(*self.image.array.shape) / 2.0 - 0.5
        return self._getSimpleStepK(R)

    def _getSimpleStepK(self, R):
        # Add xInterp range in quadrature just like convolution:
        R2 = self._x_interpolant.xrange
        R = jnp.hypot(R, R2)
        stepk = jnp.pi / R
        return stepk

    def _getMaxK(self, calculate_maxk):
        if calculate_maxk:
            _uscale = 1 / (2 * jnp.pi)
            _maxk = self._x_interpolant.urange() / _uscale

            if calculate_maxk is True:
                maxk = _find_maxk(
                    self._kim, _maxk, self._gsparams.maxk_threshold * self.flux
                )
            else:
                maxk = _find_maxk(
                    self._kim, calculate_maxk, self._gsparams.maxk_threshold * self.flux
                )

            return maxk
        else:
            return self._x_interpolant.krange

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
        return self._xim[self._image.bounds]

    @property
    def _flux(self):
        return self._image_flux

    @property
    def _centroid(self):
        x, y = self._pad_image.get_pixel_centers()
        tot = jnp.sum(self._pad_image.array)
        xpos = jnp.sum(x * self._pad_image.array) / tot
        ypos = jnp.sum(y * self._pad_image.array) / tot
        return PositionD(xpos, ypos)

    @property
    def _positive_flux(self):
        return self._pos_neg_fluxes[0]

    @property
    def _negative_flux(self):
        return self._pos_neg_fluxes[1]

    @property
    def _max_sb(self):
        return jnp.max(jnp.abs(self._pad_image.array))

    def _flux_per_photon(self):
        return self._calculate_flux_per_photon()

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

        _kim = self._kim
        kx = jnp.array([kpos.x / _kim.scale], dtype=float)
        ky = jnp.array([kpos.y / _kim.scale], dtype=float)

        _uscale = 1.0 / (2.0 * jnp.pi)
        _maxk_xint = self._x_interpolant.urange() / _uscale / _kim.scale

        val = _draw_with_interpolant_kval(
            kx,
            ky,
            _kim.bounds.ymin,
            _kim.bounds.ymin,
            _kim.array,
            self._k_interpolant,
        )

        msk = (jnp.abs(kx) <= _maxk_xint) & (jnp.abs(ky) <= _maxk_xint)
        xint_val = self._x_interpolant._kval_noraise(
            kx * _kim.scale
        ) * self._x_interpolant._kval_noraise(ky * _kim.scale)
        return jnp.where(msk, val * xint_val * pfac, 0.0)[0]

    def _shoot(self, photons, rng):
        raise NotImplementedError("Photon shooting not implemented.")

    def _drawReal(self, image, jac=None, offset=(0.0, 0.0), flux_scaling=1.0):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_xValue(self, image, _jac, jnp.asarray(offset), flux_scaling)

    def _drawKImage(self, image, jac=None):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_kValue(self, image, _jac)


@_wraps(_galsim._InterpolatedImage)
def _InterpolatedImage(
    image,
    x_interpolant=Quintic(),
    k_interpolant=Quintic(),
    use_true_center=True,
    offset=None,
    gsparams=None,
    force_stepk=0.0,
    force_maxk=0.0,
):
    return InterpolatedImage(
        image,
        x_interpolant=x_interpolant,
        k_interpolant=k_interpolant,
        use_true_center=use_true_center,
        offset=offset,
        gsparams=gsparams,
        calculate_maxk=False,
        calculate_stepk=False,
        pad_factor=1.0,
        flux=jnp.sum(image.array),
        _force_stepk=force_stepk,
        _force_maxk=force_maxk,
        _recenter_image=False,
    )


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
        [jnp.zeros(x.shape, dtype=float), xi, yi, xp, yp, zp],
    )[0]
    return z.reshape(orig_shape)


@partial(jax.jit, static_argnums=(5,))
def _draw_with_interpolant_kval(kx, ky, kxmin, kymin, zp, interp):
    orig_shape = kx.shape
    kx = kx.ravel()
    kxi = jnp.floor(kx - kxmin).astype(jnp.int32)
    kxp = kxi + kxmin
    nkx_2 = zp.shape[1] - 1
    nkx = nkx_2 * 2

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
            zp[(nky - kyind) % nky, nkx - kxind - nkx_2].conjugate(),
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
        [jnp.zeros(kx.shape, dtype=complex), kxi, kyi, nky, nkx, nkx_2, kxp, kyp, zp],
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
