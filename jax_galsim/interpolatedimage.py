import copy
import math
import textwrap
from functools import partial

import galsim as _galsim
import jax
import jax.numpy as jnp
import jax.random as jrng
from galsim.errors import (
    GalSimIncompatibleValuesError,
    GalSimRangeError,
    GalSimUndefinedBoundsError,
    GalSimValueError,
)
from galsim.utilities import doc_inherit
from jax.tree_util import register_pytree_node_class

from jax_galsim import fits
from jax_galsim.bounds import BoundsI
from jax_galsim.core.utils import (
    compute_major_minor_from_jacobian,
    ensure_hashable,
    implements,
)
from jax_galsim.gsobject import GSObject
from jax_galsim.gsparams import GSParams
from jax_galsim.image import Image
from jax_galsim.interpolant import Quintic
from jax_galsim.photon_array import PhotonArray
from jax_galsim.position import PositionD
from jax_galsim.random import UniformDeviate
from jax_galsim.transform import Transformation
from jax_galsim.utilities import convert_interpolant, lazy_property
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


@implements(
    _galsim.InterpolatedImage,
    lax_description=textwrap.dedent(
        """The JAX equivalent of galsim.InterpolatedImage does not support

            - noise padding
            - the pad_image options
            - depixelize
            - most of the bounds checks, type checks, and dtype casts done by galsim
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
        # this can happen due to incomplete initialization
        _original = getattr(self, "_original", None)

        if _original is None:
            return "galsim.InterpolatedImage(None)"

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

    @implements(_galsim.InterpolatedImage.withGSParams)
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
        self._workspace = {}
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
        return ret

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("_workspace")
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._workspace = {}

    @property
    def x_interpolant(self):
        """The real-space `Interpolant` for this profile."""
        return self._x_interpolant

    @property
    def k_interpolant(self):
        """The Fourier-space `Interpolant` for this profile."""
        return self._k_interpolant

    @lazy_property
    def image(self):
        """The underlying `Image` being interpolated."""
        return self._xim[self._image.bounds]

    @property
    def _flux(self):
        return self._image_flux

    @lazy_property
    def _centroid(self):
        x, y = self._pad_image.get_pixel_centers()
        tot = jnp.sum(self._pad_image.array)
        xpos = jnp.sum(x * self._pad_image.array) / tot
        ypos = jnp.sum(y * self._pad_image.array) / tot
        return PositionD(xpos, ypos)

    @lazy_property
    def _max_sb(self):
        return jnp.max(jnp.abs(self._pad_image.array))

    @lazy_property
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

    @lazy_property
    def _image_flux(self):
        return jnp.sum(self._image.array, dtype=float)

    @lazy_property
    def _offset(self):
        # Figure out the offset to apply based on the original image (not the padded one).
        # We will apply this below in _sbp.
        offset = self._parse_offset(self._jax_children[1]["offset"])
        return self._adjust_offset(
            self._image.bounds, offset, None, self._jax_aux_data["use_true_center"]
        )

    @lazy_property
    def _image(self):
        # Store the image as an attribute and make sure we don't change the original image
        # in anything we do here.  (e.g. set scale, etc.)
        if self._jax_aux_data["depixelize"]:
            # TODO: no depixelize in jax_galsim
            # self._image = image.view(dtype=np.float64).depixelize(self._x_interpolant)
            raise NotImplementedError(
                "InterpolatedImages do not support 'depixelize' in jax_galsim."
            )
        else:
            image = self._jax_children[0].view(dtype=float)

        if self._jax_aux_data["_recenter_image"]:
            image.setCenter(0, 0)

        return image

    @lazy_property
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

    @lazy_property
    def _jac_arr(self):
        image = self._jax_children[0]
        im_cen = (
            image.true_center if self._jax_aux_data["use_true_center"] else image.center
        )
        return self._wcs.jacobian(image_pos=im_cen).getMatrix().ravel()

    @lazy_property
    def _xim(self):
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
        # after the call to setCenter you get a WCS with an offset in
        # it instead of a pure pixel scale
        xim.wcs = PixelScale(1.0)

        # Now place the given image in the center of the padding image:
        xim[self._image.bounds] = self._image

        return xim

    @lazy_property
    def _pad_image(self):
        # These next two allow for easy pickling/repring.  We don't need to serialize all the
        # zeros around the edge.  But we do need to keep any non-zero padding as a pad_image.
        xim = self._xim
        nz_bounds = self._image.bounds
        return xim[nz_bounds]

    @lazy_property
    def _kim(self):
        return self._xim.calculate_fft()

    @lazy_property
    def _maxk(self):
        if self._jax_aux_data["_force_maxk"]:
            _, minor = compute_major_minor_from_jacobian(self._jac_arr.reshape((2, 2)))
            return self._jax_aux_data["_force_maxk"] * minor
        else:
            return self._getMaxK(self._jax_aux_data["calculate_maxk"])

    @lazy_property
    def _stepk(self):
        if self._jax_aux_data["_force_stepk"]:
            _, minor = compute_major_minor_from_jacobian(self._jac_arr.reshape((2, 2)))
            return self._jax_aux_data["_force_stepk"] * minor
        else:
            return self._getStepK(self._jax_aux_data["calculate_stepk"])

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

    def _xValue(self, pos):
        x = jnp.array([pos.x], dtype=float)
        y = jnp.array([pos.y], dtype=float)
        return _xValue_arr(
            x,
            y,
            self._offset.x,
            self._offset.y,
            self._pad_image.bounds.xmin,
            self._pad_image.bounds.ymin,
            self._pad_image.array,
            self._x_interpolant,
        )[0]

    def _kValue(self, kpos):
        kx = jnp.array([kpos.x], dtype=float)
        ky = jnp.array([kpos.y], dtype=float)
        return _kValue_arr(
            kx,
            ky,
            self._offset.x,
            self._offset.y,
            self._kim.bounds.xmin,
            self._kim.bounds.ymin,
            self._kim.array,
            self._kim.scale,
            self._x_interpolant,
            self._k_interpolant,
        )[0]

    def _drawReal(self, image, jac=None, offset=(0.0, 0.0), flux_scaling=1.0):
        jacobian = jnp.eye(2) if jac is None else jac

        flux_scaling *= image.scale**2

        # Create an array of coordinates
        coords = jnp.stack(image.get_pixel_centers(), axis=-1)
        coords = coords * image.scale  # Scale by the image pixel scale
        coords = coords - jnp.asarray(offset)  # Add the offset

        # Apply the jacobian transformation
        inv_jacobian = jnp.linalg.inv(jacobian)
        _, logdet = jnp.linalg.slogdet(inv_jacobian)
        coords = jnp.dot(coords, inv_jacobian.T)
        flux_scaling *= jnp.exp(logdet)

        im = _xValue_arr(
            coords[..., 0],
            coords[..., 1],
            self._offset.x,
            self._offset.y,
            self._pad_image.bounds.xmin,
            self._pad_image.bounds.ymin,
            self._pad_image.array,
            self._x_interpolant,
        )

        # Apply the flux scaling
        im = (im * flux_scaling).astype(image.dtype)

        # Return an image
        return Image(array=im, bounds=image.bounds, wcs=image.wcs, _check_bounds=False)

    def _drawKImage(self, image, jac=None):
        jacobian = jnp.eye(2) if jac is None else jac

        # Create an array of coordinates
        coords = jnp.stack(image.get_pixel_centers(), axis=-1)
        coords = coords * image.scale  # Scale by the image pixel scale
        coords = jnp.dot(coords, jacobian)

        im = _kValue_arr(
            coords[..., 0],
            coords[..., 1],
            self._offset.x,
            self._offset.y,
            self._kim.bounds.xmin,
            self._kim.bounds.ymin,
            self._kim.array,
            self._kim.scale,
            self._x_interpolant,
            self._k_interpolant,
        )
        im = (im).astype(image.dtype)

        # Return an image
        return Image(array=im, bounds=image.bounds, wcs=image.wcs, _check_bounds=False)

    @lazy_property
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

    @property
    def _positive_flux(self):
        return self._pos_neg_fluxes[0]

    @property
    def _negative_flux(self):
        return self._pos_neg_fluxes[1]

    def _flux_per_photon(self):
        return self._calculate_flux_per_photon()

    def _shoot(self, photons, rng):
        # we first draw the index location from the image
        img = self._pad_image
        subkey = rng._state.split_one()
        inds = jrng.choice(
            subkey,
            img.array.size,
            shape=(photons.size(),),
            replace=True,
            # we use abs here since some of the pixels could be negative
            # and for a noise image this procedure results in a fair
            # sampling of the noise
            p=jnp.abs(img.array.ravel()) / jnp.sum(jnp.abs(img.array)),
        ).astype(int)
        yinds, xinds = jnp.unravel_index(inds, img.array.shape)

        xedges = jnp.arange(img.bounds.xmin, img.bounds.xmax + 2) - 0.5
        yedges = jnp.arange(img.bounds.ymin, img.bounds.ymax + 2) - 0.5

        # now we draw the position within the pixel
        ud = UniformDeviate(rng)
        photons.x = ud.generate(photons.x) + xedges[xinds]
        photons.y = ud.generate(photons.y) + yedges[yinds]
        # this magic set of factors comes from the galsim C++ code in
        # a few spots it is
        #
        #  - the sign of the photon flux
        #  - the flux per photon = 1 - 2 neg / (pos + neg)
        #  - the total absolute flux in the image = (pos + neg)
        #  - the number of photons to draw = photons.size()
        #
        # If you unpack it all, then you get
        #
        #  sign * (1 - 2 neg / (pos + neg)) * (pos + neg) / photons.size()
        #  = sign * (pos + neg - 2 neg) / (pos + neg) * (pos + neg) / photons.size()
        #  = sign * (pos - neg) / photons.size()
        #
        # So what we have is a sign that oscillates between -1 and 1 with each photon getting
        # the flux of the object divided by the number of photons (which is inflated to get the total flux
        # correct by other bits of the code)
        photons.flux = (
            jnp.sign(img.array.ravel())[inds]
            * self._flux_per_photon()
            * (self.positive_flux + self.negative_flux)
            / photons.size()
        )

        # account for offset - we add the offset to get to
        # image pixels in the xValue method
        # here we generate photons from the image and
        # so we need to subtract it to get back to get to x as
        # it would be input in xVal
        photons.x -= self._offset.x
        photons.y -= self._offset.y

        # now we convolve with the x interpolant
        x_photons = PhotonArray(photons.size())
        self._x_interpolant._shoot(x_photons, rng)
        photons.convolve(x_photons)


@implements(_galsim._InterpolatedImage)
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


def _xValue_arr(x, y, x_offset, y_offset, xmin, ymin, arr, x_interpolant):
    vals = _draw_with_interpolant_xval(
        x + x_offset,
        y + y_offset,
        xmin,
        ymin,
        arr,
        x_interpolant,
    )
    return vals


@partial(jax.vmap, in_axes=(0, None, None, None, None, None))
@partial(jax.jit, static_argnames=("interp",))
def _interp_weight_1d_xval(ioff, xi, xp, x, nx, interp):
    xind = xi + ioff
    mskx = (xind >= 0) & (xind < nx)
    _x = x - (xp + ioff)
    wx = interp._xval_noraise(_x)
    wx = jnp.where(mskx, wx, 0)
    return wx, xind.astype(jnp.int32)


@partial(jax.jit, static_argnames=("interp",))
def _draw_with_interpolant_xval(x, y, xmin, ymin, zp, interp):
    """This helper function interpolates an image (`zp`) with an interpolant `interp`
    at the pixel locations given by `x`, `y`. The lower-left corner of the image is
    `xmin` / `ymin`.

    A more standard C/C++ code would have a set of nested for loops that iterates over each
    location to interpolate and then over the nterpolation kernel.

    In JAX, we instead write things such that the loop over the points to be interpolated
    is vectorized in the code. We represent the loops over the interpolation kernel as explicit
    for loops.
    """
    # the vectorization over the interpolation points is easier to think about
    # if they are in a 1D array. So we use ravel to flatten them and then reshape
    # at the end.
    orig_shape = x.shape

    # the variables here are
    #  x/y: the x/y coordinates of the points to be interpolated
    #  xi/yi: the index of the nerest pixel below the point
    #  xp/yp: the x/y coordinate of the nearest pixel below the point
    #  nx/ny: the size of the x/y arrays
    x = x.ravel()
    xi = jnp.floor(x - xmin).astype(jnp.int32)
    xp = xi + xmin
    nx = zp.shape[1]

    y = y.ravel()
    yi = jnp.floor(y - ymin).astype(jnp.int32)
    yp = yi + ymin
    ny = zp.shape[0]

    irange = interp.ixrange // 2
    iinds = jnp.arange(-irange, irange + 1)

    wx, xind = _interp_weight_1d_xval(
        iinds,
        xi,
        xp,
        x,
        nx,
        interp,
    )

    wy, yind = _interp_weight_1d_xval(
        iinds,
        yi,
        yp,
        y,
        ny,
        interp,
    )

    z = jnp.sum(
        wx[None, :, :] * wy[:, None, :] * zp[yind[:, None, :], xind[None, :, :]],
        axis=(0, 1),
    )

    # we reshape on the way out to match the input shape
    return z.reshape(orig_shape)


def _kValue_arr(
    kx,
    ky,
    x_offset,
    y_offset,
    kxmin,
    kymin,
    arr,
    scale,
    x_interpolant,
    k_interpolant,
):
    # phase factor due to offset
    # not we shift by -offset which explains the sign
    # in the exponent
    pfac = jnp.exp(1j * (kx * x_offset + ky * y_offset))

    kxi = kx / scale
    kyi = ky / scale

    _uscale = 1.0 / (2.0 * jnp.pi)
    _maxk_xint = x_interpolant.urange() / _uscale / scale

    # here we do the actual inteprolation in k space
    val = _draw_with_interpolant_kval(
        kxi,
        kyi,
        kymin,  # this is not a bug! we need the minimum for the full periodic space
        kymin,
        arr,
        k_interpolant,
    )

    # finally we multiply by the FFT of the real-space interpolation function
    # and mask any values that are outside the range of the real-space interpolation
    # FFT
    msk = (jnp.abs(kxi) <= _maxk_xint) & (jnp.abs(kyi) <= _maxk_xint)
    xint_val = x_interpolant._kval_noraise(kx) * x_interpolant._kval_noraise(ky)
    return jnp.where(msk, val * xint_val * pfac, 0.0)


@partial(jax.vmap, in_axes=(0, None, None, None, None, None))
@partial(jax.jit, static_argnames=("interp",))
def _interp_weight_1d_kval(ioff, kxi, kxp, kx, nkx, interp):
    kxind = (kxi + ioff) % nkx
    _kx = kx - (kxp + ioff)
    wkx = interp._xval_noraise(_kx)
    return wkx, kxind.astype(jnp.int32)


@partial(jax.jit, static_argnames=("interp",))
def _draw_with_interpolant_kval(kx, ky, kxmin, kymin, zp, interp):
    """This function interpolates complex k-space images and follows the
    same basic structure as _draw_with_interpolant_xval above.

    The key difference is that the k-space images are Hermitian and so
    only half of the data is actually in memory. We account for this by
    computing all of the interpolation weights and indicies as if we had
    the full image. Then finally, if we need a value that is not in memory,
    we get it from the values we have via the Hermitian symmetry.
    """
    # all of the code below is almost line-for-line the same as the
    # _draw_with_interpolant_xval function above.
    orig_shape = kx.shape
    kx = kx.ravel()
    kxi = jnp.floor(kx - kxmin).astype(jnp.int32)
    kxp = kxi + kxmin
    # this is the number of pixels in the half image and is needed
    # for computing values via Hermition symmetry below
    nkx_2 = zp.shape[1] - 1
    nkx = nkx_2 * 2

    ky = ky.ravel()
    kyi = jnp.floor(ky - kymin).astype(jnp.int32)
    kyp = kyi + kymin
    nky = zp.shape[0]

    wkx, kxind = _interp_weight_1d_kval(
        jnp.arange(-interp.xrange, interp.xrange + 1),
        kxi,
        kxp,
        kx,
        nkx,
        interp,
    )

    wky, kyind = _interp_weight_1d_kval(
        jnp.arange(-interp.xrange, interp.xrange + 1),
        kyi,
        kyp,
        ky,
        nky,
        interp,
    )

    wkx = wkx[None, :, :]
    kxind = kxind[None, :, :]
    wky = wky[:, None, :]
    kyind = kyind[:, None, :]

    # this is the key difference from the xval function
    # we need to use the Hermitian symmetry to get the
    # values that are not in memory
    # in memory we have the values at nkx_2 to nkx - 1
    # the Hermitian symmetry is that
    #   f(ky, kx) = conjugate(f(-kx, -ky))
    # In indices this is a symmetric flip about the central
    # pixels at kx = ky = 0.
    # we do not need to mask any values that run off the edge of the image
    # since we rewrap them using the periodicity of the image.

    val = jnp.where(
        kxind < nkx_2,
        zp[(nky - kyind) % nky, nkx - kxind - nkx_2].conjugate(),
        zp[kyind, kxind - nkx_2],
    )
    z = jnp.sum(
        val * wkx * wky,
        axis=(0, 1),
    )

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


@jax.jit
def _calculate_size_containing_flux(image, thresh):
    cenx, ceny = image.center.x, image.center.y
    x, y = image.get_pixel_centers()
    fluxes = _flux_frac(image.array, x, y, cenx, ceny)
    msk = fluxes >= -jnp.inf
    fluxes = jnp.where(msk, fluxes, jnp.max(fluxes))
    d = jnp.arange(image.array.shape[0]) + 1.0
    # below we use a linear interpolation table to find the maximum size
    # in pixels that contains a given flux (called thresh here)
    # expfac controls how much we oversample the interpolation table
    # in order to return a more accurate result
    # we have it hard coded at 4 to compromise between speed and accuracy
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
    msk = (arr * arr.conjugate()).real > thresh * thresh
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


@jax.jit
def _find_maxk(kim, max_maxk, thresh):
    kx, ky = kim.get_pixel_centers()
    kx *= kim.scale
    ky *= kim.scale
    # this minimum bounds the empirically determined
    # maxk from the image (computed by _inner_comp_find_maxk)
    # by max_maxk from above
    return jnp.minimum(
        _inner_comp_find_maxk(kim.array, thresh, kx, ky),
        max_maxk,
    )
