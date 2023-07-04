import galsim as _galsim
import jax.numpy as jnp
from jax._src.numpy.util import _wraps

from jax_galsim.gsparams import GSParams
from jax_galsim.position import Position, PositionD, PositionI
from jax_galsim.utilities import parse_pos_args


@_wraps(_galsim.GSObject)
class GSObject:
    def __init__(self, *, gsparams=None, **params):
        self._params = params  # Dictionary containing all traced parameters
        self._gsparams = gsparams  # Non-traced static parameters

    @property
    def flux(self):
        """The flux of the profile."""
        return self._flux

    @property
    def _flux(self):
        """By default, the flux is contained in the parameters dictionay."""
        return self._params["flux"]

    @property
    def gsparams(self):
        """A `GSParams` object that sets various parameters relevant for speed/accuracy trade-offs."""
        return self._gsparams

    @property
    def params(self):
        """A Dictionary object containing all parameters of the internal represention of this object."""
        return self._params

    @property
    def maxk(self):
        """The value of k beyond which aliasing can be neglected."""
        return self._maxk

    @property
    def stepk(self):
        """The sampling in k space necessary to avoid folding of image in x space."""
        return self._stepk

    @property
    def nyquist_scale(self):
        """The pixel spacing that does not alias maxk."""
        return jnp.pi / self.maxk

    @property
    def has_hard_edges(self):
        """Whether there are any hard edges in the profile, which would require very small k
        spacing when working in the Fourier domain.
        """
        return self._has_hard_edges

    @property
    def is_axisymmetric(self):
        """Whether the profile is axially symmetric; affects efficiency of evaluation."""
        return self._is_axisymmetric

    @property
    def is_analytic_x(self):
        """Whether the real-space values can be determined immediately at any position without
        requiring a Discrete Fourier Transform.
        """
        return self._is_analytic_x

    @property
    def is_analytic_k(self):
        """Whether the k-space values can be determined immediately at any position without
        requiring a Discrete Fourier Transform.
        """
        return self._is_analytic_k

    @property
    def centroid(self):
        """The (x, y) centroid of an object as a `PositionD`."""
        return self._centroid

    @property
    def _centroid(self):
        # Most profiles are centered at 0,0, so make this the default.
        return PositionD(0, 0)

    @property
    @_wraps(_galsim.GSObject.max_sb)
    def max_sb(self):
        return self._max_sb

    @property
    def _max_sb(self):
        # The way this is used, overestimates are conservative.
        # So the default value of 1.e500 will skip the optimization involving the maximum sb.
        return 1.0e500

    def __add__(self, other):
        """Add two GSObjects.

        Equivalent to Add(self, other)
        """
        from jax_galsim.sum import Sum

        return Sum([self, other])

    # op- is unusual, but allowed.  It subtracts off one profile from another.
    def __sub__(self, other):
        """Subtract two GSObjects.

        Equivalent to Add(self, -1 * other)
        """
        from .sum import Add

        return Add([self, (-1.0 * other)])

    # Make op* work to adjust the flux of an object
    def __mul__(self, other):
        """Scale the flux of the object by the given factor.

        obj * flux_ratio is equivalent to obj.withScaledFlux(flux_ratio)

        It creates a new object that has the same profile as the original, but with the
        surface brightness at every location scaled by the given amount.

        You can also multiply by an `SED`, which will create a `ChromaticObject` where the `SED`
        acts like a wavelength-dependent ``flux_ratio``.
        """
        return self.withScaledFlux(other)

    def __rmul__(self, other):
        """Equivalent to obj * other.  See `__mul__` for details."""
        return self.__mul__(other)

    # Likewise for op/
    def __div__(self, other):
        """Equivalent to obj * (1/other).  See `__mul__` for details."""
        return self * (1.0 / other)

    __truediv__ = __div__

    def __neg__(self):
        return -1.0 * self

    def __eq__(self, other):
        is_same = self is other
        is_same_class = type(other) is self.__class__
        has_same_trees = self.tree_flatten() == other.tree_flatten()
        return is_same or (is_same_class and has_same_trees)

    @_wraps(_galsim.GSObject.xValue)
    def xValue(self, *args, **kwargs):
        pos = parse_pos_args(args, kwargs, "x", "y")
        return self._xValue(pos)

    def _xValue(self, pos):
        """Equivalent to `xValue`, but ``pos`` must be a PositionD.

        Parameters:
            pos: The position at which you want the surface brightness of the object.

        Returns:
            the surface brightness at that position.
        """
        raise NotImplementedError("%s does not implement xValue" % self.__class__.__name__)

    @_wraps(_galsim.GSObject.kValue)
    def kValue(self, *args, **kwargs):
        kpos = parse_pos_args(args, kwargs, "kx", "ky")
        return self._kValue(kpos)

    @_wraps(_galsim.GSObject.kValue)
    def kValue(self, *args, **kwargs):
        kpos = parse_pos_args(args, kwargs, "kx", "ky")
        return self._kValue(kpos)

    @_wraps(_galsim.GSObject.kValue)
    def kValue(self, *args, **kwargs):
        kpos = parse_pos_args(args, kwargs, "kx", "ky")
        return self._kValue(kpos)

    def _kValue(self, kpos):
        """Equivalent to `kValue`, but ``kpos`` must be a `galsim.PositionD` instance."""
        raise NotImplementedError("%s does not implement kValue" % self.__class__.__name__)

    def withGSParams(self, gsparams=None, **kwargs):
        """Create a version of the current object with the given `GSParams`."""
        if gsparams == self.gsparams:
            return self
        # Checking gsparams
        gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        # Flattening the representation to instantiate a clean new object
        children, aux_data = self.tree_flatten()
        aux_data["gsparams"] = gsparams
        return self.tree_unflatten(aux_data, children)

    def withScaledFlux(self, flux_ratio):
        from jax_galsim.transform import _Transform

        return _Transform(self, flux_ratio=flux_ratio)

    # Make sure the image is defined with the right size and wcs for drawImage()
    def _setup_image(self, image, nx, ny, bounds, add_to_image, dtype, center, odd=False):
        from jax_galsim.bounds import BoundsI
        from jax_galsim.image import Image

        # If image is given, check validity of nx,ny,bounds:
        if image is not None:
            if bounds is not None:
                raise _galsim.GalSimIncompatibleValuesError(
                    "Cannot provide bounds if image is provided",
                    bounds=bounds,
                    image=image,
                )
            if nx is not None or ny is not None:
                raise _galsim.GalSimIncompatibleValuesError(
                    "Cannot provide nx,ny if image is provided",
                    nx=nx,
                    ny=ny,
                    image=image,
                )
            if dtype is not None and image.array.dtype != dtype:
                raise _galsim.GalSimIncompatibleValuesError(
                    "Cannot specify dtype != image.array.dtype if image is provided",
                    dtype=dtype,
                    image=image,
                )

            # Resize the given image if necessary
            if not image.bounds.isDefined():
                # Can't add to image if need to resize
                if add_to_image:
                    raise _galsim.GalSimIncompatibleValuesError(
                        "Cannot add_to_image if image bounds are not defined",
                        add_to_image=add_to_image,
                        image=image,
                    )
                N = self.getGoodImageSize(1.0)
                if odd:
                    N += 1
                bounds = BoundsI(1, N, 1, N)
                image.resize(bounds)
            # Else use the given image as is

        # Otherwise, make a new image
        else:
            # Can't add to image if none is provided.
            if add_to_image:
                raise _galsim.GalSimIncompatibleValuesError(
                    "Cannot add_to_image if image is None",
                    add_to_image=add_to_image,
                    image=image,
                )
            # Use bounds or nx,ny if provided
            if bounds is not None:
                if nx is not None or ny is not None:
                    raise _galsim.GalSimIncompatibleValuesError(
                        "Cannot set both bounds and (nx, ny)",
                        nx=nx,
                        ny=ny,
                        bounds=bounds,
                    )
                if not bounds.isDefined():
                    raise _galsim.GalSimValueError("Cannot use undefined bounds", bounds)
                image = Image.init(bounds=bounds, dtype=dtype)
            elif nx is not None or ny is not None:
                if nx is None or ny is None:
                    raise _galsim.GalSimIncompatibleValuesError(
                        "Must set either both or neither of nx, ny", nx=nx, ny=ny
                    )
                image = Image.init(nx, ny, dtype=dtype)
                if center is not None:
                    image.shift(
                        PositionI(
                            jnp.floor(center.x + 0.5 - image.true_center.x),
                            jnp.floor(center.y + 0.5 - image.true_center.y),
                        )
                    )
            else:
                N = self.getGoodImageSize(1.0)
                if odd:
                    N += 1
                image = Image.init(N, N, dtype=dtype)
                if center is not None:
                    image.setCenter(PositionI(jnp.ceil(center.x), jnp.ceil(center.y)))

        return image

    def _local_wcs(self, wcs, image, offset, center, use_true_center, new_bounds):
        # Get the local WCS at the location of the object.

        if wcs.isUniform:
            return wcs.local()
        elif image is None:
            bounds = new_bounds
        else:
            bounds = image.bounds
        if not bounds.isDefined():
            raise _galsim.GalSimIncompatibleValuesError(
                "Cannot provide non-local wcs with automatically sized image",
                wcs=wcs,
                image=image,
                bounds=new_bounds,
            )
        elif center is not None:
            obj_cen = center
        elif use_true_center:
            obj_cen = bounds.true_center
        else:
            obj_cen = bounds.center
            # Convert from PositionI to PositionD
            obj_cen = PositionD(obj_cen.x, obj_cen.y)
        # _parse_offset has already turned offset=None into PositionD(0,0), so it is safe to add.
        obj_cen += offset
        return wcs.local(image_pos=obj_cen)

    def _parse_offset(self, offset):
        if offset is None:
            return PositionD(0, 0)
        elif isinstance(offset, Position):
            return PositionD(offset.x, offset.y)
        else:
            # Let python raise the appropriate exception if this isn't valid.
            return PositionD(offset[0], offset[1])

    def _parse_center(self, center):
        # Almost the same as _parse_offset, except we leave it as None in that case.
        if center is None:
            return None
        elif isinstance(center, Position):
            return PositionD(center.x, center.y)
        else:
            # Let python raise the appropriate exception if this isn't valid.
            return PositionD(center[0], center[1])

    def _get_new_bounds(self, image, nx, ny, bounds, center):
        from jax_galsim.bounds import BoundsI

        if image is not None and image.bounds.isDefined():
            return image.bounds
        elif nx is not None and ny is not None:
            b = BoundsI(1, nx, 1, ny)
            if center is not None:
                b = b.shift(
                    PositionI(
                        jnp.floor(center.x + 0.5) - b.center.x,
                        jnp.floor(center.y + 0.5) - b.center.y,
                    )
                )
            return b
        elif bounds is not None and bounds.isDefined():
            return bounds
        else:
            return BoundsI()

    def _adjust_offset(self, new_bounds, offset, center, use_true_center):
        # Note: this assumes self is in terms of image coordinates.
        if center is not None:
            if new_bounds.isDefined():
                offset += center - new_bounds.center
            else:
                # Then will be created as even sized image.
                offset += PositionD(center.x - jnp.ceil(center.x), center.y - jnp.ceil(center.y))
        elif use_true_center:
            # For even-sized images, the SBProfile draw function centers the result in the
            # pixel just up and right of the real center.  So shift it back to make sure it really
            # draws in the center.
            # Also, remember that numpy's shape is ordered as [y,x]
            dx = offset.x
            dy = offset.y
            shape = new_bounds.numpyShape()
            dx -= 0.5 * ((shape[1] + 1) % 2)
            dy -= 0.5 * ((shape[0] + 1) % 2)

            # if shape[1] % 2 == 0: dx -= 0.5
            # if shape[0] % 2 == 0: dy -= 0.5
            offset = PositionD(dx, dy)
        return offset

    def _determine_wcs(self, scale, wcs, image, default_wcs=None):
        from jax_galsim.wcs import BaseWCS, PixelScale

        # Determine the correct wcs given the input scale, wcs and image.
        if wcs is not None:
            if scale is not None:
                raise _galsim.GalSimIncompatibleValuesError(
                    "Cannot provide both wcs and scale", wcs=wcs, scale=scale
                )
            if not isinstance(wcs, BaseWCS):
                raise TypeError("wcs must be a BaseWCS instance")
            if image is not None:
                image.wcs = None
        elif scale is not None:
            wcs = PixelScale(scale)
            if image is not None:
                image.wcs = None
        elif image is not None and image.wcs is not None:
            wcs = image.wcs

        # If the input scale <= 0, or wcs is still None at this point, then use the Nyquist scale:
        # TODO: we will need to remove this test of scale for jitting
        # if wcs is None or (wcs.isPixelScale and wcs.scale <= 0):
        if wcs is None:
            if default_wcs is None:
                wcs = PixelScale(self.nyquist_scale)
            else:
                wcs = default_wcs
        return wcs

    @_wraps(_galsim.GSObject.drawImage)
    def drawImage(
        self,
        image=None,
        nx=None,
        ny=None,
        bounds=None,
        scale=None,
        wcs=None,
        dtype=None,
        method="auto",
        area=1.0,
        exptime=1.0,
        gain=1.0,
        add_to_image=False,
        center=None,
        use_true_center=True,
        offset=None,
        n_photons=0.0,
        rng=None,
        max_extra_noise=0.0,
        poisson_flux=None,
        sensor=None,
        photon_ops=(),
        n_subsample=3,
        maxN=None,
        save_photons=False,
        bandpass=None,
        setup_only=False,
        surface_ops=None,
    ):
        from jax_galsim.wcs import PixelScale

        # Figure out what wcs we are going to use.
        wcs = self._determine_wcs(scale, wcs, image)

        # Make sure offset and center are PositionD, converting from other formats (tuple, array,..)
        # Note: If None, offset is converted to PositionD(0,0), but center will remain None.
        offset = self._parse_offset(offset)
        center = self._parse_center(center)

        # Determine the bounds of the new image for use below (if it can be known yet)
        new_bounds = self._get_new_bounds(image, nx, ny, bounds, center)

        # Get the local WCS, accounting for the offset correctly.
        local_wcs = self._local_wcs(wcs, image, offset, center, use_true_center, new_bounds)

        # Account for area and exptime.
        flux_scale = area * exptime
        # For surface brightness normalization, also scale by the pixel area.
        if method == "sb":
            flux_scale /= local_wcs.pixelArea()
        # Only do the gain here if not photon shooting, since need the number of photons to
        # reflect that actual photons, not ADU.
        if gain != 1 and method != "phot" and sensor is None:
            flux_scale /= gain

        # Determine the offset, and possibly fix the centering for even-sized images
        offset = self._adjust_offset(new_bounds, offset, center, use_true_center)

        # Convert the profile in world coordinates to the profile in image coordinates:
        prof = local_wcs.profileToImage(self, flux_ratio=flux_scale, offset=offset)

        local_wcs = local_wcs.shiftOrigin(offset)

        # Make sure image is setup correctly
        image = prof._setup_image(image, nx, ny, bounds, add_to_image, dtype, center)
        image.wcs = wcs

        if setup_only:
            image.added_flux = 0.0
            return image

        # Making a view of the image lets us change the center without messing up the original.
        original_center = image.center
        wcs = image.wcs
        image.setCenter(0, 0)
        image.wcs = PixelScale(1.0)
        image = prof.drawReal(image, add_to_image)
        image.shift(original_center)
        image.wcs = wcs
        return image

    @_wraps(_galsim.GSObject.drawReal)
    def drawReal(self, image, add_to_image=False):
        if image.wcs is None or not image.wcs.isPixelScale:
            raise _galsim.GalSimValueError(
                "drawReal requires an image with a PixelScale wcs", image
            )
        im1 = self._drawReal(image)
        if add_to_image:
            return image + im1
        else:
            return im1

    def _drawReal(self, image, jac=None, offset=(0.0, 0.0), flux_scaling=1.0):
        """A version of `drawReal` without the sanity checks or some options.

        This is nearly equivalent to the regular ``drawReal(image, add_to_image=False)``, but
        the image's dtype must be either float32 or float64, and it must have a c_contiguous array
        (``image.iscontiguous`` must be True).
        """
        raise NotImplementedError("%s does not implement drawReal" % self.__class__.__name__)

    def getGoodImageSize(self, pixel_scale):
        """Return a good size to use for drawing this profile.

        The size will be large enough to cover most of the flux of the object.  Specifically,
        at least (1-gsparams.folding_threshold) (i.e. 99.5% by default) of the flux should fall
        in the image.

        Also, the returned size is always an even number, which is usually desired in practice.
        Of course, if you prefer an odd-sized image, you can add 1 to the result.

        Parameters:
            pixel_scale:    The desired pixel scale of the image to be built.

        Returns:
            N, a good (linear) size of an image on which to draw this object.
        """
        # Start with a good size from stepk and the pixel scale
        Nd = 2.0 * jnp.pi / (pixel_scale * self.stepk)

        # Make it an integer
        # (Some slop to keep from getting extra pixels due to roundoff errors in calculations.)
        N = jnp.ceil(Nd * (1.0 - 1.0e-12)).astype(int)

        # Round up to an even value
        N = 2 * ((N + 1) // 2)
        return N

    def tree_flatten(self):
        """This function flattens the GSObject into a list of children
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
