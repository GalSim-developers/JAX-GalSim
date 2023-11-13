import galsim as _galsim
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.numpy.util import _wraps

import jax_galsim.photon_array as pa
from jax_galsim.core.utils import is_equal_with_arrays
from jax_galsim.errors import (
    GalSimError,
    GalSimIncompatibleValuesError,
    GalSimNotImplementedError,
    GalSimRangeError,
    GalSimValueError,
    galsim_warn,
)
from jax_galsim.gsparams import GSParams
from jax_galsim.position import Position, PositionD, PositionI
from jax_galsim.random import BaseDeviate, PoissonDeviate
from jax_galsim.sensor import Sensor
from jax_galsim.utilities import parse_pos_args


@_wraps(_galsim.GSObject)
class GSObject:
    def __init__(self, *, gsparams=None, **params):
        self._params = params  # Dictionary containing all traced parameters
        self._gsparams = GSParams.check(gsparams)  # Non-traced static parameters
        self._workspace = {}  # used by lazy_property

    def __getstate__(self):
        d = self.__dict__.copy()
        d["had_workspace"] = "_workspace" in d
        d.pop("_workspace", None)
        return d

    def __setstate__(self, d):
        if d.pop("had_workspace", False):
            d["_workspace"] = {}
        self.__dict__ = d

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
    @_wraps(_galsim.GSObject.positive_flux)
    def positive_flux(self):
        return self._positive_flux

    @property
    @_wraps(_galsim.GSObject.negative_flux)
    def negative_flux(self):
        return self._negative_flux

    @property
    def _positive_flux(self):
        return self.flux + self._negative_flux

    @property
    def _negative_flux(self):
        return 0.0

    @property
    def _flux_per_photon(self):
        # The usual case.
        return 1.0

    def _calculate_flux_per_photon(self):
        # If negative_flux is overriden, then _flux_per_photon should be overridden as well
        # to return this calculation.
        posflux = self.positive_flux
        negflux = self.negative_flux
        eta = negflux / (posflux + negflux)
        return 1.0 - 2.0 * eta

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
        return (self is other) or (
            (type(other) is self.__class__)
            and is_equal_with_arrays(self.tree_flatten(), other.tree_flatten())
        )

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
        raise NotImplementedError(
            "%s does not implement xValue" % self.__class__.__name__
        )

    @_wraps(_galsim.GSObject.kValue)
    def kValue(self, *args, **kwargs):
        kpos = parse_pos_args(args, kwargs, "kx", "ky")
        return self._kValue(kpos)

    def _kValue(self, kpos):
        """Equivalent to `kValue`, but ``kpos`` must be a `galsim.PositionD` instance."""
        raise NotImplementedError(
            "%s does not implement kValue" % self.__class__.__name__
        )

    @_wraps(_galsim.GSObject.withGSParams)
    def withGSParams(self, gsparams=None, **kwargs):
        if gsparams == self.gsparams:
            return self
        # Checking gsparams
        gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        # Flattening the representation to instantiate a clean new object
        children, aux_data = self.tree_flatten()
        aux_data["gsparams"] = gsparams
        return self.tree_unflatten(aux_data, children)

    @_wraps(_galsim.GSObject.withFlux)
    def withFlux(self, flux):
        return self.withScaledFlux(flux / self.flux)

    @_wraps(_galsim.GSObject.withScaledFlux)
    def withScaledFlux(self, flux_ratio):
        from jax_galsim.transform import Transform

        return Transform(self, flux_ratio=flux_ratio)

    @_wraps(_galsim.GSObject.expand)
    def expand(self, scale):
        from jax_galsim.transform import Transform

        return Transform(self, jac=[scale, 0.0, 0.0, scale])

    @_wraps(_galsim.GSObject.dilate)
    def dilate(self, scale):
        from jax_galsim.transform import Transform

        # equivalent to self.expand(scale) * (1./scale**2)
        return Transform(self, jac=[scale, 0.0, 0.0, scale], flux_ratio=scale**-2)

    @_wraps(_galsim.GSObject.magnify)
    def magnify(self, mu):
        return self.expand(jnp.sqrt(mu))

    @_wraps(_galsim.GSObject.shear)
    def shear(self, *args, **kwargs):
        from jax_galsim.shear import Shear
        from jax_galsim.transform import Transform

        if len(args) == 1:
            shear = args[0]
        if len(args) == 1:
            if kwargs:
                raise TypeError(
                    "Error, gave both unnamed and named arguments to GSObject.shear!"
                )
            if not isinstance(args[0], Shear):
                raise TypeError(
                    "Error, unnamed argument to GSObject.shear is not a Shear!"
                )
            shear = args[0]
        elif len(args) > 1:
            raise TypeError("Error, too many unnamed arguments to GSObject.shear!")
        elif len(kwargs) == 0:
            raise TypeError("Error, shear argument is required")
        else:
            shear = Shear(**kwargs)
        return Transform(self, shear.getMatrix())

    def _shear(self, shear):
        """Equivalent to `GSObject.shear`, but without the overhead of sanity checks or other
        ways to input the ``shear`` value.

        Also, it won't propagate any noise attribute.

        Parameters:
            shear:      The `Shear` to be applied.

        Returns:
            the sheared object.
        """
        from jax_galsim.transform import Transform

        return Transform(self, shear.getMatrix())

    def lens(self, g1, g2, mu):
        """Create a version of the current object with both a lensing shear and magnification
        applied to it.

        This `GSObject` method applies a lensing (reduced) shear and magnification.  The shear must
        be specified using the g1, g2 definition of shear (see `Shear` for more details).
        This is the same definition as the outputs of the PowerSpectrum and NFWHalo classes, which
        compute shears according to some lensing power spectrum or lensing by an NFW dark matter
        halo.  The magnification determines the rescaling factor for the object area and flux,
        preserving surface brightness.

        Parameters:
            g1:         First component of lensing (reduced) shear to apply to the object.
            g2:         Second component of lensing (reduced) shear to apply to the object.
            mu:         Lensing magnification to apply to the object.  This is the factor by which
                        the solid angle subtended by the object is magnified, preserving surface
                        brightness.

        Returns:
            the lensed object.
        """
        from jax_galsim.shear import Shear
        from jax_galsim.transform import Transform

        shear = Shear(g1=g1, g2=g2)
        return Transform(self, shear.getMatrix() * jnp.sqrt(mu))

    def _lens(self, g1, g2, mu):
        """Equivalent to `GSObject.lens`, but without the overhead of some of the sanity checks.

        Also, it won't propagate any noise attribute.

        Parameters:
            g1:         First component of lensing (reduced) shear to apply to the object.
            g2:         Second component of lensing (reduced) shear to apply to the object.
            mu:         Lensing magnification to apply to the object.  This is the factor by which
                        the solid angle subtended by the object is magnified, preserving surface
                        brightness.

        Returns:
            the lensed object.
        """
        from .shear import _Shear
        from .transform import Transform

        shear = _Shear(g1 + 1j * g2)
        return Transform(self, shear.getMatrix() * jnp.sqrt(mu))

    def rotate(self, theta):
        """Rotate this object by an `Angle` ``theta``.

        Parameters:
            theta:      Rotation angle (`Angle` object, positive means anticlockwise).

        Returns:
            the rotated object.

        Note: Not differentiable with respect to theta (yet).
        """
        from jax_galsim.transform import Transform

        from .angle import Angle

        if not isinstance(theta, Angle):
            raise TypeError("Input theta should be an Angle")
        s, c = theta.sincos()
        return Transform(self, jac=[c, -s, s, c])

    @_wraps(_galsim.GSObject.transform)
    def transform(self, dudx, dudy, dvdx, dvdy):
        from jax_galsim.transform import Transform

        return Transform(self, jac=[dudx, dudy, dvdx, dvdy])

    @_wraps(_galsim.GSObject.shift)
    def shift(self, *args, **kwargs):
        from jax_galsim.transform import Transform

        offset = parse_pos_args(args, kwargs, "dx", "dy")
        return Transform(self, offset=offset)

    def _shift(self, dx, dy):
        """Equivalent to `shift`, but without the overhead of sanity checks or option
        to give the shift as a PositionD.

        Also, it won't propagate any noise attribute.

        Parameters:
            dx:         The x-component of the shift to apply
            dy:         The y-component of the shift to apply

        Returns:
            the shifted object.
        """
        from jax_galsim.transform import Transform

        new_obj = Transform(self, offset=(dx, dy))
        return new_obj

    # Make sure the image is defined with the right size and wcs for drawImage()
    def _setup_image(
        self, image, nx, ny, bounds, add_to_image, dtype, center, odd=False
    ):
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
                    raise _galsim.GalSimValueError(
                        "Cannot use undefined bounds", bounds
                    )
                image = Image(bounds=bounds, dtype=dtype)
            elif nx is not None or ny is not None:
                if nx is None or ny is None:
                    raise _galsim.GalSimIncompatibleValuesError(
                        "Must set either both or neither of nx, ny", nx=nx, ny=ny
                    )
                image = Image(nx, ny, dtype=dtype)
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
                image = Image(N, N, dtype=dtype)
                if center is not None:
                    image.setCenter(PositionI(jnp.ceil(center.x), jnp.ceil(center.y)))

        return image

    def _local_wcs(self, wcs, image, offset, center, use_true_center, new_bounds):
        # Get the local WCS at the location of the object.

        if wcs.isUniform():
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
                offset += PositionD(
                    center.x - jnp.ceil(center.x), center.y - jnp.ceil(center.y)
                )
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
        if wcs is None:
            if default_wcs is None:
                wcs = PixelScale(self.nyquist_scale)
            else:
                wcs = default_wcs

        if wcs.isPixelScale() and wcs.isLocal():
            wcs = jax.lax.cond(
                wcs.scale <= 0,
                lambda wcs, nqs: PixelScale(jnp.float_(nqs))
                if default_wcs is None
                else default_wcs,
                lambda wcs, nqs: PixelScale(jnp.float_(wcs.scale)),
                wcs,
                self.nyquist_scale,
            )

        return wcs

    @_wraps(
        _galsim.GSObject.drawImage,
        lax_description="""\
The JAX-GalSim version of `drawImage` does not

 - do extensive (any?) checking of the input settings.
 - does not support the deprecated `surface_ops` argument
""",
    )
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
    ):
        from jax_galsim.box import Pixel
        from jax_galsim.convolve import Convolve
        from jax_galsim.image import Image
        from jax_galsim.wcs import PixelScale

        if image is not None and not isinstance(image, Image):
            raise TypeError("image is not an Image instance", image)

        if method == "phot" and save_photons and maxN is not None:
            raise GalSimIncompatibleValuesError(
                "Setting maxN is incompatible with save_photons=True"
            )

        # Figure out what wcs we are going to use.
        wcs = self._determine_wcs(scale, wcs, image)

        # Make sure offset and center are PositionD, converting from other formats (tuple, array,..)
        # Note: If None, offset is converted to PositionD(0,0), but center will remain None.
        offset = self._parse_offset(offset)
        center = self._parse_center(center)

        # Determine the bounds of the new image for use below (if it can be known yet)
        new_bounds = self._get_new_bounds(image, nx, ny, bounds, center)

        # Get the local WCS, accounting for the offset correctly.
        local_wcs = self._local_wcs(
            wcs, image, offset, center, use_true_center, new_bounds
        )

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

        # If necessary, convolve by the pixel
        if method in ("auto", "fft", "real_space"):
            if method == "auto":
                real_space = None
            elif method == "fft":
                real_space = False
            else:
                real_space = True
            prof = Convolve(
                prof,
                Pixel(scale=1.0, gsparams=self.gsparams),
                real_space=real_space,
                gsparams=self.gsparams,
            )

        # Make sure image is setup correctly
        image = prof._setup_image(image, nx, ny, bounds, add_to_image, dtype, center)
        image_in = (
            image  # For compatibility with normal galsim, we update image_in below.
        )
        image.wcs = wcs

        if setup_only:
            image.added_flux = 0.0
            return image

        # Making a view of the image lets us change the center without messing up the original.
        original_center = image.center
        wcs = image.wcs
        image.setCenter(0, 0)
        image.wcs = PixelScale(1.0)

        if method == "phot":
            added_photons, photons = prof.drawPhot(
                image,
                gain,
                add_to_image,
                n_photons,
                rng,
                max_extra_noise,
                poisson_flux,
                sensor,
                photon_ops,
                maxN,
                original_center,
                local_wcs,
            )
        else:
            if sensor is not None:
                raise NotImplementedError(
                    "Sensor not yet implemented in drawImage for method != 'phot'."
                )

            if prof.is_analytic_x:
                added_photons = prof.drawReal(image, add_to_image)
            else:
                added_photons = prof.drawFFT(image, add_to_image)

        image.added_flux = added_photons / flux_scale
        # Restore the original center and wcs
        image.shift(original_center)
        image.wcs = wcs
        if save_photons:
            image.photons = photons

        # Update image_in to satisfy GalSim API
        image_in._array = image._array
        image_in.added_flux = image.added_flux
        image_in._bounds = image._bounds
        image_in.wcs = image.wcs
        image_in._dtype = image._dtype
        if save_photons:
            image_in.photons = photons

        return image

    @_wraps(_galsim.GSObject.drawReal)
    def drawReal(self, image, add_to_image=False):
        if image.wcs is None or not image.wcs.isPixelScale():
            raise _galsim.GalSimValueError(
                "drawReal requires an image with a PixelScale wcs", image
            )
        im1 = self._drawReal(image)
        temp = im1.subImage(image.bounds)
        if add_to_image:
            image._array = image._array + temp._array
        else:
            image._array = temp._array

        return temp.array.sum(dtype=float)

    def _drawReal(self, image, jac=None, offset=(0.0, 0.0), flux_scaling=1.0):
        """A version of `drawReal` without the sanity checks or some options.

        This is nearly equivalent to the regular ``drawReal(image, add_to_image=False)``, but
        the image's dtype must be either float32 or float64, and it must have a c_contiguous array
        (``image.iscontiguous`` must be True).
        """
        raise NotImplementedError(
            "%s does not implement drawReal" % self.__class__.__name__
        )

    @_wraps(_galsim.GSObject.getGoodImageSize)
    def getGoodImageSize(self, pixel_scale):
        # Start with a good size from stepk and the pixel scale
        Nd = 2.0 * jnp.pi / (pixel_scale * self.stepk)

        # Make it an integer
        # (Some slop to keep from getting extra pixels due to roundoff errors in calculations.)
        N = jnp.ceil(Nd * (1.0 - 1.0e-12)).astype(int)

        # Round up to an even value
        N = 2 * ((N + 1) // 2)
        return N

    @_wraps(_galsim.GSObject.drawFFT_makeKImage)
    def drawFFT_makeKImage(self, image):
        from jax_galsim.bounds import BoundsI
        from jax_galsim.image import ImageCD, ImageCF

        # Before any computations, let's check if we actually have a choice based on the gsparams.
        if self.gsparams.maximum_fft_size == self.gsparams.minimum_fft_size:
            with jax.ensure_compile_time_eval():
                Nk = self.gsparams.maximum_fft_size
                N = Nk
            dk = 2.0 * np.pi / (N * image.scale)
        else:
            # Start with what this profile thinks a good size would be given the image's pixel scale.
            N = self.getGoodImageSize(image.scale)

            # We must make something big enough to cover the target image size:
            image_N = jnp.max(
                jnp.array(
                    [
                        jnp.max(jnp.abs(jnp.array(image.bounds._getinitargs()))) * 2,
                        jnp.max(jnp.array(image.bounds.numpyShape())),
                    ]
                )
            )
            N = jnp.max(jnp.array([N, image_N]))

            # Round up to a good size for making FFTs:
            N = image.good_fft_size(N)

            # Make sure we hit the minimum size specified in the gsparams.
            N = max(N, self.gsparams.minimum_fft_size)

            dk = 2.0 * jnp.pi / (N * image.scale)

            maxk = self.maxk
            if N * dk / 2 > maxk:
                Nk = N
            else:
                # There will be aliasing.  Make a larger image and then wrap it.
                Nk = int(jnp.ceil(maxk / dk)) * 2

            if Nk > self.gsparams.maximum_fft_size:
                raise _galsim.GalSimFFTSizeError(
                    "drawFFT requires an FFT that is too large.", Nk
                )

        bounds = BoundsI(0, Nk // 2, -Nk // 2, Nk // 2)
        if image.dtype in (np.complex128, np.float64, np.int32, np.uint32):
            kimage = ImageCD(bounds=bounds, scale=dk)
        else:
            kimage = ImageCF(bounds=bounds, scale=dk)
        return kimage, N

    def drawFFT_finish(self, image, kimage, wrap_size, add_to_image):
        """
        This is a helper routine for drawFFT that finishes the calculation, based on the
        drawn k-space image.

        It applies the Fourier transform to ``kimage`` and adds the result to ``image``.

        Parameters:
            image:          The `Image` onto which to place the flux.
            kimage:         The k-space `Image` where the object was drawn.
            wrap_size:      The size of the region to wrap kimage, which must be either the same
                            size as kimage or smaller.
            add_to_image:   Whether to add flux to the existing image rather than clear out
                            anything in the image before drawing.

        Returns:
            The total flux drawn inside the image bounds.
        """
        from jax_galsim.bounds import BoundsI
        from jax_galsim.image import Image

        # Wrap the full image to the size we want for the FT.
        # Even if N == Nk, this is useful to make this portion properly Hermitian in the
        # N/2 column and N/2 row.
        bwrap = BoundsI(0, wrap_size // 2, -wrap_size // 2, wrap_size // 2 - 1)
        kimage_wrap = kimage._wrap(bwrap, True, False)

        # Perform the fourier transform.
        breal = BoundsI(
            -wrap_size // 2, wrap_size // 2 - 1, -wrap_size // 2, wrap_size // 2 - 1
        )
        kimg_shift = jnp.fft.ifftshift(kimage_wrap.array, axes=(-2,))
        real_image_arr = jnp.fft.fftshift(
            jnp.fft.irfft2(kimg_shift, breal.numpyShape())
        )
        real_image = Image(
            bounds=breal, array=real_image_arr, dtype=image.dtype, wcs=image.wcs
        )
        # Add (a portion of) this to the original image.
        temp = real_image.subImage(image.bounds)
        if add_to_image:
            image._array = image._array + temp._array
        else:
            image._array = temp._array

        return temp.array.sum(dtype=float)

    def drawFFT(self, image, add_to_image=False):
        """
        Draw this profile into an `Image` by computing the k-space image and performing an FFT.

        This is usually called from the `drawImage` function, rather than called directly by the
        user.  In particular, the input image must be already set up with defined bounds.  The
        profile will be drawn centered on whatever pixel corresponds to (0,0) with the given
        bounds, not the image center (unlike `drawImage`).  The image also must have a `PixelScale`
        wcs.  The profile being drawn should have already been converted to image coordinates via::

            >>> image_profile = original_wcs.toImage(original_profile)

        Note that the `Image` produced by drawFFT represents the profile sampled at the center
        of each pixel and then multiplied by the pixel area.  That is, the profile is NOT
        integrated over the area of the pixel.  This is equivalent to method='no_pixel' in
        `drawImage`.  If you want to render a profile integrated over the pixel, you can convolve
        with a `Pixel` first and draw that.

        Parameters:
            image:          The `Image` onto which to place the flux. [required]
            add_to_image:   Whether to add flux to the existing image rather than clear out
                            anything in the image before drawing. [default: False]

        Returns:
            The total flux drawn inside the image bounds.
        """
        if image.wcs is None or not image.wcs.isPixelScale():
            raise _galsim.GalSimValueError(
                "drawFFT requires an image with a PixelScale wcs", image
            )

        kimage, wrap_size = self.drawFFT_makeKImage(image)
        kimage = self._drawKImage(kimage)
        return self.drawFFT_finish(image, kimage, wrap_size, add_to_image)

    @_wraps(_galsim.GSObject.drawKImage)
    def drawKImage(
        self,
        image=None,
        nx=None,
        ny=None,
        bounds=None,
        scale=None,
        add_to_image=False,
        recenter=True,
        bandpass=None,
        setup_only=False,
    ):
        from jax_galsim.image import Image
        from jax_galsim.wcs import PixelScale

        # Make sure provided image is complex
        if image is not None:
            if not isinstance(image, Image):
                raise TypeError("Provided image must be galsim.Image", image)

            if not image.iscomplex:
                raise _galsim.GalSimValueError("Provided image must be complex", image)

        # Possibly get the scale from image.
        if image is not None and scale is None:
            # Grab the scale to use from the image.
            # This will raise a TypeError if image.wcs is not a PixelScale
            scale = image.scale

        # The input scale (via scale or image.scale) is really a dk value, so call it that for
        # clarity here, since we also need the real-space pixel scale, which we will call dx.
        if scale is None or scale <= 0:
            dk = self.stepk
        else:
            dk = scale
        if image is not None and image.bounds.isDefined():
            dx = np.pi / (max(image.array.shape) // 2 * dk)
        elif scale is None or scale <= 0:
            dx = self.nyquist_scale
        else:
            # Then dk = scale, which implies that we need to have dx smaller than nyquist_scale
            # by a factor of (dk/stepk)
            dx = self.nyquist_scale * dk / self.stepk

        # If the profile needs to be constructed from scratch, the _setup_image function will
        # do that, but only if the profile is in image coordinates for the real space image.
        # So make that profile.
        if image is None or not image.bounds.isDefined():
            real_prof = PixelScale(dx).profileToImage(self)
            dtype = np.complex128 if image is None else image.dtype
            image = real_prof._setup_image(
                image, nx, ny, bounds, add_to_image, dtype, center=None, odd=True
            )
        else:
            # Do some checks that setup_image would have done for us
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

        # Can't both recenter a provided image and add to it.
        if recenter and image.center != PositionI(0, 0) and add_to_image:
            raise _galsim.GalSimIncompatibleValuesError(
                "Cannot use add_to_image=True unless image is centered at (0,0) or recenter=False",
                recenter=recenter,
                image=image,
                add_to_image=add_to_image,
            )

        # Set the center to 0,0 if appropriate
        if recenter:
            image._shift(-image.center)

        # Set the wcs of the images to use the dk scale size
        image.scale = dk

        if setup_only:
            return image

        # For GalSim compatibility, we will attempt to update the input image
        image_in = image
        im2 = Image(bounds=image.bounds, dtype=image.dtype, scale=image.scale)
        im2 = self._drawKImage(im2)

        if not add_to_image:
            image._array = im2._array
        else:
            image._array = im2._array + image._array

        image_in._array = image._array
        image_in._bounds = image._bounds
        image_in.wcs = image.wcs
        image_in._dtype = image._dtype

        return image

    @_wraps(_galsim.GSObject._drawKImage)
    def _drawKImage(
        self, image, jac=None
    ):  # pragma: no cover  (all our classes override this)
        raise NotImplementedError(
            "%s does not implement drawKImage" % self.__class__.__name__
        )

    @_wraps(_galsim.GSObject._calculate_nphotons)
    def _calculate_nphotons(self, n_photons, poisson_flux, max_extra_noise, rng):
        # For profiles that are positive definite, then N = flux. Easy.
        #
        # However, some profiles shoot some of their photons with negative flux. This means that
        # we need a few more photons to get the right S/N = sqrt(flux). Take eta to be the
        # fraction of shot photons that have negative flux.
        #
        # S^2 = (N+ - N-)^2 = (N+ + N- - 2N-)^2 = (Ntot - 2N-)^2 = Ntot^2(1 - 2 eta)^2
        # N^2 = Var(S) = (N+ + N-) = Ntot
        #
        # So flux = (S/N)^2 = Ntot (1-2eta)^2
        # Ntot = flux / (1-2eta)^2
        #
        # However, if each photon has a flux of 1, then S = (1-2eta) Ntot = flux / (1-2eta).
        # So in fact, each photon needs to carry a flux of g = 1-2eta to get the right
        # total flux.
        #
        # That's all the easy case. The trickier case is when we are sky-background dominated.
        # Then we can usually get away with fewer shot photons than the above.  In particular,
        # if the noise from the photon shooting is much less than the sky noise, then we can
        # use fewer shot photons and essentially have each photon have a flux > 1. This is ok
        # as long as the additional noise due to this approximation is "much less than" the
        # noise we'll be adding to the image for the sky noise.
        #
        # Let's still have Ntot photons, but now each with a flux of g. And let's look at the
        # noise we get in the brightest pixel that has a nominal total flux of Imax.
        #
        # The number of photons hitting this pixel will be Imax/flux * Ntot.
        # The variance of this number is the same thing (Poisson counting).
        # So the noise in that pixel is:
        #
        # N^2 = Imax/flux * Ntot * g^2
        #
        # And the signal in that pixel will be:
        #
        # S = Imax/flux * (N+ - N-) * g which has to equal Imax, so
        # g = flux / Ntot(1-2eta)
        # N^2 = Imax/Ntot * flux / (1-2eta)^2
        #
        # As expected, we see that lowering Ntot will increase the noise in that (and every
        # other) pixel.
        # The input max_extra_noise parameter is the maximum value of spurious noise we want
        # to allow.
        #
        # So setting N^2 = Imax + nu, we get
        #
        # Ntot = flux / (1-2eta)^2 / (1 + nu/Imax)
        # g = (1 - 2eta) * (1 + nu/Imax)
        #
        # Returns the total flux placed inside the image bounds by photon shooting.
        #

        flux = self.flux
        if flux == 0.0:
            return 0, 1.0

        # The _flux_per_photon property is (1-2eta)
        # This factor will already be accounted for by the shoot function, so don't include
        # that as part of our scaling here.  There may be other adjustments though, so g=1 here.
        eta_factor = self._flux_per_photon
        mod_flux = flux / (eta_factor * eta_factor)
        g = 1.0

        # If requested, let the target flux value vary as a Poisson deviate
        if poisson_flux:
            # If we have both positive and negative photons, then the mix of these
            # already gives us some variation in the flux value from the variance
            # of how many are positive and how many are negative.
            # The number of negative photons varies as a binomial distribution.
            # <F-> = eta * Ntot * g
            # <F+> = (1-eta) * Ntot * g
            # <F+ - F-> = (1-2eta) * Ntot * g = flux
            # Var(F-) = eta * (1-eta) * Ntot * g^2
            # F+ = Ntot * g - F- is not an independent variable, so
            # Var(F+ - F-) = Var(Ntot*g - 2*F-)
            #              = 4 * Var(F-)
            #              = 4 * eta * (1-eta) * Ntot * g^2
            #              = 4 * eta * (1-eta) * flux
            # We want the variance to be equal to flux, so we need an extra:
            # delta Var = (1 - 4*eta + 4*eta^2) * flux
            #           = (1-2eta)^2 * flux
            absflux = abs(flux)
            mean = eta_factor * eta_factor * absflux
            pd = PoissonDeviate(rng, mean)
            pd_val = pd() - mean + absflux
            ratio = pd_val / absflux
            g *= ratio
            mod_flux *= ratio

        if n_photons == 0.0:
            n_photons = abs(mod_flux)
            if max_extra_noise > 0.0:
                gfactor = 1.0 + max_extra_noise / abs(self.max_sb)
                n_photons /= gfactor
                g *= gfactor

        # Make n_photons an integer.
        iN = int(n_photons + 0.5)

        return iN, g

    @_wraps(
        _galsim.GSObject.makePhot,
        lax_description="The JAX-GalSim version of `makePhot` does not support the deprecated surface_ops argument.",
    )
    def makePhot(
        self,
        n_photons=0,
        rng=None,
        max_extra_noise=0.0,
        poisson_flux=None,
        photon_ops=(),
        local_wcs=None,
        surface_ops=None,
    ):
        if surface_ops is not None:
            from .deprecated import depr

            depr("surface_ops", 2.3, "photon_ops")
            photon_ops = surface_ops

        # Make sure the type of n_photons is correct and has a valid value:
        if not n_photons >= 0.0:
            raise GalSimRangeError("Invalid n_photons < 0.", n_photons, 0.0, None)

        if poisson_flux is None:
            # If n_photons is given, poisson_flux = False
            poisson_flux = n_photons == 0.0

        # Check that either n_photons is set to something or flux is set to something
        if n_photons == 0.0 and self.flux == 1.0:
            galsim_warn(
                "Warning: drawImage for object with flux == 1, area == 1, and "
                "exptime == 1, but n_photons == 0.  This will only shoot a single photon."
            )

        Ntot, g = self._calculate_nphotons(
            n_photons, poisson_flux, max_extra_noise, rng
        )

        try:
            photons = self.shoot(Ntot, rng)
        except (GalSimError, NotImplementedError) as e:
            raise GalSimNotImplementedError(
                "Unable to draw this GSObject with photon shooting.  Perhaps it "
                "is a Deconvolve or is a compound including one or more "
                "Deconvolve objects.\nOriginal error: %r" % (e)
            )

        if g != 1.0:
            photons.scaleFlux(g)

        for op in photon_ops:
            op.applyTo(photons, local_wcs, rng)

        return photons

    @_wraps(
        _galsim.GSObject.drawPhot,
        lax_description="The JAX-GalSim version of `drawPhot` does not support the deprecated surface_ops argument.",
    )
    def drawPhot(
        self,
        image,
        gain=1.0,
        add_to_image=False,
        n_photons=0,
        rng=None,
        max_extra_noise=0.0,
        poisson_flux=None,
        sensor=None,
        photon_ops=(),
        maxN=None,
        orig_center=PositionI(0, 0),
        local_wcs=None,
    ):
        # Make sure the type of n_photons is correct and has a valid value:
        if not n_photons >= 0.0:
            raise GalSimRangeError("Invalid n_photons < 0.", n_photons, 0.0, None)

        if poisson_flux is None:
            # If n_photons is given, poisson_flux = False
            poisson_flux = n_photons == 0.0

        # Check that either n_photons is set to something or flux is set to something
        if n_photons == 0.0 and self.flux == 1.0:
            galsim_warn(
                "Warning: drawImage for object with flux == 1, area == 1, and "
                "exptime == 1, but n_photons == 0.  This will only shoot a single photon."
            )

        # Make sure the image is set up to have unit pixel scale and centered at 0,0.
        if image.wcs is None or not image.wcs._isPixelScale:
            raise GalSimValueError(
                "drawPhot requires an image with a PixelScale wcs", image
            )

        if sensor is None:
            sensor = Sensor()
        elif not isinstance(sensor, Sensor):
            raise TypeError("The sensor provided is not a Sensor instance")

        Ntot, g = self._calculate_nphotons(
            n_photons, poisson_flux, max_extra_noise, rng
        )

        if gain != 1.0:
            g /= gain

        # total flux falling inside image bounds, this will be returned on exit.
        added_flux = 0.0

        if maxN is None:
            maxN = Ntot

        if not add_to_image:
            image.setZero()

        # Nleft is the number of photons remaining to shoot.
        Nleft = Ntot
        photons = None  # Just in case Nleft is already 0.
        resume = False
        while Nleft > 0:
            # Shoot at most maxN at a time
            thisN = min(maxN, Nleft)

            try:
                photons = self.shoot(thisN, rng)
            except (GalSimError, NotImplementedError) as e:
                raise GalSimNotImplementedError(
                    "Unable to draw this GSObject with photon shooting.  Perhaps it "
                    "is a Deconvolve or is a compound including one or more "
                    "Deconvolve objects.\nOriginal error: %r" % (e)
                )

            if g != 1.0 or thisN != Ntot:
                photons.scaleFlux(g * thisN / Ntot)

            if image.scale != 1.0:
                photons.scaleXY(
                    1.0 / image.scale
                )  # Convert x,y to image coords if necessary

            for op in photon_ops:
                op.applyTo(photons, local_wcs, rng)

            if image.dtype in (np.float32, np.float64):
                added_flux += sensor.accumulate(
                    photons, image, orig_center, resume=resume
                )
                resume = (
                    True  # Resume from this point if there are any further iterations.
                )
            else:
                # Need a temporary
                from jax_galsim.image import ImageD

                im1 = ImageD(bounds=image.bounds)
                added_flux += sensor.accumulate(photons, im1, orig_center)
                image += im1

            Nleft -= thisN

        return added_flux, photons

    @_wraps(_galsim.GSObject.shoot)
    def shoot(self, n_photons, rng=None):
        photons = pa.PhotonArray(n_photons)
        if n_photons == 0:
            # It's ok to shoot 0, but downstream can have problems with it, so just stop now.
            return photons
        if rng is None:
            rng = BaseDeviate()

        self._shoot(photons, rng)
        return photons

    @_wraps(_galsim.GSObject._shoot)
    def _shoot(self, photons, rng):
        raise NotImplementedError(
            "%s does not implement shoot" % self.__class__.__name__
        )

    @_wraps(_galsim.GSObject.applyTo)
    def applyTo(self, photon_array, local_wcs=None, rng=None):
        p1 = pa.PhotonArray(len(photon_array))
        if photon_array.hasAllocatedWavelengths():
            p1._wave = photon_array._wave
        if photon_array.hasAllocatedPupil():
            p1._pupil_u = photon_array._pupil_u
            p1._pupil_v = photon_array._pupil_v
        if photon_array.hasAllocatedTimes():
            p1._time = photon_array._time
        obj = local_wcs.toImage(self) if local_wcs is not None else self
        obj._shoot(p1, rng)
        photon_array.convolve(p1, rng)

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
