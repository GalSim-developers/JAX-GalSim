import galsim as _galsim
import jax.numpy as jnp
import numpy as np
from jax._src.numpy.util import _wraps
from jax.tree_util import register_pytree_node_class

from jax_galsim.bounds import BoundsD, BoundsI
from jax_galsim.position import PositionI
from jax_galsim.utilities import parse_pos_args
from jax_galsim.wcs import BaseWCS, PixelScale


@_wraps(
    _galsim.Image,
    lax_description="""
    Contrary to GalSim native Image, this implementation does not support
    sharing of the underlying numpy array between different Images or Views.
    This is due to the fact that in JAX numpy arrays are immutable, so any
    operation applied to this Image will create a new jnp.ndarray.

    In particular the following methods will create a copy of the Image:
        - Image.view()
        - Image.subImage()
""",
)
@register_pytree_node_class
class Image(object):
    _alias_dtypes = {
        int: jnp.int32,  # So that user gets what they would expect
        float: jnp.float64,  # if using dtype=int or float or complex
        complex: jnp.complex128,
        jnp.int64: jnp.int32,  # Not equivalent, but will convert
        np.uint16: jnp.uint16,
        np.uint32: jnp.uint32,
        np.int16: jnp.int16,
        np.int32: jnp.int32,
        np.float32: jnp.float32,
        np.float64: jnp.float64,
        np.complex64: jnp.complex64,
        np.complex128: jnp.complex128,
    }
    valid_dtypes = _alias_dtypes.keys()

    def __init__(self, *args, dtype=jnp.float32, **kwargs):
        # Parse the args, kwargs
        ncol = None
        nrow = None
        bounds = None
        array = None
        image = None
        if len(args) > 2:
            raise TypeError("Error, too many unnamed arguments to Image constructor")
        elif len(args) == 2:
            ncol = args[0]
            nrow = args[1]
            xmin = kwargs.pop("xmin", 1)
            ymin = kwargs.pop("ymin", 1)
        elif len(args) == 1:
            if isinstance(args[0], np.ndarray):
                array = jnp.array(args[0])
                array, xmin, ymin = self._get_xmin_ymin(array, kwargs)
            elif isinstance(args[0], jnp.ndarray):
                array = args[0]
                array, xmin, ymin = self._get_xmin_ymin(array, kwargs)
            elif isinstance(args[0], (list, tuple)):
                array = jnp.array(args[0])
                array, xmin, ymin = self._get_xmin_ymin(array, kwargs)
            elif isinstance(args[0], BoundsI):
                bounds = args[0]
            elif isinstance(args[0], Image):
                image = args[0]
            else:
                raise TypeError("Unable to parse %s as an array, bounds, or image." % args[0])
        else:
            if "array" in kwargs:
                array = kwargs.pop("array")
                check_bounds = kwargs.pop("check_bounds", True)
                array, xmin, ymin = self._get_xmin_ymin(array, kwargs, check_bounds=check_bounds)
            elif "bounds" in kwargs:
                bounds = kwargs.pop("bounds")
            elif "image" in kwargs:
                image = kwargs.pop("image")
            else:
                ncol = kwargs.pop("ncol", None)
                nrow = kwargs.pop("nrow", None)
                xmin = kwargs.pop("xmin", 1)
                ymin = kwargs.pop("ymin", 1)

        # Pop off the other valid kwargs:
        dtype = kwargs.pop("dtype", None)
        init_value = kwargs.pop("init_value", None)
        scale = kwargs.pop("scale", None)
        wcs = kwargs.pop("wcs", None)

        # Check that we got them all
        if kwargs:
            if "copy" in kwargs.keys():
                raise TypeError(
                    "'copy' is not a valid keyword argument for the JAX-GalSim version"
                    "of the Image constructor"
                )
            if "make_const" in kwargs.keys():
                raise TypeError(
                    "'make_const' is not a valid keyword argument for the JAX-GalSim"
                    "version of the Image constructor"
                )
            raise TypeError("Image constructor got unexpected keyword arguments: %s", kwargs)

        # Figure out what dtype we want:
        dtype = Image._alias_dtypes.get(dtype, dtype)
        if dtype is None:
            raise TypeError("dtype cannot be None.")
        if dtype is not None and dtype not in Image.valid_dtypes:
            raise _galsim.GalSimValueError("Invlid dtype.", dtype, Image.valid_dtypes)
        if array is not None:
            array = array.astype(dtype)
            # Be careful here: we have to watch out for little-endian / big-endian issues.
            # The path of least resistance is to check whether the array.dtype is equal to the
            # native one (using the dtype.isnative flag), and if not, make a new array that has a
            # type equal to the same one but with the appropriate endian-ness.
            if not array.dtype.isnative:
                array = array.astype(array.dtype.newbyteorder("="))
            self._dtype = array.dtype.type
        elif dtype is not None:
            self._dtype = dtype
        else:
            self._dtype = jnp.float32

        # Construct the image attribute
        if ncol is not None or nrow is not None:
            # TODO: vmapping/jitting with ncol/nrow
            if ncol is None or nrow is None:
                raise _galsim.GalSimIncompatibleValuesError(
                    "Both nrow and ncol must be provided", ncol=ncol, nrow=nrow
                )
            if ncol != int(ncol) or nrow != int(nrow):
                raise TypeError("nrow, ncol must be integers")
            ncol = int(ncol)
            nrow = int(nrow)
            self._array = self._make_empty(shape=(nrow, ncol), dtype=self._dtype)
            self._bounds = BoundsI(xmin, xmin + ncol - 1, ymin, ymin + nrow - 1)
            if init_value:
                self._array = self._array + init_value
        elif bounds is not None:
            if not isinstance(bounds, BoundsI):
                raise TypeError("bounds must be a galsim.BoundsI instance")
            self._array = self._make_empty(bounds.numpyShape(), dtype=self._dtype)
            self._bounds = bounds
            if init_value:
                self._array = self._array + init_value
        elif array is not None:
            self._array = array.view()
            nrow, ncol = array.shape
            self._bounds = BoundsI(xmin, xmin + ncol - 1, ymin, ymin + nrow - 1)
            if init_value is not None:
                raise _galsim.GalSimIncompatibleValuesError(
                    "Cannot specify init_value with array",
                    init_value=init_value,
                    array=array,
                )
        elif image is not None:
            if not isinstance(image, Image):
                raise TypeError("image must be an Image")
            if init_value is not None:
                raise _galsim.GalSimIncompatibleValuesError(
                    "Cannot specify init_value with image",
                    init_value=init_value,
                    image=image,
                )
            if wcs is None and scale is None:
                wcs = image.wcs
            self._bounds = image.bounds
            if dtype is None:
                self._dtype = image.dtype
            else:
                # Allow dtype to force a retyping of the provided image
                # e.g. im = ImageF(...)
                #      im2 = ImageD(im)
                self._dtype = dtype
            self._array = image.array.astype(self._dtype)
        else:
            # TODO: remove this possiblity of creating an empty image.
            self._array = jnp.zeros(shape=(1, 1), dtype=self._dtype)
            self._bounds = BoundsI()
            if init_value is not None:
                raise _galsim.GalSimIncompatibleValuesError(
                    "Cannot specify init_value without setting an initial size",
                    init_value=init_value,
                    ncol=ncol,
                    nrow=nrow,
                    bounds=bounds,
                )

        # Construct the wcs attribute
        if scale is not None:
            if wcs is not None:
                raise _galsim.GalSimIncompatibleValuesError(
                    "Cannot provide both scale and wcs to Image constructor",
                    wcs=wcs,
                    scale=scale,
                )
            self.wcs = PixelScale(float(scale))
        else:
            if wcs is not None and not isinstance(wcs, BaseWCS):
                raise TypeError("wcs parameters must be a galsim.BaseWCS instance")
            self.wcs = wcs

    @staticmethod
    def _get_xmin_ymin(array, kwargs, check_bounds=True):
        """A helper function for parsing xmin, ymin, bounds options with a given array."""
        if not isinstance(array, (np.ndarray, jnp.ndarray)):
            raise TypeError("array must be a ndarray instance")
        xmin = kwargs.pop("xmin", 1)
        ymin = kwargs.pop("ymin", 1)
        if "bounds" in kwargs:
            b = kwargs.pop("bounds")
            if check_bounds:
                if b.xmax - b.xmin + 1 != array.shape[1]:
                    raise _galsim.GalSimIncompatibleValuesError(
                        "Shape of array is inconsistent with provided bounds",
                        array=array,
                        bounds=b,
                    )
                if b.ymax - b.ymin + 1 != array.shape[0]:
                    raise _galsim.GalSimIncompatibleValuesError(
                        "Shape of array is inconsistent with provided bounds",
                        array=array,
                        bounds=b,
                    )
            if b.isDefined():
                xmin = b.xmin
                ymin = b.ymin
            else:
                # Indication that array is formally undefined, even though provided.
                if "dtype" not in kwargs:
                    kwargs["dtype"] = array.dtype.type
                array = None
                xmin = None
                ymin = None
        elif array.shape[1] == 0:
            # Another way to indicate that we don't have a defined image.
            if "dtype" not in kwargs:
                kwargs["dtype"] = array.dtype.type
            array = None
            xmin = None
            ymin = None
        return array, xmin, ymin

    def __repr__(self):
        s = "galsim.Image(bounds=%r" % self.bounds
        if self.bounds.isDefined():
            s += ", array=\n%r" % np.array(self.array)
        s += ", wcs=%r" % self.wcs
        s += ")"
        return s

    def __str__(self):
        # Get the type name without the <type '...'> part.
        t = str(self.dtype).split("'")[1]
        if self.wcs is not None and self.wcs._isPixelScale:
            return "galsim.Image(bounds=%s, scale=%s, dtype=%s)" % (
                self.bounds,
                self.scale,
                t,
            )
        else:
            return "galsim.Image(bounds=%s, wcs=%s, dtype=%s)" % (
                self.bounds,
                self.wcs,
                t,
            )

    # Read-only attributes:
    @property
    def dtype(self):
        """The dtype of the underlying numpy array."""
        return self._dtype

    @property
    def bounds(self):
        """The bounds of the `Image`."""
        return self._bounds

    @property
    def array(self):
        """The underlying numpy array."""
        return self._array

    @property
    def isconst(self):
        """Whether the `Image` is constant.  I.e. modifying its values is an error."""
        return True

    @property
    def iscomplex(self):
        """Whether the `Image` values are complex."""
        return self._array.dtype.kind == "c"

    @property
    def isinteger(self):
        """Whether the `Image` values are integral."""
        return self._array.dtype.kind in ("i", "u")

    # Allow scale to work as a PixelScale wcs.
    @property
    def scale(self):
        """The pixel scale of the `Image`.  Only valid if the wcs is a `PixelScale`.

        If the WCS is either not set (i.e. it is ``None``) or it is a `PixelScale`, then
        it is permissible to change the scale with::

            >>> image.scale = new_pixel_scale
        """
        try:
            return self.wcs.scale
        except Exception:
            if self.wcs:
                raise _galsim.GalSimError(
                    "image.wcs is not a simple PixelScale; scale is undefined."
                )
            else:
                return None

    @scale.setter
    def scale(self, value):
        if self.wcs is not None and not self.wcs._isPixelScale:
            raise _galsim.GalSimError("image.wcs is not a simple PixelScale; scale is undefined.")
        else:
            self.wcs = PixelScale(value)

    # Convenience functions
    @property
    def xmin(self):
        """Alias for self.bounds.xmin."""
        return self._bounds.xmin

    @property
    def xmax(self):
        """Alias for self.bounds.xmax."""
        return self._bounds.xmax

    @property
    def ymin(self):
        """Alias for self.bounds.ymin."""
        return self._bounds.ymin

    @property
    def ymax(self):
        """Alias for self.bounds.ymax."""
        return self._bounds.ymax

    @property
    def outer_bounds(self):
        """The bounds of the outer edge of the pixels.

        Equivalent to galsim.BoundsD(im.xmin-0.5, im.xmax+0.5, im.ymin-0.5, im.ymax+0.5)
        """
        return BoundsD(self.xmin - 0.5, self.xmax + 0.5, self.ymin - 0.5, self.ymax + 0.5)

    # real, imag for everything, even real images.
    @property
    def real(self):
        """Return the real part of an image.

        This is a property, not a function.  So write ``im.real``, not ``im.real()``.

        This works for real or complex.  For real images, it acts the same as `view`.
        """
        return self.__class__(self.array.real, bounds=self.bounds, wcs=self.wcs)

    @property
    def imag(self):
        """Return the imaginary part of an image.

        This is a property, not a function.  So write ``im.imag``, not ``im.imag()``.

        This works for real or complex.  For real images, the returned array is read-only and
        all elements are 0.
        """
        return self.__class__(self.array.imag, bounds=self.bounds, wcs=self.wcs)

    @property
    def conjugate(self):
        """Return the complex conjugate of an image.

        This works for real or complex.  For real images, it acts the same as `view`.

        Note that for complex images, this is not a conjugate view into the original image.
        So changing the original image does not change the conjugate (or vice versa).
        """
        return self.__class__(self.array.conjugate(), bounds=self.bounds, wcs=self.wcs)

    def copy(self):
        """Make a copy of the `Image`"""
        return self.__class__(self.array.copy(), bounds=self.bounds, wcs=self.wcs)

    def get_pixel_centers(self):
        """A convenience function to get the x and y values at the centers of the image pixels.

        Returns:
            (x, y), each of which is a numpy array the same shape as ``self.array``
        """
        x, y = jnp.meshgrid(
            jnp.arange(self.array.shape[1], dtype=float),
            jnp.arange(self.array.shape[0], dtype=float),
        )
        x += self.bounds.xmin
        y += self.bounds.ymin
        return x, y

    def _make_empty(self, shape, dtype):
        """Helper function to make an empty numpy array of the given shape."""
        return jnp.zeros(shape=shape, dtype=dtype)

    def resize(self, bounds, wcs=None):
        """Resize the image to have a new bounds (must be a `BoundsI` instance)

        Note that the resized image will have uninitialized data.  If you want to preserve
        the existing data values, you should either use `subImage` (if you want a smaller
        portion of the current `Image`) or make a new `Image` and copy over the current values
        into a portion of the new image (if you are resizing to a larger `Image`).

        Parameters:
            bounds:     The new bounds to resize to.
            wcs:        If provided, also update the wcs to the given value. [default: None,
                        which means keep the existing wcs]
        """
        if not isinstance(bounds, BoundsI):
            raise TypeError("bounds must be a galsim.BoundsI instance")
        self._array = self._make_empty(shape=bounds.numpyShape(), dtype=self.dtype)
        self._bounds = bounds
        if wcs is not None:
            self.wcs = wcs

    def subImage(self, bounds):
        """Return a view of a portion of the full image

        This is equivalent to self[bounds]
        """
        if not isinstance(bounds, BoundsI):
            raise TypeError("bounds must be a galsim.BoundsI instance")
        if not self.bounds.isDefined():
            raise _galsim.GalSimUndefinedBoundsError(
                "Attempt to access subImage of undefined image"
            )
        if not self.bounds.includes(bounds):
            raise _galsim.GalSimBoundsError(
                "Attempt to access subImage not (fully) in image", bounds, self.bounds
            )
        i1 = bounds.ymin - self.ymin
        i2 = bounds.ymax - self.ymin + 1
        j1 = bounds.xmin - self.xmin
        j2 = bounds.xmax - self.xmin + 1
        subarray = self.array[i1:i2, j1:j2]
        # NB. The wcs is still accurate, since the sub-image uses the same (x,y) values
        # as the original image did for those pixels.  It's only once you recenter or
        # reorigin that you need to update the wcs.  So that's taken care of in im.shift.
        return self.__class__(subarray, bounds=bounds, wcs=self.wcs)

    def setSubImage(self, bounds, rhs):
        """Set a portion of the full image to the values in another image

        This is equivalent to self[bounds] = rhs
        """
        if not isinstance(bounds, BoundsI):
            raise TypeError("bounds must be a galsim.BoundsI instance")
        if not self.bounds.isDefined():
            raise _galsim.GalSimUndefinedBoundsError(
                "Attempt to access values of an undefined image"
            )
        if not self.bounds.includes(bounds):
            raise _galsim.GalSimBoundsError(
                "Attempt to access subImage not (fully) in image", bounds, self.bounds
            )
        if not isinstance(rhs, Image):
            raise TypeError("Trying to copyFrom a non-image")
        if bounds.numpyShape() != rhs.bounds.numpyShape():
            raise _galsim.GalSimIncompatibleValuesError(
                "Trying to copy images that are not the same shape",
                self_image=self,
                rhs=rhs,
            )
        i1 = bounds.ymin - self.ymin
        i2 = bounds.ymax - self.ymin + 1
        j1 = bounds.xmin - self.xmin
        j2 = bounds.xmax - self.xmin + 1
        self._array = self._array.at[i1:i2, j1:j2].set(rhs.array)

    def __getitem__(self, *args):
        """Return either a subimage or a single pixel value.

        For example,::

            >>> subimage = im[galsim.BoundsI(3,7,3,7)]
            >>> value = im[galsim.PositionI(5,5)]
            >>> value = im[5,5]
        """
        if len(args) == 1:
            if isinstance(args[0], BoundsI):
                return self.subImage(*args)
            elif isinstance(args[0], PositionI):
                return self(*args)
            elif isinstance(args[0], tuple):
                return self.getValue(*args[0])
            else:
                raise TypeError("image[index] only accepts BoundsI or PositionI for the index")
        elif len(args) == 2:
            return self(*args)
        else:
            raise TypeError("image[..] requires either 1 or 2 args")

    def __setitem__(self, *args):
        """Set either a subimage or a single pixel to new values.

        For example,::

            >>> im[galsim.BoundsI(3,7,3,7)] = im2
            >>> im[galsim.PositionI(5,5)] = 17.
            >>> im[5,5] = 17.
        """
        if len(args) == 2:
            if isinstance(args[0], BoundsI):
                self.setSubImage(*args)
            elif isinstance(args[0], PositionI):
                self.setValue(*args)
            elif isinstance(args[0], tuple):
                self.setValue(*args)
            else:
                raise TypeError("image[index] only accepts BoundsI or PositionI for the index")
        elif len(args) == 3:
            return self.setValue(*args)
        else:
            raise TypeError("image[..] requires either 1 or 2 args")

    @_wraps(_galsim.Image.calculate_fft)
    def calculate_fft(self):
        if not self.bounds.isDefined():
            raise _galsim.GalSimUndefinedBoundsError(
                "calculate_fft requires that the image have defined bounds."
            )
        if self.wcs is None:
            raise _galsim.GalSimError("calculate_fft requires that the scale be set.")
        if not self.wcs._isPixelScale:
            raise _galsim.GalSimError("calculate_fft requires that the image has a PixelScale wcs.")

        No2 = jnp.maximum(
            -self.bounds.xmin,
            self.bounds.xmax + 1,
            -self.bounds.ymin,
            self.bounds.ymax + 1,
        )

        full_bounds = BoundsI(-No2, No2 - 1, -No2, No2 - 1)
        if self.bounds == full_bounds:
            # Then the image is already in the shape we need.
            ximage = self
        else:
            # Then we pad out with zeros
            ximage = Image(full_bounds, dtype=self.dtype, init_value=0)
            ximage[self.bounds] = self[self.bounds]

        dx = self.scale
        # dk = 2pi / (N dk)
        dk = jnp.pi / (No2 * dx)

        out = Image(BoundsI(0, No2, -No2, No2 - 1), dtype=np.complex128, scale=dk)
        out._image = jnp.fft.rfft2(ximage._image)

        out *= dx * dx
        out.setOrigin(0, -No2)
        return out

    @_wraps(_galsim.Image.calculate_inverse_fft)
    def calculate_inverse_fft(self):
        if not self.bounds.isDefined():
            raise _galsim.GalSimUndefinedBoundsError(
                "calculate_fft requires that the image have defined bounds."
            )
        if self.wcs is None:
            raise _galsim.GalSimError("calculate_inverse_fft requires that the scale be set.")
        if not self.wcs._isPixelScale:
            raise _galsim.GalSimError(
                "calculate_inverse_fft requires that the image has a PixelScale wcs."
            )
        if not self.bounds.includes(0, 0):
            raise _galsim.GalSimBoundsError(
                "calculate_inverse_fft requires that the image includes (0,0)",
                PositionI(0, 0),
                self.bounds,
            )

        No2 = jnp.maximum(self.bounds.xmax, -self.bounds.ymin, self.bounds.ymax)

        target_bounds = BoundsI(0, No2, -No2, No2 - 1)
        if self.bounds == target_bounds:
            # Then the image is already in the shape we need.
            kimage = self
        else:
            # Then we can pad out with zeros and wrap to get this in the form we need.
            full_bounds = BoundsI(0, No2, -No2, No2)
            kimage = Image(full_bounds, dtype=self.dtype, init_value=0)
            posx_bounds = BoundsI(0, self.bounds.xmax, self.bounds.ymin, self.bounds.ymax)
            kimage[posx_bounds] = self[posx_bounds]
            kimage = kimage.wrap(target_bounds, hermitian="x")

        dk = self.scale
        # dx = 2pi / (N dk)
        dx = jnp.pi / (No2 * dk)

        # For the inverse, we need a bit of extra space for the fft.
        out_extra = Image(BoundsI(-No2, No2 + 1, -No2, No2 - 1), dtype=float, scale=dx)
        out_extra._image = jnp.fft.irfft2(kimage._image)
        # Now cut off the bit we don't need.
        out = out_extra.subImage(BoundsI(-No2, No2 - 1, -No2, No2 - 1))
        out *= (dk * No2 / jnp.pi) ** 2
        out.setCenter(0, 0)
        return out

    @classmethod
    def good_fft_size(cls, input_size):
        """Round the given input size up to the next higher power of 2 or 3 times a power of 2.

        This rounds up to the next higher value that is either 2^k or 3*2^k.  If you are
        going to be performing FFTs on an image, these will tend to be faster at performing
        the FFT.
        """
        # Reference from GalSim C++
        # https://github.com/GalSim-developers/GalSim/blob/ece3bd32c1ae6ed771f2b489c5ab1b25729e0ea4/src/Image.cpp#L1009
        input_size = int(input_size)
        if input <= 2:
            return 2
        # Reduce slightly to eliminate potential rounding errors:
        insize = (1.0 - 1.0e-5) * input
        log2n = np.log(2.0) * np.ceil(np.log(insize) / np.log(2.0))
        log2n3 = np.log(3.0) + np.log(2.0) * np.ceil((np.log(insize) - np.log(3.0)) / np.log(2.0))
        log2n3 = np.max(log2n3, np.log(6.0))  # must be even number
        Nk = int(np.ceil(np.exp(np.min(log2n, log2n3)) - 1.0e-5))
        return Nk

    def copyFrom(self, rhs):
        """Copy the contents of another image"""
        if not isinstance(rhs, Image):
            raise TypeError("Trying to copyFrom a non-image")
        if self.bounds.numpyShape() != rhs.bounds.numpyShape():
            raise _galsim.GalSimIncompatibleValuesError(
                "Trying to copy images that are not the same shape",
                self_image=self,
                rhs=rhs,
            )
        self._array = rhs._array

    @_wraps(
        _galsim.Image.view,
        lax_description="Contrary to GalSim, this will create a copy of the orginal image.",
    )
    def view(
        self,
        scale=None,
        wcs=None,
        origin=None,
        center=None,
        dtype=None,
        make_const=False,
        contiguous=False,
    ):
        if make_const:
            raise TypeError("'make_const' is not a valid option in JAX-GalSim. Set to `False`.")

        if origin is not None and center is not None:
            raise _galsim.GalSimIncompatibleValuesError(
                "Cannot provide both center and origin", center=center, origin=origin
            )

        if scale is not None:
            if wcs is not None:
                raise _galsim.GalSimIncompatibleValuesError(
                    "Cannot provide both scale and wcs", scale=scale, wcs=wcs
                )
            wcs = PixelScale(scale)
        elif wcs is not None:
            if not isinstance(wcs, BaseWCS):
                raise TypeError("wcs parameters must be a galsim.BaseWCS instance")
        else:
            wcs = self.wcs

        # Figure out the dtype for the return Image
        dtype = dtype if dtype else self.dtype

        # If currently empty, just return a new empty image.
        if not self.bounds.isDefined():
            return Image(wcs=wcs, dtype=dtype)

        # Recast the array type if necessary
        if dtype != self.array.dtype:
            array = self.array.astype(dtype)
        elif contiguous:
            array = np.ascontiguousarray(self.array)
        else:
            array = self.array

        # Make the return Image
        ret = self.__class__(array, bounds=self.bounds, wcs=wcs)

        # Update the origin if requested
        if origin is not None:
            ret.setOrigin(origin)
        elif center is not None:
            ret.setCenter(center)

        return ret

    @_wraps(_galsim.Image.shift)
    def shift(self, *args, **kwargs):
        delta = parse_pos_args(args, kwargs, "dx", "dy", integer=True)
        self._shift(delta)

    def _shift(self, delta):
        """Equivalent to `shift`, but without some of the sanity checks and ``delta`` must
        be a `PositionI` instance.

        Parameters:
            delta:  The amount to shift as a `PositionI`.
        """
        # The parse_pos_args function is a bit slow, so go directly to this point when we
        # call shift from setCenter or setOrigin.
        if delta.x != 0 or delta.y != 0:
            self._bounds = self._bounds.shift(delta)
            if self.wcs is not None:
                self.wcs = self.wcs.shiftOrigin(delta)

    @_wraps(_galsim.Image.setCenter)
    def setCenter(self, *args, **kwargs):
        cen = parse_pos_args(args, kwargs, "xcen", "ycen", integer=True)
        self._shift(cen - self.center)

    @_wraps(_galsim.Image.setOrigin)
    def setOrigin(self, *args, **kwargs):
        origin = parse_pos_args(args, kwargs, "x0", "y0", integer=True)
        self._shift(origin - self.origin)

    @property
    @_wraps(_galsim.Image.center)
    def center(self):
        return self.bounds.center

    @property
    @_wraps(_galsim.Image.true_center)
    def true_center(self):
        return self.bounds.true_center

    @property
    @_wraps(_galsim.Image.origin)
    def origin(self):
        return self.bounds.origin

    def __call__(self, *args, **kwargs):
        """Get the pixel value at given position

        The arguments here may be either (x, y) or a PositionI instance.
        Or you can provide x, y as named kwargs.
        """
        pos = parse_pos_args(args, kwargs, "x", "y", integer=True)
        return self.getValue(pos.x, pos.y)

    @_wraps(_galsim.Image.getValue)
    def getValue(self, x, y):
        if not self.bounds.isDefined():
            raise _galsim.GalSimUndefinedBoundsError(
                "Attempt to access values of an undefined image"
            )
        if not self.bounds.includes(x, y):
            raise _galsim.GalSimBoundsError(
                "Attempt to access position not in bounds of image.",
                PositionI(x, y),
                self.bounds,
            )
        return self._getValue(x, y)

    def _getValue(self, x, y):
        """Equivalent to `getValue`, except there are no checks that the values fall
        within the bounds of the image.
        """
        return self.array[y - self.ymin, x - self.xmin]

    @_wraps(_galsim.Image.setValue)
    def setValue(self, *args, **kwargs):
        if not self.bounds.isDefined():
            raise _galsim.GalSimUndefinedBoundsError("Attempt to set value of an undefined image")
        pos, value = parse_pos_args(args, kwargs, "x", "y", integer=True, others=["value"])
        if not self.bounds.includes(pos):
            raise _galsim.GalSimBoundsError(
                "Attempt to set position not in bounds of image", pos, self.bounds
            )
        self._setValue(pos.x, pos.y, value)

    def _setValue(self, x, y, value):
        """Equivalent to `setValue` except that there are no checks that the values
        fall within the bounds of the image, and the coordinates must be given as ``x``, ``y``.

        Parameters:
            x:      The x coordinate of the pixel to set.
            y:      The y coordinate of the pixel to set.
            value:  The value to set the pixel to.
        """
        self._array = self._array.at[y - self.ymin, x - self.xmin].set(value)

    @_wraps(_galsim.Image.addValue)
    def addValue(self, *args, **kwargs):
        if not self.bounds.isDefined():
            raise _galsim.GalSimUndefinedBoundsError("Attempt to set value of an undefined image")
        pos, value = parse_pos_args(args, kwargs, "x", "y", integer=True, others=["value"])
        if not self.bounds.includes(pos):
            raise _galsim.GalSimBoundsError(
                "Attempt to set position not in bounds of image", pos, self.bounds
            )
        self._addValue(pos.x, pos.y, value)

    def _addValue(self, x, y, value):
        """Equivalent to `addValue` except that there are no checks that the values
        fall within the bounds of the image, and the coordinates must be given as ``x``, ``y``.

        Parameters:
            x:      The x coordinate of the pixel to add to.
            y:      The y coordinate of the pixel to add to.
            value:  The value to add to this pixel.
        """
        self._array = self._array.at[y - self.ymin, x - self.xmin].add(value)

    def fill(self, value):
        """Set all pixel values to the given ``value``

        Parameter:
            value:  The value to set all the pixels to.
        """
        if not self.bounds.isDefined():
            raise _galsim.GalSimUndefinedBoundsError("Attempt to set values of an undefined image")
        self._fill(value)

    def _fill(self, value):
        """Equivalent to `fill`, except that there are no checks that the bounds are defined."""
        self._array = jnp.zeros_like(self._array) + value

    def setZero(self):
        """Set all pixel values to zero."""
        self._fill(0)

    def invertSelf(self):
        """Set all pixel values to their inverse: x -> 1/x.

        Note: any pixels whose value is 0 originally are ignored.  They remain equal to 0
        on the output, rather than turning into inf.
        """
        if not self.bounds.isDefined():
            raise _galsim.GalSimUndefinedBoundsError("Attempt to set values of an undefined image")
        self._invertSelf()

    def _invertSelf(self):
        """Equivalent to `invertSelf`, except there are no checks that bounds are defined."""
        array = 1.0 / self._array
        array = array.at[jnp.isinf(array)].set(0.0)
        self._array = array.astype(self._array.dtype)

    def replaceNegative(self, replace_value=0):
        """Replace any negative values currently in the image with 0 (or some other value).

        Sometimes FFT drawing can result in tiny negative values, which may be undesirable for
        some purposes.  This method replaces those values with 0 or some other value if desired.

        Parameters:
            replace_value:  The value with which to replace any negative pixels. [default: 0]
        """
        self.array.at[self.array < 0].set(replace_value)

    def __eq__(self, other):
        # Note that numpy.array_equal can return True if the dtypes of the two arrays involved are
        # different, as long as the contents of the two arrays are logically the same.  For example:
        #
        # >>> double_array = np.arange(1024).reshape(32, 32)*np.pi
        # >>> int_array = np.arange(1024).reshape(32, 32)
        # >>> assert galsim.ImageD(int_array) == galsim.ImageF(int_array) # passes
        # >>> assert galsim.ImageD(double_array) == galsim.ImageF(double_array) # fails

        return self is other or (
            isinstance(other, Image)
            and self.bounds == other.bounds
            and self.wcs == other.wcs
            and (not self.bounds.isDefined() or jnp.array_equal(self.array, other.array))
            and self.isconst == other.isconst
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def tree_flatten(self):
        """Flatten the image into a list of values."""
        # Define the children nodes of the PyTree that need tracing
        children = (self.array, self.wcs, self.bounds)
        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        return cls(array=children[0], wcs=children[1], bounds=children[2], check_bounds=False)


# These are essentially aliases for the regular Image with the correct dtype
def ImageUS(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.uint16)"""
    kwargs["dtype"] = jnp.uint16
    return Image(*args, **kwargs)


def ImageUI(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.uint32)"""
    kwargs["dtype"] = jnp.uint32
    return Image(*args, **kwargs)


def ImageS(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.int16)"""
    kwargs["dtype"] = jnp.int16
    return Image(*args, **kwargs)


def ImageI(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.int32)"""
    kwargs["dtype"] = jnp.int32
    return Image(*args, **kwargs)


def ImageF(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.float32)"""
    kwargs["dtype"] = jnp.float32
    return Image(*args, **kwargs)


def ImageD(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.float64)"""
    kwargs["dtype"] = jnp.float64
    return Image(*args, **kwargs)


def ImageCF(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.complex64)"""
    kwargs["dtype"] = jnp.complex64
    return Image(*args, **kwargs)


def ImageCD(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.complex128)"""
    kwargs["dtype"] = jnp.complex128
    return Image(*args, **kwargs)


################################################################################################
#
# Now we have to make some modifications to the C++ layer objects.  Mostly adding some
# arithmetic functions, so they work more intuitively.
#


# Define a utility function to be used by the arithmetic functions below
def check_image_consistency(im1, im2, integer=False):
    if integer and not im1.isinteger:
        raise _galsim.GalSimValueError("Image must have integer values.", im1)
    if isinstance(im2, Image):
        if im1.array.shape != im2.array.shape:
            raise _galsim.GalSimIncompatibleValuesError(
                "Image shapes are inconsistent", im1=im1, im2=im2
            )
        if integer and not im2.isinteger:
            raise _galsim.GalSimValueError("Image must have integer values.", im2)


def Image_add(self, other):
    check_image_consistency(self, other)
    try:
        a = other.array
    except AttributeError:
        a = other
    return Image(self.array + a, bounds=self.bounds, wcs=self.wcs)


def Image_iadd(self, other):
    check_image_consistency(self, other)
    try:
        a = other.array
        dt = a.dtype
    except AttributeError:
        a = other
        dt = type(a)
    if dt == self.array.dtype:
        array = self.array + a
    else:
        array = (self.array + a).astype(self.array.dtype)
    return Image(array, bounds=self.bounds, wcs=self.wcs)


def Image_sub(self, other):
    check_image_consistency(self, other)
    try:
        a = other.array
    except AttributeError:
        a = other
    return Image(self.array - a, bounds=self.bounds, wcs=self.wcs)


def Image_rsub(self, other):
    return Image(other - self.array, bounds=self.bounds, wcs=self.wcs)


def Image_isub(self, other):
    check_image_consistency(self, other)
    try:
        a = other.array
        dt = a.dtype
    except AttributeError:
        a = other
        dt = type(a)
    if dt == self.array.dtype:
        array = self.array - a
    else:
        array = (self.array - a).astype(self.array.dtype)
    return Image(array, bounds=self.bounds, wcs=self.wcs)


def Image_mul(self, other):
    check_image_consistency(self, other)
    try:
        a = other.array
    except AttributeError:
        a = other
    return Image(self.array * a, bounds=self.bounds, wcs=self.wcs)


def Image_imul(self, other):
    check_image_consistency(self, other)
    try:
        a = other.array
        dt = a.dtype
    except AttributeError:
        a = other
        dt = type(a)
    if dt == self.array.dtype:
        array = self.array * a
    else:
        array = (self.array * a).astype(self.array.dtype)
    return Image(array, bounds=self.bounds, wcs=self.wcs)


def Image_div(self, other):
    check_image_consistency(self, other)
    try:
        a = other.array
    except AttributeError:
        a = other
    return Image(self.array / a, bounds=self.bounds, wcs=self.wcs)


def Image_rdiv(self, other):
    return Image(other / self.array, bounds=self.bounds, wcs=self.wcs)


def Image_idiv(self, other):
    check_image_consistency(self, other)
    try:
        a = other.array
        dt = a.dtype
    except AttributeError:
        a = other
        dt = type(a)
    if dt == self.array.dtype and not self.isinteger:
        # if dtype is an integer type, then numpy doesn't allow true division /= to assign
        # back to an integer array.  So for integers (or mixed types), don't use /=.
        array = self.array / a
    else:
        array = (self.array / a).astype(self.array.dtype)
    return Image(array, bounds=self.bounds, wcs=self.wcs)


def Image_floordiv(self, other):
    check_image_consistency(self, other, integer=True)
    try:
        a = other.array
    except AttributeError:
        a = other
    return Image(self.array // a, bounds=self.bounds, wcs=self.wcs)


def Image_rfloordiv(self, other):
    check_image_consistency(self, other, integer=True)
    return Image(other // self.array, bounds=self.bounds, wcs=self.wcs)


def Image_ifloordiv(self, other):
    check_image_consistency(self, other, integer=True)
    try:
        a = other.array
        dt = a.dtype
    except AttributeError:
        a = other
        dt = type(a)
    if dt == self.array.dtype:
        array = self.array // a
    else:
        array = (self.array // a).astype(self.array.dtype)
    return Image(array, bounds=self.bounds, wcs=self.wcs)


def Image_mod(self, other):
    check_image_consistency(self, other, integer=True)
    try:
        a = other.array
    except AttributeError:
        a = other
    return Image(self.array % a, bounds=self.bounds, wcs=self.wcs)


def Image_rmod(self, other):
    check_image_consistency(self, other, integer=True)
    return Image(other % self.array, bounds=self.bounds, wcs=self.wcs)


def Image_imod(self, other):
    check_image_consistency(self, other, integer=True)
    try:
        a = other.array
        dt = a.dtype
    except AttributeError:
        a = other
        dt = type(a)
    if dt == self.array.dtype:
        array = self.array % a
    else:
        array = (self.array % a).astype(self.array.dtype)
    return Image(array, bounds=self.bounds, wcs=self.wcs)


def Image_pow(self, other):
    return Image(self.array**other, bounds=self.bounds, wcs=self.wcs)


def Image_ipow(self, other):
    if not isinstance(other, int) and not isinstance(other, float):
        raise TypeError("Can only raise an image to a float or int power!")
    return Image(self.array**other, bounds=self.bounds, wcs=self.wcs)


def Image_neg(self):
    result = self.copy()
    result *= -1
    return result


# Define &, ^ and | only for integer-type images
def Image_and(self, other):
    check_image_consistency(self, other, integer=True)
    try:
        a = other.array
    except AttributeError:
        a = other
    return Image(self.array & a, bounds=self.bounds, wcs=self.wcs)


def Image_xor(self, other):
    check_image_consistency(self, other, integer=True)
    try:
        a = other.array
    except AttributeError:
        a = other
    return Image(self.array ^ a, bounds=self.bounds, wcs=self.wcs)


def Image_or(self, other):
    check_image_consistency(self, other, integer=True)
    try:
        a = other.array
    except AttributeError:
        a = other
    return Image(self.array | a, bounds=self.bounds, wcs=self.wcs)


# inject the arithmetic operators as methods of the Image class:
Image.__add__ = Image_add
Image.__radd__ = Image_add
Image.__iadd__ = Image_iadd
Image.__sub__ = Image_sub
Image.__rsub__ = Image_rsub
Image.__isub__ = Image_isub
Image.__mul__ = Image_mul
Image.__rmul__ = Image_mul
Image.__imul__ = Image_imul
Image.__div__ = Image_div
Image.__rdiv__ = Image_rdiv
Image.__truediv__ = Image_div
Image.__rtruediv__ = Image_rdiv
Image.__idiv__ = Image_idiv
Image.__itruediv__ = Image_idiv
Image.__mod__ = Image_mod
Image.__rmod__ = Image_rmod
Image.__imod__ = Image_imod
Image.__floordiv__ = Image_floordiv
Image.__rfloordiv__ = Image_rfloordiv
Image.__ifloordiv__ = Image_ifloordiv
Image.__ipow__ = Image_ipow
Image.__pow__ = Image_pow
Image.__neg__ = Image_neg
Image.__and__ = Image_and
Image.__xor__ = Image_xor
Image.__or__ = Image_or
Image.__rand__ = Image_and
Image.__rxor__ = Image_xor
Image.__ror__ = Image_or
Image.__iand__ = Image_and
Image.__ixor__ = Image_xor
Image.__ior__ = Image_or
