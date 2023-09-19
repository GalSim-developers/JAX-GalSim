from contextlib import contextmanager, ExitStack
import jax
from jax._src.numpy.util import _wraps
import jax.numpy as jnp
import numpy as np

import galsim as _galsim
from galsim.fits import writeFile, FitsHeader, closeHDUList, readFile  # noqa: F401
from galsim.utilities import galsim_warn

from jax_galsim.image import Image


# We wrap the galsim FITS read functions to return jax_galsim Image objects.


def _maybe_convert_and_warn(image):
    if image.array.dtype.type not in Image.valid_dtypes:
        galsim_warn(
            "The dtype of the input image is not supported by jax_galsim. "
            "Converting to float64."
        )
        _image = image.view(dtype=jnp.float64)
        if hasattr(image, "header"):
            _image.header = image.header
        return _image
    else:
        return image


@_wraps(_galsim.fits.read)
def read(*args, **kwargs):
    gsimage = _galsim.fits.read(*args, **kwargs)
    # galsim tests the dtypes against its Image class, so we need to test again here
    return _maybe_convert_and_warn(Image.from_galsim(gsimage))


@_wraps(_galsim.fits.readMulti)
def readMulti(*args, **kwargs):
    gsimage_list = _galsim.fits.readMulti(*args, **kwargs)
    return [
        _maybe_convert_and_warn(Image.from_galsim(gsimage)) for gsimage in gsimage_list
    ]


@_wraps(_galsim.fits.readCube)
def readCube(*args, **kwargs):
    gsimage_list = _galsim.fits.readCube(*args, **kwargs)
    return [
        _maybe_convert_and_warn(Image.from_galsim(gsimage)) for gsimage in gsimage_list
    ]


# We wrap the galsim FITS write functions to accept jax_galsim Image objects.


@contextmanager
def _image_as_numpy(image):
    if isinstance(image, Image):
        try:
            orig_array = image._array
            # convert to numpy so astropy doesn't complain
            image._array = np.array(image.array, dtype=orig_array.dtype)
            # some of these check for Image instances, so we hackily set the class
            # on the way in
            old_class = image.__class__
            image.__class__ = _galsim.Image
            yield image
        finally:
            image.__class__ = old_class
            image._array = orig_array
    else:
        try:
            yield np.array(image, dtype=image.dtype)
        finally:
            pass


@_wraps(_galsim.fits.read)
def write(*args, **kwargs):
    if len(args) >= 1 and isinstance(args[0], Image):
        with _image_as_numpy(args[0]) as image:
            _galsim.fits.write(image, *args[1:], **kwargs)
    else:
        _galsim.fits.write(*args, **kwargs)


@_wraps(_galsim.fits.writeMulti)
def writeMulti(*args, **kwargs):
    if len(args) >= 1:
        with ExitStack() as stack:
            gsimage_list = [
                stack.enter_context(_image_as_numpy(image))
                if isinstance(image, Image)
                else image
                for image in args[0]
            ]
            _galsim.fits.writeMulti(gsimage_list, *args[1:], **kwargs)
    else:
        _galsim.fits.writeMulti(*args, **kwargs)


@_wraps(_galsim.fits.writeCube)
def writeCube(*args, **kwargs):
    if len(args) >= 1:
        with ExitStack() as stack:
            if isinstance(args[0], list):
                gsimage_list = [
                    stack.enter_context(_image_as_numpy(image))
                    if (isinstance(image, Image) or isinstance(image, jax.Array))
                    else image
                    for image in args[0]
                ]
            else:
                gsimage_list = args[0]
            _galsim.fits.writeCube(gsimage_list, *args[1:], **kwargs)
    else:
        _galsim.fits.writeCube(*args, **kwargs)


Image.write = write
