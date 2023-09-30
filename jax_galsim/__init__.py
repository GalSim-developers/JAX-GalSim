from galsim.angle import arcmin, arcsec, degrees, hours, radians

# Inherit all Exception and Warning classes from galsim
from galsim.errors import (
    GalSimBoundsError,
    GalSimConfigError,
    GalSimConfigValueError,
    GalSimDeprecationWarning,
    GalSimError,
    GalSimFFTSizeError,
    GalSimHSMError,
    GalSimImmutableError,
    GalSimIncompatibleValuesError,
    GalSimIndexError,
    GalSimKeyError,
    GalSimNotImplementedError,
    GalSimRangeError,
    GalSimSEDError,
    GalSimUndefinedBoundsError,
    GalSimValueError,
    GalSimWarning,
)

# Basic building blocks
from .bounds import Bounds, BoundsD, BoundsI
from .gsparams import GSParams
from .position import Position, PositionD, PositionI

# Image
from .image import (
    Image,
    ImageCD,
    ImageCF,
    ImageD,
    ImageF,
    ImageI,
    ImageS,
    ImageUI,
    ImageUS,
)

# GSObject
from .exponential import Exponential
from .gaussian import Gaussian
from .box import Box, Pixel
from .gsobject import GSObject

# Interpolation
from .moffat import Moffat

from .sum import Add, Sum
from .transform import Transform, Transformation
from .convolve import Convolve, Convolution

# WCS
from .wcs import (
    AffineTransform,
    JacobianWCS,
    OffsetWCS,
    PixelScale,
    ShearWCS,
    OffsetShearWCS,
)
from .fits import FitsHeader

# Shear
from .shear import Shear, _Shear
from . import fits
