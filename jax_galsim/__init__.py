try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.1.dev0"

# Exception and Warning classes
from .errors import GalSimError, GalSimRangeError, GalSimValueError
from .errors import GalSimKeyError, GalSimIndexError, GalSimNotImplementedError
from .errors import GalSimBoundsError, GalSimUndefinedBoundsError, GalSimImmutableError
from .errors import GalSimIncompatibleValuesError, GalSimSEDError, GalSimHSMError
from .errors import GalSimFFTSizeError
from .errors import GalSimConfigError, GalSimConfigValueError
from .errors import GalSimWarning, GalSimDeprecationWarning

# noise
from .random import (
    BaseDeviate,
    UniformDeviate,
    GaussianDeviate,
    PoissonDeviate,
    Chi2Deviate,
    GammaDeviate,
    WeibullDeviate,
    BinomialDeviate,
)
from .noise import (
    BaseNoise,
    GaussianNoise,
    DeviateNoise,
    PoissonNoise,
    VariableGaussianNoise,
    CCDNoise,
)

# Basic building blocks
from .bounds import Bounds, BoundsD, BoundsI
from .gsparams import GSParams
from .position import Position, PositionD, PositionI
from .angle import Angle, AngleUnit, _Angle, radians, hours, degrees, arcmin, arcsec

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
    _Image,
)

# GSObject
from .exponential import Exponential
from .gaussian import Gaussian
from .box import Box, Pixel
from .gsobject import GSObject
from .moffat import Moffat
from .spergel import Spergel
from .sum import Add, Sum
from .transform import Transform, Transformation
from .convolve import Convolve, Convolution, Deconvolution, Deconvolve
from .deltafunction import DeltaFunction

# WCS
from .wcs import (
    BaseWCS,
    AffineTransform,
    JacobianWCS,
    OffsetWCS,
    PixelScale,
    ShearWCS,
    OffsetShearWCS,
)
from .fits import FitsHeader
from .celestial import CelestialCoord
from .fitswcs import TanWCS, FitsWCS, GSFitsWCS

# Shear
from .shear import Shear, _Shear

# Interpolations
from .interpolant import (
    Interpolant,
    Delta,
    Nearest,
    SincInterpolant,
    Linear,
    Cubic,
    Quintic,
    Lanczos,
)
from .interpolatedimage import InterpolatedImage, _InterpolatedImage

# Photon Shooting
from .photon_array import PhotonArray
from .sensor import Sensor

# packages kept separate
from . import bessel
from . import fits
from . import integ

# this one is specific to jax_galsim
from . import core

from . import hsm
