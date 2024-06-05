import warnings

import galsim as _galsim
from jax._src.numpy.util import implements

from jax_galsim.errors import GalSimDeprecationWarning


@implements(
    _galsim.deprecated.depr,
    lax_description="""\
The JAX version of this function uses `stacklevel=3` to show where the
warning is generated.""",
)
def depr(f, v, s1, s2=None):
    s = str(f) + " has been deprecated since GalSim version " + str(v) + "."
    if s1:
        s += "  Use " + s1 + " instead."
    if s2:
        s += "  " + s2
    warnings.warn(s, GalSimDeprecationWarning, stacklevel=3)
