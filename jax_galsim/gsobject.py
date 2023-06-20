import jax.numpy as jnp

from jax._src.numpy.util import _wraps
import galsim as _galsim

from jax_galsim.gsparams import GSParams
from jax_galsim.position import PositionD
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
        """Equivalent to `xValue`, but ``pos`` must be a `galsim.PositionD` instance

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
