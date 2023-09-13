"""
This module holds the the JAX-Galsim equivalents to Galsim's various
interpolant classes. The code here assumes that all properties of the
interpolants themselves (e.g., the coefficients that define the kernel
shapes, the integrals of the kernels, etc.) are constants.
"""
import galsim as _galsim
from galsim.errors import GalSimValueError

import jax
import jax.numpy as jnp
from jax._src.numpy.util import _wraps
from jax.tree_util import register_pytree_node_class

from jax_galsim.gsparams import GSParams
from jax_galsim.bessel import si


@_wraps(_galsim.interpolant.Interpolant)
class Interpolant:
    def __init__(self):
        raise NotImplementedError(
            "The Interpolant base class should not be instantiated directly. "
            "Use one of the subclasses instead, or use the `from_name` factory function."
        )

    @staticmethod
    def from_name(name, tol=None, gsparams=None):
        """A factory function to create an `Interpolant` of the correct type according to
        the (string) name of the `Interpolant`.

        This is mostly used to simplify how config files specify the `Interpolant` to use.

        Valid names are:

            - 'delta' = `Delta`
            - 'nearest' = `Nearest`
            - 'sinc' = `SincInterpolant`
            - 'linear' = `Linear`
            - 'cubic' = `Cubic`
            - 'quintic' = `Quintic`
            - 'lanczosN' = `Lanczos`  (where N is an integer, given the ``n`` parameter)

        In addition, if you want to specify the ``conserve_dc`` option for `Lanczos`, you can
        append either T or F to represent ``conserve_dc = True/False`` (respectively).  Otherwise,
        the default ``conserve_dc=True`` is used.

        Parameters:
            name:       The name of the interpolant to create.
            tol:        [deprecated]
            gsparams:   An optional `GSParams` argument. [default: None]
        """
        if tol is not None:
            from galsim.deprecated import depr

            depr("tol", 2.2, "gsparams=GSParams(kvalue_accuracy=tol)")
            gsparams = GSParams(kvalue_accuracy=tol)
        gsparams = GSParams.check(gsparams)

        # Do these in rough order of likelihood (most to least)
        # FIXME commented for testing
        if name.lower() == "quintic":
            return Quintic(gsparams=gsparams)
        # elif name.lower().startswith('lanczos'):
        #     conserve_dc = True
        #     if name[-1].upper() in ('T', 'F'):
        #         conserve_dc = (name[-1].upper() == 'T')
        #         name = name[:-1]
        #     try:
        #         n = int(name[7:])
        #     except Exception:
        #         raise GalSimValueError("Invalid Lanczos specification. Should look like "
        #                                "lanczosN, where N is an integer", name) from None
        #     return Lanczos(n, conserve_dc, gsparams=gsparams)
        elif name.lower() == "linear":
            return Linear(gsparams=gsparams)
        elif name.lower() == "cubic":
            return Cubic(gsparams=gsparams)
        elif name.lower() == "nearest":
            return Nearest(gsparams=gsparams)
        elif name.lower() == "delta":
            return Delta(gsparams=gsparams)
        elif name.lower() == "sinc":
            return SincInterpolant(gsparams=gsparams)
        else:
            raise GalSimValueError(
                "Invalid Interpolant name %s.",
                name,
                (
                    "linear",
                    "cubic",
                    "quintic",
                    "lanczosN",
                    "nearest",
                    "delta",
                    "sinc",
                ),
            )

    @property
    def gsparams(self):
        """The `GSParams` of the `Interpolant`"""
        return self._gsparams

    @property
    def tol(self):
        from galsim.deprecated import depr

        depr("interpolant.tol", 2.2, "interpolant.gsparams.kvalue_accuracy")
        return self._gsparams.kvalue_accuracy

    def withGSParams(self, gsparams=None, **kwargs):
        """Create a version of the current interpolant with the given gsparams"""
        if gsparams == self.gsparams:
            return self
        from copy import copy

        ret = copy(self)
        ret._gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        return ret

    def __repr__(self):
        return "jax_galsim.%s(gsparams=%r)" % (self.__class__.__name__, self._gsparams)

    def __str__(self):
        return "jax_galsim.%s()" % self.__class__.__name__

    # hack for galsim which sometimes uses this private attribute in
    # its code
    @property
    def _i(self):
        return self

    def __eq__(self, other):
        return (self is other) or (
            type(other) is self.__class__
            and self.tree_flatten() == other.tree_flatten()
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(repr(self))

    def xval(self, x):
        """Calculate the value of the interpolant kernel at one or more x values

        Parameters:
            x:      The value (as a float) or values (as a np.array) at which to compute the
                    amplitude of the Interpolant kernel.

        Returns:
            xval:   The value(s) at the x location(s).  If x was an array, then this is also
                    an array.
        """
        if jnp.ndim(x) > 1:
            raise GalSimValueError("kval only takes scalar or 1D array values", x)

        # we use functions attached directly to the class rather than static methods
        # this enables jax.jit which didn't react nicely with the @staticmethod decorator
        # when I tried it - MRB
        return self.__class__._xval(x)

    def kval(self, k):
        """Calculate the value of the interpolant kernel in Fourier space at one or more k values.

        Parameters:
            k:      The value (as a float) or values (as a np.array) at which to compute the
                    amplitude of the Interpolant kernel in Fourier space.

        Returns:
            kval:   The k-value(s) at the k location(s).  If k was an array, then this is also
                    an array.
        """
        if jnp.ndim(k) > 1:
            raise GalSimValueError("kval only takes scalar or 1D array values", k)

        # we use functions attached directly to the class rather than static methods
        # this enables jax.jit which didn't react nicely with the @staticmethod decorator
        # when I tried it - MRB
        # Note: self._uval uses u = k/2pi rather than k.
        return self.__class__._uval(k / 2.0 / jnp.pi)

    def unit_integrals(self, max_len=None):
        """Compute the unit integrals of the real-space kernel.

        integrals[i] = int(xval(x), i-0.5, i+0.5)

        Parameters:
            max_len:    The maximum length of the returned array. (ignored)

        Returns:
            integrals:  An array of unit integrals of length max_len or smaller.
        """
        return self._unit_integrals

    def tree_flatten(self):
        """This function flattens the Interpolant into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        children = tuple()
        # Define auxiliary static data that doesn’t need to be traced
        aux_data = {
            "gsparams": self._gsparams,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flattened representation"""
        return cls(**aux_data)

    @property
    def positive_flux(self):
        """The positive-flux fraction of the interpolation kernel."""
        return self._positive_flux

    @property
    def negative_flux(self):
        """The negative-flux fraction of the interpolation kernel."""
        return self._negative_flux

    @property
    def xrange(self):
        """The maximum extent of the interpolant from the origin (in pixels)."""
        raise NotImplementedError(
            "xrange is not implemented for interpolant type %s."
            % self.__class__.__name__
        )

    @property
    def ixrange(self):
        """The total integral range of the interpolant.  Typically 2 * xrange."""
        return 2 * int(jnp.ceil(self.xrange))

    @property
    def krange(self):
        """The maximum extent of the interpolant in Fourier space (in 1/pixels)."""
        return 2.0 * jnp.pi * self.urange()

    def urange(self):
        """The maximum extent of the interpolant in Fourier space (in 2pi/pixels)."""
        raise NotImplementedError(
            "urange() is not implemented for interpolant type %s."
            % self.__class__.__name__
        )

    # subclasses should implement __init__, _xval, _uval,
    # _unit_integrals, _positive_flux, _negative_flux, urange, and xrange
    # _xval and _uval should be static methods without the @staticmethod decorator


@_wraps(_galsim.interpolant.Delta)
@register_pytree_node_class
class Delta(Interpolant):
    _positive_flux = 1.0
    _negative_flux = 0.0
    _unit_integrals = jnp.array([1], dtype=float)

    def __init__(self, tol=None, gsparams=None):
        if tol is not None:
            from galsim.deprecated import depr

            depr("tol", 2.2, "gsparams=GSParams(kvalue_accuracy=tol)")
            gsparams = GSParams(kvalue_accuracy=tol)
        self._gsparams = GSParams.check(gsparams)

    # we use functions attached directly to the class rather than static methods
    # this enables jax.jit which didn't react nicely with the @staticmethod decorator
    # when I tried it - MRB
    @jax.jit
    def _xval(x, kvalue_accuracy):
        return jnp.where(
            jnp.abs(x) > 0.5 * kvalue_accuracy,
            0.0,
            1.0 / kvalue_accuracy,
        )

    def xval(self, x):
        """Calculate the value of the interpolant kernel at one or more x values

        Parameters:
            x:      The value (as a float) or values (as a np.array) at which to compute the
                    amplitude of the Interpolant kernel.

        Returns:
            xval:   The value(s) at the x location(s).  If x was an array, then this is also
                    an array.
        """
        # we need an override here since we use a parameter from self but the _xval method is
        # a pure function
        return self.__class__._xval(x, self._gsparams.kvalue_accuracy)

    # we use functions attached directly to the class rather than static methods
    # this enables jax.jit which didn't react nicely with the @staticmethod decorator
    # when I tried it - MRB
    @jax.jit
    def _uval(u):
        return jnp.ones_like(u)

    def urange(self):
        """The maximum extent of the interpolant in Fourier space (in 2pi/pixels)."""
        return 1.0 / self._gsparams.kvalue_accuracy

    @property
    def xrange(self):
        """The maximum extent of the interpolant from the origin (in pixels)."""
        return 0.0

    @property
    def ixrange(self):
        """The total integral range of the interpolant.  Typically 2 * xrange."""
        return 0


@_wraps(_galsim.interpolant.Nearest)
@register_pytree_node_class
class Nearest(Interpolant):
    _positive_flux = 1.0
    _negative_flux = 0.0
    _unit_integrals = jnp.array([1], dtype=float)

    def __init__(self, tol=None, gsparams=None):
        if tol is not None:
            from galsim.deprecated import depr

            depr("tol", 2.2, "gsparams=GSParams(kvalue_accuracy=tol)")
            gsparams = GSParams(kvalue_accuracy=tol)
        self._gsparams = GSParams.check(gsparams)

    # we use functions attached directly to the class rather than static methods
    # this enables jax.jit which didn't react nicely with the @staticmethod decorator
    # when I tried it - MRB
    @jax.jit
    def _xval(x):
        return jnp.where(jnp.abs(x) > 0.5, 0.0, 1.0)

    # we use functions attached directly to the class rather than static methods
    # this enables jax.jit which didn't react nicely with the @staticmethod decorator
    # when I tried it - MRB
    @jax.jit
    def _uval(u):
        return jnp.sinc(u)

    def urange(self):
        """The maximum extent of the interpolant in Fourier space (in 2pi/pixels)."""
        return 1.0 / (jnp.pi * self._gsparams.kvalue_accuracy)

    @property
    def xrange(self):
        """The maximum extent of the interpolant from the origin (in pixels)."""
        return 0.5

    @property
    def ixrange(self):
        """The total integral range of the interpolant.  Typically 2 * xrange."""
        return 1


@_wraps(_galsim.interpolant.SincInterpolant)
@register_pytree_node_class
class SincInterpolant(Interpolant):
    # these magic numbers are from galsim itself via
    # In [1]: import galsim
    # In [2]: s = galsim.SincInterpolant()
    # In [3]: s.positive_flux
    # Out[3]: 3.18724409580418
    # In [4]: s.negative_flux
    # Out[4]: 2.18724409580418
    _positive_flux = 3.18724409580418
    _negative_flux = 2.18724409580418

    def __init__(self, tol=None, gsparams=None):
        if tol is not None:
            from galsim.deprecated import depr

            depr("tol", 2.2, "gsparams=GSParams(kvalue_accuracy=tol)")
            gsparams = GSParams(kvalue_accuracy=tol)
        self._gsparams = GSParams.check(gsparams)

    # we use functions attached directly to the class rather than static methods
    # this enables jax.jit which didn't react nicely with the @staticmethod decorator
    # when I tried it - MRB
    @jax.jit
    def _xval(x):
        return jnp.sinc(x)

    # we use functions attached directly to the class rather than static methods
    # this enables jax.jit which didn't react nicely with the @staticmethod decorator
    # when I tried it - MRB
    @jax.jit
    def _uval(u):
        absu = jnp.abs(u)
        return jnp.where(
            absu > 0.5,
            0.0,
            jnp.where(absu < 0.5, 1.0, 0.5),
        )

    def urange(self):
        """The maximum extent of the interpolant in Fourier space (in 2pi/pixels)."""
        return 0.5

    @property
    def xrange(self):
        """The maximum extent of the interpolant from the origin (in pixels)."""
        # Technically infinity, but truncated by the tolerance.
        return 1.0 / (jnp.pi * self._gsparams.kvalue_accuracy)

    def unit_integrals(self, max_len=None):
        """Compute the unit integrals of the real-space kernel.

        integrals[i] = int(xval(x), i-0.5, i+0.5)

        Parameters:
            max_len:    The maximum length of the returned array.

        Returns:
            integrals:  An array of unit integrals of length max_len or smaller.
        """
        n = self.ixrange // 2 + 1
        n = n if max_len is None else min(n, max_len)
        if not hasattr(self, "_unit_integrals") or n > len(self._unit_integrals):
            narr = jnp.arange(n)
            # these are constants so we do not propagate gradients
            self._unit_integrals = jax.lax.stop_gradient(
                (si(jnp.pi * (narr + 0.5)) - si(jnp.pi * (narr - 0.5))) / jnp.pi
            )
        return self._unit_integrals[:n]


@_wraps(_galsim.interpolant.Linear)
@register_pytree_node_class
class Linear(Interpolant):
    _positive_flux = 1.0
    _negative_flux = 0.0
    # from galsim itself via
    # >>> galsim.Linear().unit_integrals()
    _unit_integrals = jnp.array([0.75, 0.125], dtype=float)

    def __init__(self, tol=None, gsparams=None):
        if tol is not None:
            from galsim.deprecated import depr

            depr("tol", 2.2, "gsparams=GSParams(kvalue_accuracy=tol)")
            gsparams = GSParams(kvalue_accuracy=tol)
        self._gsparams = GSParams.check(gsparams)

    # we use functions attached directly to the class rather than static methods
    # this enables jax.jit which didn't react nicely with the @staticmethod decorator
    # when I tried it - MRB
    @jax.jit
    def _xval(x):
        absx = jnp.abs(x)
        return jnp.where(
            absx > 1,
            0.0,
            1.0 - absx,
        )

    # we use functions attached directly to the class rather than static methods
    # this enables jax.jit which didn't react nicely with the @staticmethod decorator
    # when I tried it - MRB
    @jax.jit
    def _uval(u):
        s = jnp.sinc(u)
        return s * s

    def urange(self):
        """The maximum extent of the interpolant in Fourier space (in 2pi/pixels)."""
        return 1.0 / jnp.sqrt(self._gsparams.kvalue_accuracy) / jnp.pi

    @property
    def xrange(self):
        """The maximum extent of the interpolant from the origin (in pixels)."""
        return 1.0

    @property
    def ixrange(self):
        """The total integral range of the interpolant.  Typically 2 * xrange."""
        return 2


@_wraps(_galsim.interpolant.Cubic)
@register_pytree_node_class
class Cubic(Interpolant):
    # these constants are from galsim itself in the cpp layer
    # at include/Interpolant.h
    _positive_flux = 13.0 / 12.0
    _negative_flux = 1.0 / 12.0
    # from galsim itself via the source at galsim.interpolant.Cubic
    _unit_integrals = jnp.array([161.0 / 192, 3.0 / 32, -5.0 / 384], dtype=float)

    def __init__(self, tol=None, gsparams=None):
        if tol is not None:
            from galsim.deprecated import depr

            depr("tol", 2.2, "gsparams=GSParams(kvalue_accuracy=tol)")
            gsparams = GSParams(kvalue_accuracy=tol)
        self._gsparams = GSParams.check(gsparams)

    # we use functions attached directly to the class rather than static methods
    # this enables jax.jit which didn't react nicely with the @staticmethod decorator
    # when I tried it - MRB
    @jax.jit
    def _xval(x):
        x = jnp.abs(x)

        def _one(x):
            return 1.0 + x * x * (1.5 * x - 2.5)

        def _two(x):
            return -0.5 * (x - 1.0) * (x - 2.0) * (x - 2.0)

        msk1 = x < 1.0
        msk2 = x < 2.0

        return jnp.piecewise(
            x,
            [msk1, (~msk1) & msk2],
            [_one, _two, lambda x: jnp.array(0.0)],
        )

    # we use functions attached directly to the class rather than static methods
    # this enables jax.jit which didn't react nicely with the @staticmethod decorator
    # when I tried it - MRB
    @jax.jit
    def _uval(u):
        u = jnp.abs(u)
        s = jnp.sinc(u)
        c = jnp.cos(jnp.pi * u)
        return s * s * s * (3.0 * s - 2.0 * c)

    def urange(self):
        """The maximum extent of the interpolant in Fourier space (in 2pi/pixels)."""
        # magic formula from galsim CPP layer in src/Interpolant.cpp
        return (
            jnp.power(
                (3.0 * jnp.sqrt(3.0) / 8.0) / self._gsparams.kvalue_accuracy, 1.0 / 3.0
            )
            / jnp.pi
        )

    @property
    def xrange(self):
        """The maximum extent of the interpolant from the origin (in pixels)."""
        return 2.0

    @property
    def ixrange(self):
        """The total integral range of the interpolant.  Typically 2 * xrange."""
        return 4


@_wraps(_galsim.interpolant.Quintic)
@register_pytree_node_class
class Quintic(Interpolant):
    # these constants are from galsim itself in the cpp layer
    # at include/Interpolant.h
    _positive_flux = 1.1293413499280066555
    _negative_flux = 0.1293413499280066555
    # from galsim itself via the source at galsim.interpolant.Cubic
    _unit_integrals = jnp.array([161.0 / 192, 3.0 / 32, -5.0 / 384], dtype=float)

    def __init__(self, tol=None, gsparams=None):
        if tol is not None:
            from galsim.deprecated import depr

            depr("tol", 2.2, "gsparams=GSParams(kvalue_accuracy=tol)")
            gsparams = GSParams(kvalue_accuracy=tol)
        self._gsparams = GSParams.check(gsparams)

    # we use functions attached directly to the class rather than static methods
    # this enables jax.jit which didn't react nicely with the @staticmethod decorator
    # when I tried it - MRB
    @jax.jit
    def _xval(x):
        x = jnp.abs(x)

        def _one(x):
            return 1.0 + x * x * x * (
                -95.0 / 12.0 + x * (23.0 / 2.0 + x * (-55.0 / 12.0))
            )

        def _two(x):
            return (
                (x - 1.0)
                * (x - 2.0)
                * (
                    -23.0 / 4.0
                    + x * (29.0 / 2.0 + x * (-83.0 / 8.0 + x * (55.0 / 24.0)))
                )
            )

        def _three(x):
            return (
                (x - 2.0)
                * (x - 3.0)
                * (x - 3.0)
                * (-9.0 / 4.0 + x * (25.0 / 12.0 + x * (-11.0 / 24.0)))
            )

        msk1 = x <= 1.0
        msk2 = x <= 2.0
        msk3 = x <= 3.0

        return jnp.piecewise(
            x,
            [msk1, (~msk1) & msk2, (~msk2) & msk3],
            [_one, _two, _three, lambda x: jnp.array(0.0)],
        )

    # we use functions attached directly to the class rather than static methods
    # this enables jax.jit which didn't react nicely with the @staticmethod decorator
    # when I tried it - MRB
    @jax.jit
    def _uval(u):
        u = jnp.abs(u)
        s = jnp.sinc(u)
        piu = jnp.pi * u
        c = jnp.cos(piu)
        ssq = s * s
        piusq = piu * piu
        return s * ssq * ssq * (s * (55.0 - 19.0 * piusq) + 2.0 * c * (piusq - 27.0))

    def urange(self):
        """The maximum extent of the interpolant in Fourier space (in 2pi/pixels)."""
        # magic formula from galsim CPP layer in src/Interpolant.cpp
        return (
            jnp.power(
                (25.0 * jnp.sqrt(5.0) / 108.0) / self._gsparams.kvalue_accuracy,
                1.0 / 3.0,
            )
            / jnp.pi
        )

    @property
    def xrange(self):
        """The maximum extent of the interpolant from the origin (in pixels)."""
        return 3.0

    @property
    def ixrange(self):
        """The total integral range of the interpolant.  Typically 2 * xrange."""
        return 6
