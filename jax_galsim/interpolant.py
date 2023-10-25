"""
This module holds the the JAX-Galsim equivalents to Galsim's various
interpolant classes. The code here assumes that all properties of the
interpolants themselves (e.g., the coefficients that define the kernel
shapes, the integrals of the kernels, etc.) are constants.
"""
import galsim as _galsim
import jax
import jax.numpy as jnp
from galsim.errors import GalSimValueError
from jax._src.numpy.util import _wraps
from jax.tree_util import register_pytree_node_class

from jax_galsim.bessel import si
from jax_galsim.core.utils import is_equal_with_arrays
from jax_galsim.gsparams import GSParams


@_wraps(_galsim.interpolant.Interpolant)
@register_pytree_node_class
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
        if name.lower() == "quintic":
            return Quintic(gsparams=gsparams)
        elif name.lower().startswith("lanczos"):
            conserve_dc = True
            if name[-1].upper() in ("T", "F"):
                conserve_dc = name[-1].upper() == "T"
                name = name[:-1]
            try:
                n = int(name[7:])
            except Exception:
                raise GalSimValueError(
                    "Invalid Lanczos specification. Should look like "
                    "lanczosN, where N is an integer",
                    name,
                ) from None
            return Lanczos(n, conserve_dc, gsparams=gsparams)
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
        # Checking gsparams
        gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        # Flattening the representation to instantiate a clean new object
        children, aux_data = self.tree_flatten()
        aux_data["gsparams"] = gsparams
        return self.tree_unflatten(aux_data, children)

    def __repr__(self):
        return "galsim.%s(gsparams=%r)" % (self.__class__.__name__, self._gsparams)

    def __str__(self):
        return "galsim.%s()" % self.__class__.__name__

    # hack for galsim which sometimes uses this private attribute in
    # its code
    @property
    def _i(self):
        return self

    def __eq__(self, other):
        return (self is other) or (
            type(other) is self.__class__
            and is_equal_with_arrays(self.tree_flatten()[1], other.tree_flatten()[1])
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
            raise GalSimValueError("xval only takes scalar or 1D array values", x)

        return self._xval_noraise(x)

    def _xval_noraise(self, x):
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

        return self._kval_noraise(k)

    def _kval_noraise(self, k):
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
        if not hasattr(self, "_positive_flux"):
            # subclasses can define this method if _positive_flux is not set
            self._comp_fluxes()
        return self._positive_flux

    @property
    def negative_flux(self):
        """The negative-flux fraction of the interpolation kernel."""
        if not hasattr(self, "_negative_flux"):
            # subclasses can define this method if _negative_flux is not set
            self._comp_fluxes()
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

    def _xval_noraise(self, x):
        return Delta._xval(x, self._gsparams.kvalue_accuracy)

    @jax.jit
    def _xval(x, kva):
        return jnp.where(
            jnp.abs(x) > 0.5 * kva,
            0.0,
            1.0 / kva,
        )

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

    @jax.jit
    def _xval(x):
        return jnp.where(jnp.abs(x) > 0.5, 0.0, 1.0)

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
    def __init__(self, tol=None, gsparams=None):
        if tol is not None:
            from galsim.deprecated import depr

            depr("tol", 2.2, "gsparams=GSParams(kvalue_accuracy=tol)")
            gsparams = GSParams(kvalue_accuracy=tol)
        self._gsparams = GSParams.check(gsparams)

    @jax.jit
    def _xval(x):
        return jnp.sinc(x)

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

    def _comp_fluxes(self):
        # the sinc function oscillates so we want to integrate over an even number of periods
        n = self.ixrange // 2 + 1
        if n % 2 != 0:
            n += 1
        narr = jnp.arange(n)

        val = (si(jnp.pi * (narr + 1)) - si(jnp.pi * (narr))) / jnp.pi
        self._positive_flux = jax.lax.stop_gradient(jnp.sum(val[val > 0])).item() * 2.0
        self._negative_flux = self._positive_flux - 1.0


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

    @jax.jit
    def _xval(x):
        absx = jnp.abs(x)
        return jnp.where(
            absx > 1,
            0.0,
            1.0 - absx,
        )

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
            [_one, _two, lambda x: jnp.array(0, dtype=float)],
        )

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
    # from galsim itself via `galsim.Quintic().unit_integrals()`
    _unit_integrals = jnp.array(
        [
            0.8724826228119177,
            0.07332899380883082,
            -0.010894097523532266,
            0.0013237847222222375,
        ],
        dtype=float,
    )

    def __init__(self, tol=None, gsparams=None):
        if tol is not None:
            from galsim.deprecated import depr

            depr("tol", 2.2, "gsparams=GSParams(kvalue_accuracy=tol)")
            gsparams = GSParams(kvalue_accuracy=tol)
        self._gsparams = GSParams.check(gsparams)

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
            [_one, _two, _three, lambda x: jnp.array(0, dtype=float)],
        )

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


@_wraps(_galsim.interpolant.Lanczos)
@register_pytree_node_class
class Lanczos(Interpolant):
    # this data was generated in the dev notebook at
    # dev/notebooks/lanczos_interp_devel.ipynb
    _posflux_conserve_dc = {
        1: 1.0,
        2: 1.0886717592825461,
        3: 1.1793666081853116,
        4: 1.2346151179761133,
        5: 1.281463486428329,
        6: 1.3175325895635532,
        7: 1.3493598581550328,
        8: 1.3760445182705399,
        9: 1.4001934379681609,
        10: 1.4213484298874304,
        11: 1.4408189363340553,
        12: 1.45833565807884,
        13: 1.474652278722511,
        14: 1.4895955544666875,
        15: 1.5036403723153653,
        16: 1.5166680411334044,
        17: 1.5289977813911073,
        18: 1.5405443862220922,
        19: 1.551533132364491,
        20: 1.5619005957998309,
        21: 1.5718119037317102,
        22: 1.581218460467469,
        23: 1.5902450443637517,
        24: 1.5988535289987609,
        25: 1.6071405870703896,
        26: 1.6150757091477028,
        27: 1.622735371645929,
        28: 1.6300947482196946,
        29: 1.6372154180886938,
        30: 1.6440768869193918,
    }
    _posflux_no_conserve_dc = {
        1: 1.0,
        2: 1.0888955808394716,
        3: 1.1792072356792431,
        4: 1.2347046618592585,
        5: 1.2814072306391975,
        6: 1.3175695751403054,
        7: 1.3493340189069116,
        8: 1.3760632172275888,
        9: 1.4001794032906607,
        10: 1.4213592250706284,
        11: 1.440810429974071,
        12: 1.4583424802439546,
        13: 1.4746467128028455,
        14: 1.4896001558822005,
        15: 1.5036365194014063,
        16: 1.5166713004988444,
        17: 1.528994996764082,
        18: 1.5405467847300993,
        19: 1.5515310501098805,
        20: 1.5619024155425638,
        21: 1.57181030316382,
        22: 1.581219876037236,
        23: 1.590243785711868,
        24: 1.598854653338271,
        25: 1.6071395781758782,
        26: 1.6150766180510743,
        27: 1.6227345496614405,
        28: 1.6300954941514494,
        29: 1.637214738911383,
        30: 1.6440775071701577,
    }
    _negflux_conserve_dc = {
        1: 0.0,
        2: 0.08867175928254616,
        3: 0.17936660818531167,
        4: 0.23461511797611329,
        5: 0.2814634864283287,
        6: 0.3175325895635534,
        7: 0.3493598581550328,
        8: 0.3760445182705398,
        9: 0.40019343796816076,
        10: 0.4213484298874303,
        11: 0.4408189363340553,
        12: 0.45833565807883997,
        13: 0.4746522787225108,
        14: 0.48959555446668734,
        15: 0.5036403723153652,
        16: 0.5166680411334043,
        17: 0.5289977813911073,
        18: 0.5405443862220923,
        19: 0.551533132364491,
        20: 0.5619005957998309,
        21: 0.5718119037317101,
        22: 0.5812184604674691,
        23: 0.5902450443637517,
        24: 0.598853528998761,
        25: 0.6071405870703895,
        26: 0.6150757091477027,
        27: 0.622735371645929,
        28: 0.6300947482196946,
        29: 0.6372154180886938,
        30: 0.6440768869193919,
    }
    _negflux_no_conserve_dc = {
        1: 0.0,
        2: 0.08889558083947162,
        3: 0.17920723567924315,
        4: 0.23470466185925848,
        5: 0.2814072306391975,
        6: 0.31756957514030526,
        7: 0.3493340189069117,
        8: 0.3760632172275888,
        9: 0.4001794032906608,
        10: 0.4213592250706285,
        11: 0.44081042997407094,
        12: 0.45834248024395463,
        13: 0.47464671280284537,
        14: 0.4896001558822007,
        15: 0.5036365194014063,
        16: 0.5166713004988444,
        17: 0.5289949967640818,
        18: 0.5405467847300994,
        19: 0.5515310501098805,
        20: 0.5619024155425638,
        21: 0.57181030316382,
        22: 0.5812198760372359,
        23: 0.5902437857118682,
        24: 0.598854653338271,
        25: 0.6071395781758783,
        26: 0.6150766180510744,
        27: 0.6227345496614405,
        28: 0.6300954941514493,
        29: 0.6372147389113828,
        30: 0.6440775071701577,
    }

    # fmt: off
    _unit_integrals_no_conserve_dc = {
        1: jnp.array([0.7736950099028163, 0.0645641638701392], dtype=float),
        2: jnp.array([
            0.8465887094380671, 0.09303749006303791,
            -0.011436924726642468], dtype=float),
        3: jnp.array([
            0.8609564689315261, 0.0854975941631095,
            -0.02198305969046234, 0.0045349041140008714], dtype=float),
        4: jnp.array([
            0.866051828001578, 0.08164164710461341,
            -0.0211168606127283, 0.009485060642921659,
            -0.0024077071164748673], dtype=float),
        5: jnp.array([
            0.8684220505647028, 0.07961972252745429,
            -0.020016735819945855, 0.009558891823702252,
            -0.005184575956143513, 0.0014879848847649067], dtype=float),
        6: jnp.array([
            0.8697127191238894, 0.07845585622633226,
            -0.01920340255869482, 0.009237930102390803,
            -0.005384978311733045, 0.0032356834836430046,
            -0.001009369059991005], dtype=float),
        7: jnp.array([
            8.7049202177269380e-01, 7.7731321479942828e-02,
            -1.8633958804874725e-02, 8.8979604030649008e-03,
            -5.3109613153060203e-03, 3.4211298515002897e-03,
            -2.1994566985353557e-03, 7.2920194182248238e-04], dtype=float),
        8: jnp.array([
            8.7099825045836277e-01, 7.7251798117792966e-02,
            -1.8231183550600525e-02, 8.6123371774333903e-03,
            -5.1683105487452634e-03, 3.4297387648459152e-03,
            -2.3495038301018420e-03, 1.5869647419779489e-03,
            -5.5126917083166398e-04], dtype=float),
        9: jnp.array([
            8.7134551547294570e-01, 7.6918789259528400e-02,
            -1.7939545253488193e-02, 8.3851399529145665e-03,
            -5.0223675581809370e-03, 3.3740587008968420e-03,
            -2.3845688672012753e-03, 1.7048998137498051e-03,
            -1.1964980987349909e-03, 4.3128985494303028e-04], dtype=float),
        10: jnp.array([
            8.7159401009600912e-01, 7.6678458714107514e-02,
            -1.7723066666894282e-02, 8.2063352309026966e-03,
            -4.8921265119170072e-03, 3.2995228350380813e-03,
            -2.3686328888224547e-03, 1.7460701688724418e-03,
            -1.2891159996234734e-03, 9.3303369245166100e-04,
            -3.4658944544847537e-04], dtype=float),
        11: jnp.array([
            8.7177792063920001e-01, 7.6499493089007983e-02,
            -1.7558618559349929e-02, 8.0650477479507002e-03,
            -4.7811660243816710e-03, 3.2241347417886492e-03,
            -2.3318307976264293e-03, 1.7484791053141236e-03,
            -1.3289789960901517e-03, 1.0063792430271374e-03,
            -7.4723017598378385e-04, 2.8458455048163499e-04], dtype=float),
        12: jnp.array([
            8.7191782966320464e-01, 7.6362719513935945e-02,
            -1.7431083730257708e-02, 7.9523619355260939e-03,
            -4.6881497744781127e-03, 3.1545680896655440e-03,
            -2.2882627635925478e-03, 1.7321081511208159e-03,
            -1.3396107940286301e-03, 1.0424634183502162e-03,
            -8.0597292403970474e-04, 6.1147711864485818e-04,
            -2.3783833713667121e-04], dtype=float),
        13: jnp.array([
            8.7202672970489126e-01, 7.6255884824415227e-02,
            -1.7330351693013456e-02, 7.8614942460054384e-03,
            -4.6104639074225039e-03, 3.0927905379740887e-03,
            -2.2444112684260379e-03, 1.7074085581526748e-03,
            -1.3344523827163023e-03, 1.0563945723023067e-03,
            -8.3773694673650576e-04, 6.5908595977630401e-04,
            -5.0937905883310702e-04, 2.0172731901756032e-04], dtype=float),
        14: jnp.array([
            8.7211314971773835e-01, 7.6170869830959512e-02,
            -1.7249497299346866e-02, 7.7873949593983190e-03,
            -4.5454533508222910e-03, 3.0388526410040475e-03,
            -2.2031061699838961e-03, 1.6798162769512065e-03,
            -1.3211365252795528e-03, 1.0573874634861341e-03,
            -8.5254662631379762e-04, 6.8672453302556010e-04,
            -5.4841096286049721e-04, 4.3071793405843433e-04,
            -1.7325480547849065e-04], dtype=float),
        15: jnp.array([
            8.7218287617389989e-01, 7.6102125928469905e-02,
            -1.7183668054715523e-02, 7.7263146760676790e-03,
            -4.4907973134569864e-03, 2.9920817097864516e-03,
            -2.1654191960773467e-03, 1.6521120470743935e-03,
            -1.3039865774841996e-03, 1.0510206999873825e-03,
            -8.5684619912733797e-04, 7.0123701205311345e-04,
            -5.7236733234294261e-04, 4.6306552021452882e-04,
            -3.6886475367444002e-04, 1.5040954902298390e-04], dtype=float),
        16: jnp.array([
            8.7223994695383245e-01, 7.6045758887633361e-02,
            -1.7129391549300460e-02, 7.6754541194282476e-03,
            -4.4445780003925933e-03, 2.9515932830727230e-03,
            -2.1315844437767456e-03, 1.6256610135092448e-03,
            -1.2854367921998294e-03, 1.0406641239266898e-03,
            -8.5474326271209029e-04, 7.0721788371212930e-04,
            -5.8604031641364715e-04, 4.8383224833227180e-04,
            -3.9594161181058529e-04, 3.1937075418754133e-04,
            -1.3180105869222726e-04], dtype=float),
        17: jnp.array([
            8.7228724908408983e-01, 7.5998971448148489e-02,
            -1.7084134836402289e-02, 7.6327045089459814e-03,
            -4.4052470143101669e-03, 2.9165042259113850e-03,
            -2.1014515246492978e-03, 1.6010715354475096e-03,
            -1.2668368993947337e-03, 1.0283484922183938e-03,
            -8.4884900225765786e-04, 7.0772279880576214e-04,
            -5.9276553972714599e-04, 4.9644913927172625e-04,
            -4.1398268512120632e-04, 3.4224428911411873e-04,
            -2.7916255528629238e-04, 1.1644315484903639e-04], dtype=float),
        18: jnp.array([
            8.7232689100798066e-01, 7.5959712956827338e-02,
            -1.7046017805648218e-02, 7.5964606773141321e-03,
            -4.3715650387017120e-03, 2.8860139700952014e-03,
            -2.0747098206043615e-03, 1.5785497707302631e-03,
            -1.2489107355928228e-03, 1.0152899634641470e-03,
            -8.4082118027182986e-04, 7.0477714372669902e-04,
            -5.9484198587958525e-04, 5.0337889394365019e-04,
            -4.2549364826763872e-04, 3.5796893458213912e-04,
            -2.9864805145927314e-04, 2.4606304027479520e-04,
            -1.0362070336560659e-04], dtype=float),
        19: jnp.array([
            8.7236044158998860e-01, 7.5926452741888195e-02,
            -1.7013622704380147e-02, 7.5654876082696665e-03,
            -4.3425416523476855e-03, 2.8594278657083846e-03,
            -2.0509986610199346e-03, 1.5580925146567838e-03,
            -1.2320201808410233e-03, 1.0022067635564579e-03,
            -8.3170946225879843e-04, 6.9972181747986054e-04,
            -5.9384779557183941e-04, 5.0637122182818867e-04,
            -4.3232104020823621e-04, 3.6840727705250476e-04,
            -3.1240629187239864e-04, 2.6279030743023345e-04,
            -2.1849583061638169e-04, 9.2804910540045815e-05], dtype=float),
        20: jnp.array([
            8.7238908771063439e-01, 7.5898029757744429e-02,
            -1.6985864988803168e-02, 7.5388249542491558e-03,
            -4.3173835651028961e-03, 2.8361560238244453e-03,
            -2.0299599390871613e-03, 1.5395927071481215e-03,
            -1.2163185520714930e-03, 9.8951153105018451e-04,
            -8.2217430108841988e-04, 6.9344345157377183e-04,
            -5.9086471907923341e-04, 5.0666356989120939e-04,
            -4.3581160443271713e-04, 3.7496363711723285e-04,
            -3.2184401158880078e-04, 2.7487712165928825e-04,
            -2.3295672923797007e-04, 1.9529727563949068e-04,
            -8.3597962681059536e-05], dtype=float),
        21: jnp.array([
            8.7241374066547217e-01, 7.5873550551340735e-02,
            -1.6961904110473453e-02, 7.5157182853181478e-03,
            -4.2954526665775292e-03, 2.8157027761250950e-03,
            -2.0112615411254481e-03, 1.5228971254527585e-03,
            -1.2018405020159915e-03, 9.7742906202459394e-04,
            -8.1262604588436683e-04, 6.8652658754780871e-04,
            -5.8663367507584910e-04, 5.0512949871272454e-04,
            -4.3694262611906919e-04, 3.7868584373075532e-04,
            -3.2804320987374932e-04, 2.8340127614368980e-04,
            -2.4361905533554343e-04, 2.0787992034422971e-04,
            -1.7559354356606557e-04, 7.5695943263817707e-05], dtype=float),
        22: jnp.array([
            8.7243510952874703e-01, 7.5852318603272587e-02,
            -1.6941080818261168e-02, 7.4955691550713617e-03,
            -4.2762329287431644e-03, 2.7976535151004213e-03,
            -1.9946060597048234e-03, 1.5078376040994460e-03,
            -1.1885550078002533e-03, 9.6606895596750117e-04,
            -8.0331383050412635e-04, 6.7935422874696577e-04,
            -5.8166142427177995e-04, 5.0238523244853410e-04,
            -4.3642171145326028e-04, 3.8035152865207455e-04,
            -3.3182752898445328e-04, 2.8920645306073549e-04,
            -2.5131937483122271e-04, 2.1732403533461197e-04,
            -1.8660717479672992e-04, 1.5871831036838648e-04,
            -6.8863443560681072e-05], dtype=float),
        23: jnp.array([
            8.7245375270399328e-01, 7.5833784494476197e-02,
            -1.6922872388902346e-02, 7.4778984814789003e-03,
            -4.2593046027017932e-03, 2.7816618095121959e-03,
            -1.9797322506680923e-03, 1.4942474897794416e-03,
            -1.1763966613264351e-03, 9.5547074586343561e-04,
            -7.9438273789779500e-04, 6.7217448589296637e-04,
            -5.7629349381965126e-04, 4.9886514149724635e-04,
            -4.3476074882933607e-04, 3.8053630812966675e-04,
            -3.3382041542361171e-04, 2.9294634276861588e-04,
            -2.5672281594250180e-04, 2.2428666875100081e-04,
            -1.9500547130615447e-04, 1.6841157506093199e-04,
            -1.4415643723277331e-04, 6.2915835299457630e-05], dtype=float),
        24: jnp.array([
            8.7247011477047076e-01, 7.5817510174385608e-02,
            -1.6906860187092181e-02, 7.4623194228004095e-03,
            -4.2443241853712000e-03, 2.7674379023744462e-03,
            -1.9664130964798065e-03, 1.4819698421221781e-03,
            -1.1652840808074618e-03, 9.4563207893409045e-04,
            -7.8591079895459857e-04, 6.6514496036414769e-04,
            -5.7076407672435373e-04, 4.9487492254501235e-04,
            -4.3232985155486526e-04, 3.7966578688618756e-04,
            -3.3449241034248865e-04, 2.9512448908871914e-04,
            -2.6035290047572783e-04, 2.2929752273423575e-04,
            -2.0131008554716406e-04, 1.7590842016538329e-04,
            -1.5273096884774962e-04, 1.3150467872105143e-04,
            -5.7706672578191118e-05], dtype=float),
        25: jnp.array([
            8.7248455323036556e-01, 7.5803142951737873e-02,
            -1.6892705839008548e-02, 7.4485170857846545e-03,
            -4.2310088634876777e-03, 2.7547388884928393e-03,
            -1.9544524665390493e-03, 1.4708610415793437e-03,
            -1.1551306211969111e-03, 9.3652636098220802e-04,
            -7.7793307699596925e-04, 6.5836246090984535e-04,
            -5.6523028196857003e-04, 4.9062902739097905e-04,
            -4.2939640042571181e-04, 3.7805449419290975e-04,
            -3.3419816604274879e-04, 2.9612754832035524e-04,
            -2.6261925382132248e-04, 2.3277944130117913e-04,
            -2.0594683276384606e-04, 1.8162712002948100e-04,
            -1.5944761834648776e-04, 1.3912529030291550e-04,
            -1.2044378908257909e-04, 5.3118594590042846e-05], dtype=float),
        26: jnp.array([
            8.7249735820063523e-01, 7.5790396300352780e-02,
            -1.6880133511645610e-02, 7.4362331971641049e-03,
            -4.2191244024610613e-03, 2.7433605195079330e-03,
            -1.9436813821434409e-03, 1.4607918794507227e-03,
            -1.1458504583868243e-03, 9.2811379710192315e-04,
            -7.7045741697033098e-04, 6.5188300669198976e-04,
            -5.5979574126075674e-04, 4.8627704597194768e-04,
            -4.2615322316130310e-04, 3.7593475822376282e-04,
            -3.3320482802084127e-04, 2.9625194592567382e-04,
            -2.6384130015194340e-04, 2.3506794911517606e-04,
            -2.0925996398028855e-04, 1.8591295438802140e-04,
            -1.6464483369968912e-04, 1.4516394569028066e-04,
            -1.2724620052677611e-04, 1.1071840388516905e-04,
            -4.9056660028197339e-05], dtype=float),
        27: jnp.array([
            8.7250876709889491e-01, 7.5779035514417295e-02,
            -1.6868916579437666e-02, 7.4252544156374715e-03,
            -4.2084756787176842e-03, 2.7331304515287470e-03,
            -1.9339543756837175e-03, 1.4516473107149879e-03,
            -1.1373618915720602e-03, 9.2034826526717550e-04,
            -7.6347477396382650e-04, 6.4573535705184754e-04,
            -5.5452695411212379e-04, 4.8192235798248960e-04,
            -4.2273893459985977e-04, 3.7347802510557940e-04,
            -3.3171355222982299e-04, 2.9572476409232314e-04,
            -2.6426767094824054e-04, 2.3642833751774098e-04,
            -2.1152611996376781e-04, 1.8904828245449665e-04,
            -1.6860484863931961e-04, 1.4989680901512291e-04,
            -1.3269333113477609e-04, 1.1681515570485079e-04,
            -1.0212231839813374e-04, 4.5443397789426126e-05], dtype=float),
        28: jnp.array([
            8.7251897572658565e-01, 7.5768866862828438e-02,
            -1.6858867483315169e-02, 7.4154033396151200e-03,
            -4.1988992421305438e-03, 2.7239027165823207e-03,
            -1.9251461615742587e-03, 1.4433255406375988e-03,
            -1.1295889792020629e-03, 9.1318154687901267e-04,
            -7.5696599186509653e-04, 6.3993019946345393e-04,
            -5.4946465602485992e-04, 4.7763536687237917e-04,
            -4.1925264419631126e-04, 3.7081057968550653e-04,
            -3.2987572316521882e-04, 2.9471989961655750e-04,
            -2.6409171933591894e-04, 2.3706994310270847e-04,
            -2.1296679065825429e-04, 1.9126262668764352e-04,
            -1.7156087891359538e-04, 1.5355630859212208e-04,
            -1.3701227971507239e-04, 1.2174420936681089e-04,
            -1.0760733851429796e-04, 9.4487571582612369e-05,
            -4.2215086924736379e-05], dtype=float),
        29: jnp.array([
            8.7252814672517331e-01, 7.5759729300683573e-02,
            -1.6849829940161199e-02, 7.4065315297353416e-03,
            -4.1902574418264428e-03, 2.7155532080510806e-03,
            -1.9171486956285626e-03, 1.4357368233282864e-03,
            -1.1224621871665536e-03, 9.0656587595296816e-04,
            -7.5090624316456755e-04, 6.3446640677301607e-04,
            -5.4463175791790970e-04, 4.7346292654700531e-04,
            -4.1576461953847185e-04, 3.6802514557556475e-04,
            -3.2780514752917038e-04, 2.9337047419487888e-04,
            -2.6346374560621628e-04, 2.3715777325274403e-04,
            -2.1375892296893672e-04, 1.9274185415666959e-04,
            -1.7370448943493665e-04, 1.5633633450967894e-04,
            -1.4039579409560427e-04, 1.2569366589233272e-04,
            -1.1208095148205686e-04, 9.9439732125518435e-05,
            -8.7676252916474282e-05, 3.9318928927118588e-05], dtype=float),
        30: jnp.array([
            8.7253641609360022e-01, 7.5751488071776771e-02,
            -1.6841672901764325e-02, 7.3985140506990640e-03,
            -4.1824337607948240e-03, 2.7079759953663679e-03,
            -1.9098686290674434e-03, 1.4288021814002165e-03,
            -1.1159184635418527e-03, 9.0045541839242475e-04,
            -7.4526792009516009e-04, 6.2933530439841128e-04,
            -5.4003891046984942e-04, 4.6943508197744874e-04,
            -4.1232404262523802e-04, 3.6518946006937510e-04,
            -3.2558721861076828e-04, 2.9177833214063051e-04,
            -2.6250055118292828e-04, 2.3682182498082198e-04,
            -2.1404370601122558e-04, 1.9363613021185940e-04,
            -1.7519242664293955e-04, 1.5839767601534205e-04,
            -1.4300606356997960e-04, 1.2882440798063727e-04,
            -1.1570000483758914e-04, 1.0351153240198906e-04,
            -9.2162162830111905e-05, 8.1574283556929255e-05,
            -3.6710877004442137e-05], dtype=float),
    }
    _unit_integrals_conserve_dc = {
        1: jnp.array([0.8465162792377735, 0.07670762325450639], dtype=float),
        2: jnp.array([
            0.8392495486404705, 0.09164303260613273,
            -0.011263991256750852], dtype=float),
        3: jnp.array([
            0.8632681168763658, 0.08589201158463194,
            -0.022083030965503198, 0.004555815429348342], dtype=float),
        4: jnp.array([
            0.8650618038636482, 0.08148195151403913,
            -0.021076170181741696, 0.009466820720409927,
            -0.00240302266580677], dtype=float),
        5: jnp.array([
            0.8689359970259309, 0.07970029234018064,
            -0.020036683900998796, 0.009568395787771822,
            -0.005189727741783341, 0.0014894780779962505], dtype=float),
        6: jnp.array([
            0.8694137791076688, 0.07840978854063436,
            -0.0191922989228868, 0.009232601681276926,
            -0.0053818745239808796, 0.0032338189122908144,
            -0.0010087824853216595], dtype=float),
        7: jnp.array([
            8.7068121586962599e-01, 7.7760175859505912e-02,
            -1.8640769800640181e-02, 8.9012045828225275e-03,
            -5.3128961484314877e-03, 3.4223758240124881e-03,
            -2.2002576627743846e-03, 7.2946945446464068e-04], dtype=float),
        8: jnp.array([
            8.7087122663453687e-01, 7.7232562835075971e-02,
            -1.8226713770444866e-02, 8.6102310410169185e-03,
            -5.1670476898779145e-03, 3.4289010057657597e-03,
            -2.3489300172765616e-03, 1.5865771790561077e-03,
            -5.5113365980760675e-04], dtype=float),
        9: jnp.array([
            8.7143493458904431e-01, 7.6932265134858360e-02,
            -1.7942639888338646e-02, 8.3865827097040022e-03,
            -5.0232309792990620e-03, 3.3746385423195635e-03,
            -2.3849785913625278e-03, 1.7051927318036352e-03,
            -1.1967036642138026e-03, 4.3136438901798563e-04], dtype=float),
        10: jnp.array([
            8.7152875015003661e-01, 7.6668658898206218e-02,
            -1.7720836410665596e-02, 8.2053052134872188e-03,
            -4.8915130064100063e-03, 3.2991092084359097e-03,
            -2.3683360136017872e-03, 1.7458513443762314e-03,
            -1.2889544498105386e-03, 9.3291676738900878e-04,
            -3.4654578014316011e-04], dtype=float),
        11: jnp.array([
            8.7182701261013895e-01, 7.6506845944911556e-02,
            -1.7560280273053473e-02, 8.0658090349930087e-03,
            -4.7816169415337648e-03, 3.2244386963457415e-03,
            -2.3320505873390125e-03, 1.7486438928736800e-03,
            -1.3291042400695723e-03, 1.0064740823627395e-03,
            -7.4730059331932163e-04, 2.8461150002295668e-04], dtype=float),
        12: jnp.array([
            8.7187999180800302e-01, 7.6357063833305372e-02,
            -1.7429812583281152e-02, 7.9517835187695729e-03,
            -4.6878090801196039e-03, 3.1543389326472498e-03,
            -2.2880965711535176e-03, 1.7319823654493472e-03,
            -1.3395135178857025e-03, 1.0423877224169636e-03,
            -8.0591440128258737e-04, 6.1143271851679844e-04,
            -2.3782098971811296e-04], dtype=float),
        13: jnp.array([
            8.7205651245197036e-01, 7.6260329598980603e-02,
            -1.7331346297259138e-02, 7.8619442533867916e-03,
            -4.6107275866607853e-03, 3.0929673491064623e-03,
            -2.2445395521833180e-03, 1.7075061369574867e-03,
            -1.3345286416164889e-03, 1.0564549386888288e-03,
            -8.3778481703841387e-04, 6.5912362112473504e-04,
            -5.0940816568803755e-04, 2.0173889416053422e-04], dtype=float),
        14: jnp.array([
            8.7208929429187798e-01, 7.6167314173495135e-02,
            -1.7248704480452334e-02, 7.7870379659251690e-03,
            -4.5452451604283050e-03, 3.0387135114274592e-03,
            -2.2030053250851472e-03, 1.6797393944222787e-03,
            -1.3210760634191955e-03, 1.0573390743462247e-03,
            -8.5250761240626010e-04, 6.8669310792062658e-04,
            -5.4838586727713799e-04, 4.3069822413788851e-04,
            -1.7324684641373308e-04], dtype=float),
        15: jnp.array([
            8.7220228086915630e-01, 7.6105015315345276e-02,
            -1.7184310432719251e-02, 7.7266027585287623e-03,
            -4.4909646081825226e-03, 2.9921931283320119e-03,
            -2.1654998143708498e-03, 1.6521735473972161e-03,
            -1.3040351150346218e-03, 1.0510598196223819e-03,
            -8.5687809046123338e-04, 7.0126311118012027e-04,
            -5.7238863486558128e-04, 4.6308275463621986e-04,
            -3.6887848215115331e-04, 1.5041516736960623e-04], dtype=float),
        16: jnp.array([
            8.7222395364765015e-01, 7.6043379470208547e-02,
            -1.7128863829277020e-02, 7.6752182700120285e-03,
            -4.4444415499702789e-03, 2.9515027045387025e-03,
            -2.1315190437295600e-03, 1.6256111421785228e-03,
            -1.2853973612081734e-03, 1.0406322029994628e-03,
            -8.5471704553188052e-04, 7.0719619199953443e-04,
            -5.8602234170264432e-04, 4.8381740861131231e-04,
            -3.9592946783596728e-04, 3.1936095870500593e-04,
            -1.3179700236266062e-04], dtype=float),
        17: jnp.array([
            8.7230058718898085e-01, 7.6000954485776234e-02,
            -1.7084573752855357e-02, 7.6329000941166500e-03,
            -4.4053597965662749e-03, 2.9165788633803381e-03,
            -2.1015052921385842e-03, 1.6011124950665969e-03,
            -1.2668693058281661e-03, 1.0283747966025258e-03,
            -8.4887071441444009e-04, 7.0774090075281446e-04,
            -5.9278070108256223e-04, 4.9646183697964912e-04,
            -4.1399327351884985e-04, 3.4225304265024032e-04,
            -2.7916969540569040e-04, 1.1644614272353818e-04], dtype=float),
        18: jnp.array([
            8.7231565258007016e-01, 7.5958043070210701e-02,
            -1.7045648834524654e-02, 7.5962966753538221e-03,
            -4.3714707439979471e-03, 2.8859517440674581e-03,
            -2.0746650970216033e-03, 1.5785157470114936e-03,
            -1.2488838190209560e-03, 1.0152680830659427e-03,
            -8.4080306049046961e-04, 7.0476195609163082e-04,
            -5.9482916752352539e-04, 5.0336804667204028e-04,
            -4.2548447940956257e-04, 3.5796122082968956e-04,
            -2.9864161599898439e-04, 2.4605773793840048e-04,
            -1.0361846365053765e-04], dtype=float),
        19: jnp.array([
            8.7236999951964245e-01, 7.5927872240629551e-02,
            -1.7013935891760718e-02, 7.5656265115178452e-03,
            -4.3426213107755395e-03, 2.8594802969067116e-03,
            -2.0510362604427919e-03, 1.5581210743082249e-03,
            -1.2320427617758753e-03, 1.0022251313841430e-03,
            -8.3172470474742344e-04, 6.9973464073432479e-04,
            -5.9385867835347556e-04, 5.0638050140298073e-04,
            -4.3232896269473626e-04, 3.6841402825014334e-04,
            -3.1241201681421422e-04, 2.6279512314145257e-04,
            -2.1849983462646975e-04, 9.2806616155833853e-05], dtype=float),
        20: jnp.array([
            8.7238089185504586e-01, 7.5896813066217039e-02,
            -1.6985596884775269e-02, 7.5387062716141470e-03,
            -4.3173156578507734e-03, 2.8361114327190564e-03,
            -2.0299280302907086e-03, 1.5395685095462468e-03,
            -1.2162994369125798e-03, 9.8949598115445864e-04,
            -8.2216138134292758e-04, 6.9343255501454616e-04,
            -5.9085543459117820e-04, 5.0665560860022301e-04,
            -4.3580475651929868e-04, 3.7495774534949223e-04,
            -3.2183895450690501e-04, 2.7487280256993391e-04,
            -2.3295306883591545e-04, 1.9529420696674002e-04,
            -8.3596645493438584e-05], dtype=float),
        21: jnp.array([
            8.7242082177323477e-01, 7.5874601381641801e-02,
            -1.6962135414044184e-02, 7.5158205076777986e-03,
            -4.2955110374426845e-03, 2.8157410230346064e-03,
            -2.0112888549589400e-03, 1.5229178043762144e-03,
            -1.2018568200569545e-03, 9.7744233236910015e-04,
            -8.1263707830611241e-04, 6.8653590775308543e-04,
            -5.8664163898683756e-04, 5.0513635605394316e-04,
            -4.3694855773161029e-04, 3.7869098445211287e-04,
            -3.2804766308928859e-04, 2.8340512332804767e-04,
            -2.4362236246924825e-04, 2.0788274231868027e-04,
            -1.7559592725666228e-04, 7.5696973546588643e-05], dtype=float),
        22: jnp.array([
            8.7242895016736588e-01, 7.5851404850236531e-02,
            -1.6940879878094558e-02, 7.4954804807582463e-03,
            -4.2761823851407499e-03, 2.7976204613368570e-03,
            -1.9945824989892028e-03, 1.5078197955338899e-03,
            -1.1885409713512114e-03, 9.6605754764908459e-04,
            -8.0330434453524530e-04, 6.7934620678664049e-04,
            -5.8165455603034549e-04, 5.0237930038844129e-04,
            -4.3641655833503697e-04, 3.8034703762902377e-04,
            -3.3182361093520566e-04, 2.8920303827365825e-04,
            -2.5131640740119093e-04, 2.1732146930435599e-04,
            -1.8660497145197701e-04, 1.5871643631526562e-04,
            -6.8862628413863947e-05], dtype=float),
        23: jnp.array([
            8.7245914379788081e-01, 7.5834584057672294e-02,
            -1.6923048071897602e-02, 7.4779759103049716e-03,
            -4.2593486656233017e-03, 2.7816905742820146e-03,
            -1.9797527182815974e-03, 1.4942629361384024e-03,
            -1.1764088209681753e-03, 9.5548062137818164e-04,
            -7.9439094811981385e-04, 6.7218143285163723e-04,
            -5.7629944971810461e-04, 4.9887029710562461e-04,
            -4.3476524188678262e-04, 3.8054024076692155e-04,
            -3.3382386525448852e-04, 2.9294937017621518e-04,
            -2.5672546899496528e-04, 2.2428898659365308e-04,
            -1.9500748654685722e-04, 1.6841331547353097e-04,
            -1.4415792698833072e-04, 6.2916487058676165e-05], dtype=float),
        24: jnp.array([
            8.7246536950028220e-01, 7.5816806566238668e-02,
            -1.6906705700787391e-02, 7.4622514135075655e-03,
            -4.2442855384994460e-03, 2.7674127136869966e-03,
            -1.9663952025207516e-03, 1.4819563582778028e-03,
            -1.1652734792559317e-03, 9.4562347622962207e-04,
            -7.8590364956998430e-04, 6.6513890975441841e-04,
            -5.7075888477810678e-04, 4.9487042099542299e-04,
            -4.3232591898422011e-04, 3.7966233339187158e-04,
            -3.3448936777312466e-04, 2.9512180462729613e-04,
            -2.6035053230690072e-04, 2.2929543705052386e-04,
            -2.0130825443927662e-04, 1.7590682011087909e-04,
            -1.5272957961364562e-04, 1.3150348255857268e-04,
            -5.7706146465297619e-05], dtype=float),
        25: jnp.array([
            8.7248875196659104e-01, 7.5803765391887615e-02,
            -1.6892842415272138e-02, 7.4485771496326375e-03,
            -4.2310429512542420e-03, 2.7547610733581005e-03,
            -1.9544682029385644e-03, 1.4708728827450544e-03,
            -1.1551399197853461e-03, 9.3653389941369936e-04,
            -7.7793933860125061e-04, 6.5836775993287984e-04,
            -5.6523483129138790e-04, 4.9063297621223471e-04,
            -4.2939985637385807e-04, 3.7805753689235680e-04,
            -3.3420085575301537e-04, 2.9612993161527164e-04,
            -2.6262136742560100e-04, 2.3278131474373118e-04,
            -2.0594849025015860e-04, 1.8162858178584428e-04,
            -1.5944890159912006e-04, 1.3912640999964577e-04,
            -1.2044475842971024e-04, 5.3119023046254830e-05], dtype=float),
        26: jnp.array([
            8.7249362530848429e-01, 7.5789843024876430e-02,
            -1.6880012181469790e-02, 7.4361798866050205e-03,
            -4.2190941825087317e-03, 2.7433408779477366e-03,
            -1.9436674691052872e-03, 1.4607814243627409e-03,
            -1.1458422580764374e-03, 9.2810715541317285e-04,
            -7.7045190371279688e-04, 6.5187834206969738e-04,
            -5.5979173566712967e-04, 4.8627356649585946e-04,
            -4.2615017393026942e-04, 3.7593206834472630e-04,
            -3.3320244390024425e-04, 2.9624982622004778e-04,
            -2.6383941235569845e-04, 2.3506626719940909e-04,
            -2.0925846672497142e-04, 1.8591162418287637e-04,
            -1.6464365566885251e-04, 1.4516290704510429e-04,
            -1.2724529008227488e-04, 1.1071761169578391e-04,
            -4.9056308276636971e-05], dtype=float),
        27: jnp.array([
            8.7251210066417650e-01, 7.5779529521346076e-02,
            -1.6869024856588942e-02, 7.4253019522472295e-03,
            -4.2085025973400683e-03, 2.7331479262374778e-03,
            -1.9339667379907419e-03, 1.4516565887662154e-03,
            -1.1373691602801880e-03, 9.2035414673329486e-04,
            -7.6347965273381542e-04, 6.4573948331863717e-04,
            -5.5453049747617670e-04, 4.8192543735962960e-04,
            -4.2274163577301282e-04, 3.7348041149242112e-04,
            -3.3171567174031355e-04, 2.9572665363779966e-04,
            -2.6426935948910561e-04, 2.3642984817320333e-04,
            -2.1152747150299207e-04, 1.8904949036979184e-04,
            -1.6860592593073375e-04, 1.4989776677182335e-04,
            -1.3269417897076742e-04, 1.1681590208857079e-04,
            -1.0212297090357753e-04, 4.5443688746986498e-05], dtype=float),
        28: jnp.array([
            8.7251598657722484e-01, 7.5768423963066975e-02,
            -1.6858770452699619e-02, 7.4153607717001024e-03,
            -4.1988751600420919e-03, 2.7238871005104146e-03,
            -1.9251351272038342e-03, 1.4433172690205252e-03,
            -1.1295825061456220e-03, 9.1317631423928913e-04,
            -7.5696165453734870e-04, 6.3992653284702993e-04,
            -5.4946150782041344e-04, 4.7763263026652557e-04,
            -4.1925024212513946e-04, 3.7080845518074399e-04,
            -3.2987383320603683e-04, 2.9471821108671855e-04,
            -2.6409020629089271e-04, 2.3706858487739421e-04,
            -2.1296557052896117e-04, 1.9126153090844394e-04,
            -1.7155989601106889e-04, 1.5355542884195168e-04,
            -1.3701149474891369e-04, 1.2174351187400578e-04,
            -1.0760672201367076e-04, 9.4487030246761034e-05,
            -4.2214844585348015e-05], dtype=float),
        29: jnp.array([
            8.7253083736575510e-01, 7.5760127917557707e-02,
            -1.6849917232974677e-02, 7.4065698003132514e-03,
            -4.1902790740866039e-03, 2.7155672213687097e-03,
            -1.9171585866701777e-03, 1.4357442296619696e-03,
            -1.1224679769578239e-03, 9.0657055185326099e-04,
            -7.5091011604135776e-04, 6.3446967900078477e-04,
            -5.4463456676485098e-04, 4.7346536831066957e-04,
            -4.1576676370934151e-04, 3.6802704352577076e-04,
            -3.2780683804597873e-04, 2.9337198711939659e-04,
            -2.6346510429324673e-04, 2.3715899627428067e-04,
            -2.1376002531903562e-04, 1.9274284811961569e-04,
            -1.7370538522109262e-04, 1.5633714072789948e-04,
            -1.4039651810862980e-04, 1.2569431408703590e-04,
            -1.1208152947681375e-04, 9.9440244930514059e-05,
            -8.7676705058335925e-05, 3.9319132083084142e-05], dtype=float),
        30: jnp.array([
            8.7253398555054584e-01, 7.5751128032722340e-02,
            -1.6841594086663234e-02, 7.3984795175748392e-03,
            -4.1824142564461488e-03, 2.7079633721913449e-03,
            -1.9098597282310602e-03, 1.4287955234341908e-03,
            -1.1159132640054354e-03, 9.0045122303467301e-04,
            -7.4526444792308581e-04, 6.2933237244023048e-04,
            -5.4003639458428162e-04, 4.6943289505266571e-04,
            -4.1232212178557979e-04, 3.6518775882741876e-04,
            -3.2558570186938144e-04, 2.9177697290642225e-04,
            -2.6249932834437916e-04, 2.3682072176945632e-04,
            -2.1404270891300109e-04, 1.9363522818251908e-04,
            -1.7519161053303320e-04, 1.5839693814270154e-04,
            -1.4300539739772801e-04, 1.2882380787195038e-04,
            -1.1569946586710202e-04, 1.0351105020957766e-04,
            -9.2161733506772358e-05, 8.1573903555252171e-05,
            -3.6710705673823994e-05], dtype=float),
    }
    # fmt: on

    def __init__(
        self,
        n,
        conserve_dc=True,
        tol=None,
        gsparams=None,
    ):
        if tol is not None:
            from galsim.deprecated import depr

            depr("tol", 2.2, "gsparams=GSParams(kvalue_accuracy=tol)")
            gsparams = GSParams(kvalue_accuracy=tol)
        self._n = n
        self._conserve_dc = conserve_dc
        self._gsparams = GSParams.check(gsparams)

    @property
    def _K_arr(self):
        return self._K_arrs[self._n]

    @property
    def _C_arr(self):
        return self._C_arrs[self._n]

    @property
    def _du(self):
        return (
            self._gsparams.table_spacing
            * jnp.power(self._gsparams.kvalue_accuracy / 200.0, 0.25)
            / self._n
        )

    @property
    def _umax(self):
        return _find_umax_lanczos(
            self._du,
            self._n,
            self._conserve_dc,
            self._C_arr,
            self._gsparams.kvalue_accuracy,
        )

    def tree_flatten(self):
        """This function flattens the Interpolant into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        children = tuple()
        # Define auxiliary static data that doesn’t need to be traced
        aux_data = {
            "gsparams": self._gsparams,
            "n": self._n,
            "conserve_dc": self._conserve_dc,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flattened representation"""
        n = aux_data.pop("n")
        return cls(n, **aux_data)

    def __repr__(self):
        return "galsim.Lanczos(%r, %r, gsparams=%r)" % (
            self._n,
            self._conserve_dc,
            self._gsparams,
        )

    def __str__(self):
        return "galsim.Lanczos(%s)" % (self._n)

    # this is a pure function and we apply JIT ahead of time since this
    # one is pretty slow
    @jax.jit
    def _xval(x, n, conserve_dc, _K):
        x = jnp.abs(x)

        def _low(x, n):
            # from galsim
            # // res = n/(pi x)^2 * sin(pi x) * sin(pi x / n)
            # //     ~= (1 - 1/6 pix^2) * (1 - 1/6 pix^2 / n^2)
            # //     = 1 - 1/6 pix^2 ( 1 + 1/n^2 )
            # // For x < 1.e-4, the errors in this approximation are less than 1.e-16.
            pix = jnp.pi * x
            temp = (1.0 / 6.0) * pix * pix
            return 1.0 - temp * (1.0 + 1.0 / (n * n))

        def _high(x, n):
            pix = jnp.pi * x
            s = jnp.sin(pix)
            sn = jnp.sin(pix / n)
            return n * s * sn / (pix * pix)

        msk_s = x <= 1e-4
        msk_l = x <= n
        val = jnp.piecewise(
            x,
            [msk_s, (~msk_s) & msk_l],
            [_low, _high, lambda x, n: jnp.array(0, dtype=float)],
            n,
        )

        def _low_s(x, n):
            pix = jnp.pi * x
            temp = (1.0 / 6.0) * pix * pix
            return pix * (1.0 - temp)

        def _high_s(x, n):
            return jnp.sin(jnp.pi * x)

        def _dcval(val, x, n, _K):
            s = jnp.piecewise(
                x,
                [msk_s],
                [_low_s, _high_s],
                n,
            )
            ssq = s * s
            factor = (
                1.0
                - 4.0 * _K[1] * ssq
                - 16.0 * _K[2] * ssq * (1.0 - ssq)
                - 4.0 * _K[3] * ssq * (9.0 - ssq * (24.0 - 16.0 * ssq))
                - 64.0 * _K[4] * ssq * (1.0 - ssq * (5.0 - ssq * (8.0 - 4.0 * ssq)))
                - 4.0
                * _K[5]
                * ssq
                * (25.0 - ssq * (200.0 - ssq * (560.0 - ssq * (640.0 - 256.0 * ssq))))
            )
            val = val / factor

            return val

        def _no_dcval(val, x, n, _K):
            return val

        return jax.lax.cond(
            conserve_dc,
            _dcval,
            _no_dcval,
            val,
            x,
            n,
            _K,
        )

    def _xval_noraise(self, x):
        return Lanczos._xval(x, self._n, self._conserve_dc, self._K_arr)

    def _raw_uval(u, n):
        # this function is used in the init and so was causing a recursion depth error
        # when jitted, so I made it a pure function - MRB
        # from galsim
        # // F(u) = ( (vp+1) Si((vp+1)pi) - (vp-1) Si((vp-1)pi) +
        # //          (vm-1) Si((vm-1)pi) - (vm+1) Si((vm+1)pi) ) / 2pi
        vp = n * (2.0 * u + 1.0)
        vm = n * (2.0 * u - 1.0)
        retval = (
            (vm - 1.0) * si(jnp.pi * (vm - 1.0))
            - (vm + 1.0) * si(jnp.pi * (vm + 1.0))
            - (vp - 1.0) * si(jnp.pi * (vp - 1.0))
            + (vp + 1.0) * si(jnp.pi * (vp + 1.0))
        )
        return retval / (2.0 * jnp.pi)

    # this is a pure function and we apply JIT ahead of time since this
    # one is pretty slow
    @jax.jit
    def _uval(u, n, conserve_dc, _C):
        retval = Lanczos._raw_uval(u, n)

        def _dcval(retval, u, n, _C):
            retval *= _C[0]
            retval += _C[1] * (
                Lanczos._raw_uval(u + 1.0, n) + Lanczos._raw_uval(u - 1.0, n)
            )
            retval += _C[2] * (
                Lanczos._raw_uval(u + 2.0, n) + Lanczos._raw_uval(u - 2.0, n)
            )
            retval += _C[3] * (
                Lanczos._raw_uval(u + 3.0, n) + Lanczos._raw_uval(u - 3.0, n)
            )
            retval += _C[4] * (
                Lanczos._raw_uval(u + 4.0, n) + Lanczos._raw_uval(u - 4.0, n)
            )
            retval += _C[5] * (
                Lanczos._raw_uval(u + 5.0, n) + Lanczos._raw_uval(u - 5.0, n)
            )
            return retval

        def _no_dcval(retval, u, n, _C):
            return retval

        return jax.lax.cond(
            conserve_dc,
            _dcval,
            _no_dcval,
            retval,
            u,
            n,
            _C,
        )

    def _kval_noraise(self, k):
        return Lanczos._uval(k / 2.0 / jnp.pi, self._n, self._conserve_dc, self._C_arr)

    def urange(self):
        """The maximum extent of the interpolant in Fourier space (in 2pi/pixels)."""
        return self._umax

    @property
    def n(self):
        """The order of the Lanczos function."""
        return self._n

    @property
    def conserve_dc(self):
        """Whether this interpolant is modified to improve flux conservation."""
        return self._conserve_dc

    @property
    def xrange(self):
        """The maximum extent of the interpolant from the origin (in pixels)."""
        return self._n

    @property
    def ixrange(self):
        """The total integral range of the interpolant.  Typically 2 * xrange."""
        return 2 * self._n

    @property
    def positive_flux(self):
        """The positive-flux fraction of the interpolation kernel."""
        if self._conserve_dc:
            return self._posflux_conserve_dc[self._n]
        else:
            return self._posflux_no_conserve_dc[self._n]

    @property
    def negative_flux(self):
        """The negative-flux fraction of the interpolation kernel."""
        if self._conserve_dc:
            return self._negflux_conserve_dc[self._n]
        else:
            return self._negflux_no_conserve_dc[self._n]

    def unit_integrals(self, max_len=None):
        """Compute the unit integrals of the real-space kernel.

        integrals[i] = int(xval(x), i-0.5, i+0.5)

        Parameters:
            max_len:    The maximum length of the returned array. (ignored)

        Returns:
            integrals:  An array of unit integrals of length max_len or smaller.
        """
        if max_len is not None and max_len < self._n + 1:
            n = max_len
        else:
            n = self._n + 1
        if self._conserve_dc:
            return self._unit_integrals_conserve_dc[self._n][:n]
        else:
            return self._unit_integrals_no_conserve_dc[self._n][:n]


# we apply JIT here to esnure the class init is fast
@jax.jit
def _find_umax_lanczos(_du, n, conserve_dc, _C, kva):
    def _cond(vals):
        umax, u = vals
        return (u - umax < 1.0 / n) | (u < 1.1)

    def _body(vals):
        umax, u = vals
        uval = Lanczos._uval(u, n, conserve_dc, _C)
        umax = jax.lax.cond(
            jnp.abs(uval) > kva,
            lambda umax, u: u,
            lambda umax, u: umax,
            umax,
            u,
        )
        return [umax, u + _du]

    return jax.lax.while_loop(
        _cond,
        _body,
        [0.0, 0.0],
    )[0]


@jax.jit
def _compute_C_K_lanczos(n):
    _K = jnp.concatenate(
        (jnp.zeros(1), Lanczos._raw_uval(jnp.arange(5) + 1.0, n)), axis=0
    )
    _C = jnp.zeros(6)
    _C = _C.at[0].set(
        1.0
        + 2.0
        * (_K[1] * (1.0 + 3.0 * _K[1] + _K[2] + _K[3]) + _K[2] + _K[3] + _K[4] + _K[5])
    )
    _C = _C.at[1].set(-_K[1] * (1.0 + 4.0 * _K[1] + _K[2] + 2.0 * _K[3]))
    _C = _C.at[2].set(_K[1] * (_K[1] - 2.0 * _K[2] + _K[3]) - _K[2])
    _C = _C.at[3].set(_K[1] * (_K[1] - 2.0 * _K[3]) - _K[3])
    _C = _C.at[4].set(_K[1] * _K[3] - _K[4])
    _C = _C.at[5].set(-_K[5])

    return _C, _K


Lanczos._K_arrs = {}
Lanczos._C_arrs = {}
for n in range(1, 31):
    _C_arr, _K_arr = _compute_C_K_lanczos(n)
    Lanczos._K_arrs[n] = _K_arr
    Lanczos._C_arrs[n] = _C_arr
