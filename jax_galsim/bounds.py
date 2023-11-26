import galsim as _galsim
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.numpy.util import _wraps
from jax.tree_util import register_pytree_node_class

from jax_galsim.core.utils import (
    cast_to_float,
    cast_to_int,
    ensure_hashable,
    has_tracers,
)
from jax_galsim.errors import GalSimUndefinedBoundsError
from jax_galsim.position import Position, PositionD, PositionI


# The reason for avoid these tests is that they are not easy to do for jitted code.
@_wraps(
    _galsim.Bounds,
    lax_description="""\
"The JAX implementation of galsim.Bounds

  - will not always test for properly defined bounds, especially in jitted code
  - will not test whether BoundsI is indeed initialized with integers during vmap/jit/grad transforms
""",
)
@register_pytree_node_class
class Bounds(object):
    def __init__(self):
        raise NotImplementedError(
            "Cannot instantiate the base class. " "Use either BoundsD or BoundsI."
        )

    def _parse_args(self, *args, **kwargs):
        if len(kwargs) == 0:
            if len(args) == 4:
                self._isdefined = True
                self.xmin, self.xmax, self.ymin, self.ymax = args
            elif len(args) == 0:
                self._isdefined = False
                self.xmin = self.xmax = self.ymin = self.ymax = jnp.nan
            elif len(args) == 1:
                if isinstance(args[0], Bounds):
                    self._isdefined = True
                    self.xmin = args[0].xmin
                    self.xmax = args[0].xmax
                    self.ymin = args[0].ymin
                    self.ymax = args[0].ymax
                elif isinstance(args[0], Position):
                    self._isdefined = True
                    self.xmin = self.xmax = args[0].x
                    self.ymin = self.ymax = args[0].y
                else:
                    raise TypeError(
                        "Single argument to %s must be either a Bounds or a Position"
                        % (self.__class__.__name__)
                    )
                self._isdefined = True
            elif len(args) == 2:
                if isinstance(args[0], Position) and isinstance(args[1], Position):
                    self._isdefined = True
                    self.xmin = min(args[0].x, args[1].x)
                    self.xmax = max(args[0].x, args[1].x)
                    self.ymin = min(args[0].y, args[1].y)
                    self.ymax = max(args[0].y, args[1].y)
                else:
                    raise TypeError(
                        "Two arguments to %s must be Positions"
                        % (self.__class__.__name__)
                    )
            else:
                raise TypeError(
                    "%s takes either 1, 2, or 4 arguments (%d given)"
                    % (self.__class__.__name__, len(args))
                )
        elif len(args) != 0:
            raise TypeError(
                "Cannot provide both keyword and non-keyword arguments to %s"
                % (self.__class__.__name__)
            )
        else:
            try:
                self._isdefined = True
                self.xmin = kwargs.pop("xmin")
                self.xmax = kwargs.pop("xmax")
                self.ymin = kwargs.pop("ymin")
                self.ymax = kwargs.pop("ymax")
            except KeyError:
                raise TypeError(
                    "Keyword arguments, xmin, xmax, ymin, ymax are required for %s"
                    % (self.__class__.__name__)
                )
            if kwargs:
                raise TypeError("Got unexpected keyword arguments %s" % kwargs.keys())

    @_wraps(_galsim.Bounds.isDefined)
    def isDefined(self, _static=False):
        if _static:
            return (
                self._isdefined
                and np.all(self.xmin <= self.xmax)
                and np.all(self.ymin <= self.ymax)
            )
        else:
            return (
                jnp.isfinite(self.xmin)
                & jnp.isfinite(self.xmax)
                & jnp.isfinite(self.ymin)
                & jnp.isfinite(self.ymax)
                & (self.xmin <= self.xmax)
                & (self.ymin <= self.ymax)
            )

    def area(self):
        """Return the area of the enclosed region.

        The area is a bit different for integer-type `BoundsI` and float-type `BoundsD` instances.
        For floating point types, it is simply ``(xmax-xmin)*(ymax-ymin)``.  However, for integer
        types, we add 1 to each size to correctly count the number of pixels being described by the
        bounding box.
        """
        return self._area()

    def withBorder(self, dx, dy=None):
        """Return a new `Bounds` object that expands the current bounds by the specified width.

        If two arguments are given, then these are separate dx and dy borders.
        """
        self._check_scalar(dx, "dx")
        if dy is None:
            dy = dx
        else:
            self._check_scalar(dy, "dy")
        return self.__class__(
            self.xmin - dx, self.xmax + dx, self.ymin - dy, self.ymax + dy
        )

    @property
    def origin(self):
        "The lower left position of the `Bounds`."
        return self._pos_class(self.xmin, self.ymin)

    @property
    @_wraps(
        _galsim.Bounds.center,
        lax_description="The JAX implementation of galsim.Bounds.center does not raise for undefined bounds.",
    )
    def center(self):
        if not self.isDefined(_static=True):
            raise GalSimUndefinedBoundsError(
                "center is invalid for an undefined Bounds"
            )
        return self._center

    @property
    @_wraps(
        _galsim.Bounds.true_center,
        lax_description="The JAX implementation of galsim.Bounds.true_center does not raise for undefined bounds.",
    )
    def true_center(self):
        if not self.isDefined(_static=True):
            raise GalSimUndefinedBoundsError(
                "true_center is invalid for an undefined Bounds"
            )
        return PositionD((self.xmax + self.xmin) / 2.0, (self.ymax + self.ymin) / 2.0)

    @_wraps(_galsim.Bounds.includes)
    def includes(self, *args, _static=False):
        if len(args) == 1:
            if isinstance(args[0], Bounds):
                b = args[0]
                return (
                    self.isDefined(_static=_static)
                    & b.isDefined(_static=_static)
                    & (self.xmin <= b.xmin)
                    & (self.xmax >= b.xmax)
                    & (self.ymin <= b.ymin)
                    & (self.ymax >= b.ymax)
                )
            elif isinstance(args[0], Position):
                p = args[0]
                return (
                    self.isDefined(_static=_static)
                    & (self.xmin <= p.x)
                    & (p.x <= self.xmax)
                    & (self.ymin <= p.y)
                    & (p.y <= self.ymax)
                )
            else:
                raise TypeError("Invalid argument %s" % args[0])
        elif len(args) == 2:
            x, y = args
            return (
                self.isDefined(_static=_static)
                & (self.xmin <= x)
                & (x <= self.xmax)
                & (self.ymin <= y)
                & (y <= self.ymax)
            )
        elif len(args) == 0:
            raise TypeError("include takes at least 1 argument (0 given)")
        else:
            raise TypeError("include takes at most 2 arguments (%d given)" % len(args))

    @_wraps(_galsim.Bounds.expand)
    def expand(self, factor_x, factor_y=None):
        if factor_y is None:
            factor_y = factor_x
        dx = (self.xmax - self.xmin) * 0.5 * (factor_x - 1.0)
        dy = (self.ymax - self.ymin) * 0.5 * (factor_y - 1.0)
        if isinstance(self, BoundsI):
            dx = jnp.ceil(dx)
            dy = jnp.ceil(dy)
        return self.withBorder(dx, dy)

    def getXMin(self):
        "Get the value of xmin."
        return self.xmin

    def getXMax(self):
        "Get the value of xmax."
        return self.xmax

    def getYMin(self):
        "Get the value of ymin."
        return self.ymin

    def getYMax(self):
        "Get the value of ymax."
        return self.ymax

    def shift(self, delta):
        """Shift the `Bounds` instance by a supplied `Position`.

        Examples:

        The shift method takes either a `PositionI` or `PositionD` instance, which must match
        the type of the `Bounds` instance::

            >>> bounds = BoundsI(1,32,1,32)
            >>> bounds = bounds.shift(galsim.PositionI(3, 2))
            >>> bounds = BoundsD(0, 37.4, 0, 49.9)
            >>> bounds = bounds.shift(galsim.PositionD(3.9, 2.1))
        """
        if not isinstance(delta, self._pos_class):
            raise TypeError("delta must be a %s instance" % self._pos_class)
        return self.__class__(
            self.xmin + delta.x,
            self.xmax + delta.x,
            self.ymin + delta.y,
            self.ymax + delta.y,
        )

    def __and__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError("other must be a %s instance" % self.__class__.__name__)
        # NaNs always propagate, so if either is undefined, the result is undefined
        return self.__class__(
            jnp.maximum(self.xmin, other.xmin),
            jnp.minimum(self.xmax, other.xmax),
            jnp.maximum(self.ymin, other.ymin),
            jnp.minimum(self.ymax, other.ymax),
        )

    def __add__(self, other):
        if isinstance(other, self.__class__):
            # galsim logic is
            # if not other.isDefined():
            #     return self
            # elif self.isDefined():
            #     xmin = jnp.minimum(self.xmin, other.xmin)
            #     xmax = jnp.maximum(self.xmax, other.xmax)
            #     ymin = jnp.minimum(self.ymin, other.ymin)
            #     ymax = jnp.maximum(self.ymax, other.ymax)
            #     return self.__class__(xmin, xmax, ymin, ymax)
            # else:
            #     return other
            return self.__class__(
                jax.lax.cond(
                    ~jnp.any(other.isDefined()),
                    lambda: BoundsD(self),
                    lambda: BoundsD(
                        jax.lax.cond(
                            jnp.any(self.isDefined()),
                            lambda: BoundsD(
                                jnp.minimum(self.xmin, other.xmin),
                                jnp.maximum(self.xmax, other.xmax),
                                jnp.minimum(self.ymin, other.ymin),
                                jnp.maximum(self.ymax, other.ymax),
                            ),
                            lambda: BoundsD(other),
                        )
                    ),
                )
            )
        elif isinstance(other, self._pos_class):
            # the galsim logic is
            # if self.isDefined():
            #     xmin = jnp.minimum(self.xmin, other.x)
            #     xmax = jnp.maximum(self.xmax, other.x)
            #     ymin = jnp.minimum(self.ymin, other.y)
            #     ymax = jnp.maximum(self.ymax, other.y)
            #     return self.__class__(xmin, xmax, ymin, ymax)
            # else:
            #     return self.__class__(other)
            return self.__class__(
                jax.lax.cond(
                    jnp.any(self.isDefined()),
                    lambda: BoundsD(
                        jnp.minimum(self.xmin, other.x),
                        jnp.maximum(self.xmax, other.x),
                        jnp.minimum(self.ymin, other.y),
                        jnp.maximum(self.ymax, other.y),
                    ),
                    lambda: BoundsD(other),
                )
            )
        else:
            raise TypeError(
                "other must be either a %s or a %s"
                % (self.__class__.__name__, self._pos_class.__name__)
            )

    def __repr__(self):
        if self.isDefined(_static=True):
            return "galsim.%s(xmin=%r, xmax=%r, ymin=%r, ymax=%r)" % (
                self.__class__.__name__,
                ensure_hashable(self.xmin),
                ensure_hashable(self.xmax),
                ensure_hashable(self.ymin),
                ensure_hashable(self.ymax),
            )
        else:
            return "galsim.%s()" % (self.__class__.__name__)

    def __str__(self):
        if self.isDefined(_static=True):
            return "galsim.%s(%s,%s,%s,%s)" % (
                self.__class__.__name__,
                ensure_hashable(self.xmin),
                ensure_hashable(self.xmax),
                ensure_hashable(self.ymin),
                ensure_hashable(self.ymax),
            )
        else:
            return "galsim.%s()" % (self.__class__.__name__)

    def __hash__(self):
        return hash(
            (
                self.__class__.__name__,
                ensure_hashable(self.xmin),
                ensure_hashable(self.xmax),
                ensure_hashable(self.ymin),
                ensure_hashable(self.ymax),
            )
        )

    def _getinitargs(self):
        if self.isDefined(_static=True):
            return (self.xmin, self.xmax, self.ymin, self.ymax)
        else:
            return ()

    def __eq__(self, other):
        return self is other or (
            isinstance(other, self.__class__)
            and (
                (
                    np.array_equal(self.xmin, other.xmin, equal_nan=True)
                    and np.array_equal(self.xmax, other.xmax, equal_nan=True)
                    and np.array_equal(self.ymin, other.ymin, equal_nan=True)
                    and np.array_equal(self.ymax, other.ymax, equal_nan=True)
                )
                or (
                    (not self.isDefined(_static=True))
                    and (not other.isDefined(_static=True))
                )
            )
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def tree_flatten(self):
        """This function flattens the Bounds into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        children = (self.xmin, self.xmax, self.ymin, self.ymax)
        # Define auxiliary static data that doesn’t need to be traced
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        return cls(*children)

    @classmethod
    def from_galsim(cls, galsim_bounds):
        """Create a jax_galsim `BoundsD/I` from a `galsim.BoundsD/I` object."""
        if isinstance(galsim_bounds, _galsim.BoundsD):
            _cls = BoundsD
        elif isinstance(galsim_bounds, _galsim.BoundsI):
            _cls = BoundsI
        else:
            raise TypeError(
                "galsim_bounds must be either a %s or a %s"
                % (_galsim.BoundsD.__name__, _galsim.BoundsI.__name__)
            )
        if not galsim_bounds.isDefined():
            return _cls()
        else:
            return _cls(
                galsim_bounds.xmin,
                galsim_bounds.xmax,
                galsim_bounds.ymin,
                galsim_bounds.ymax,
            )


@_wraps(
    _galsim.BoundsD,
    lax_description="The JAX implementation of galsim.BoundsD does not always check for float values.",
)
@register_pytree_node_class
class BoundsD(Bounds):
    _pos_class = PositionD

    def __init__(self, *args, **kwargs):
        self._parse_args(*args, **kwargs)
        self.xmin = cast_to_float(self.xmin)
        self.xmax = cast_to_float(self.xmax)
        self.ymin = cast_to_float(self.ymin)
        self.ymax = cast_to_float(self.ymax)

    def _check_scalar(self, x, name):
        try:
            if (
                isinstance(x, jax.Array)
                and x.shape == ()
                and x.dtype.name in ["float32", "float64", "float"]
            ):
                return
            elif x == float(x):
                return
        except (TypeError, ValueError):
            pass
        raise TypeError("%s must be a float value" % name)

    def _area(self):
        return jax.lax.cond(
            jnp.any(self.isDefined()),
            lambda xmin, xmax, ymin, ymax: (xmax - xmin) * (ymax - ymin),
            lambda xmin, xmax, ymin, ymax: jnp.zeros_like(xmin),
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
        )

    @property
    def _center(self):
        return PositionD((self.xmax + self.xmin) / 2.0, (self.ymax + self.ymin) / 2.0)


@_wraps(
    _galsim.BoundsI,
    lax_description="The JAX implementation of galsim.BoundsI does not always check for integer values.",
)
@register_pytree_node_class
class BoundsI(Bounds):
    _pos_class = PositionI

    def __init__(self, *args, **kwargs):
        self._parse_args(*args, **kwargs)

        # best-effort error checking
        raise_notint = False
        try:
            bnds = (self.xmin, self.xmax, self.ymin, self.ymax)
            if not has_tracers(bnds) and np.all(np.isfinite(bnds)) & np.any(
                (self.xmin != np.floor(self.xmin))
                | (self.xmax != np.floor(self.xmax))
                | (self.ymin != np.floor(self.ymin))
                | (self.ymax != np.floor(self.ymax))
            ):
                raise_notint = True
        except Exception:
            pass

        if raise_notint:
            raise TypeError("BoundsI must be initialized with integer values")

        self.xmin = cast_to_int(self.xmin)
        self.xmax = cast_to_int(self.xmax)
        self.ymin = cast_to_int(self.ymin)
        self.ymax = cast_to_int(self.ymax)

    def _check_scalar(self, x, name):
        try:
            if (
                isinstance(x, jax.Array)
                and x.shape == ()
                and x.dtype.name in ["int32", "int64", "int"]
            ):
                return
            elif x == int(x):
                return
        except (TypeError, ValueError):
            pass
        raise TypeError("%s must be an integer value" % name)

    def numpyShape(self, _static=False):
        "A simple utility function to get the numpy shape that corresponds to this `Bounds` object."
        if _static:
            if self.isDefined(_static=True):
                return (self.ymax - self.ymin + 1, self.xmax - self.xmin + 1)
            else:
                return (0, 0)
        else:
            return jax.lax.cond(
                jnp.any(self.isDefined()),
                lambda xmin, xmax, ymin, ymax: (ymax - ymin + 1, xmax - xmin + 1),
                lambda xmin, xmax, ymin, ymax: (
                    jnp.zeros_like(xmin),
                    jnp.zeros_like(xmin),
                ),
                self.xmin,
                self.xmax,
                self.ymin,
                self.ymax,
            )

    def _area(self):
        # Remember the + 1 this time to include the pixels on both edges of the bounds.
        return jax.lax.cond(
            jnp.any(self.isDefined()),
            lambda xmin, xmax, ymin, ymax: (xmax - xmin + 1) * (ymax - ymin + 1),
            lambda xmin, xmax, ymin, ymax: jnp.zeros_like(xmin),
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
        )

    @property
    def _center(self):
        # Write it this way to make sure the integer rounding goes the same way regardless
        # of whether the values are positive or negative.
        # e.g. (1,10,1,10) -> (6,6)
        #      (-10,-1,-10,-1) -> (-5,-5)
        # Just up and to the right of the true center in both cases.
        return PositionI(
            self.xmin + (self.xmax - self.xmin + 1) // 2,
            self.ymin + (self.ymax - self.ymin + 1) // 2,
        )
