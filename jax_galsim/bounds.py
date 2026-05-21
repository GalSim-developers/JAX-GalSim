import galsim as _galsim
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from jax_galsim.core.utils import (
    cast_to_float,
    cast_to_int,
    check_is_int_then_cast,
    ensure_hashable,
    implements,
)
from jax_galsim.position import Position, PositionD, PositionI

BOUNDS_LAX_DESCR = """\
The JAX implementation

- will not always test whether the bounds are valid

Further, the JAX implementation adds a new method, ``isStatic`` to the
``BoundsI`` class. If JAX-GalSim detects that the ``BoundsI`` instance
has been instantiated with static, known values, ``isStatic()`` will
return ``True``.

``BoundsI`` objects in JAX-Galsim support an additional initialization
call ``BoundsI(xmin=..., deltax=..., ymin=..., deltay=...)``. In this case,
the values for ``deltax/y`` indicate the width of the bounds and must be
static constants.

When calling ``jax.vmap`` over ``BoundsI`` objects, only ``x/ymin``
are vectorized over. This restriction allows for code that renders
objects in fixed sized stamps with variable locations, a common
operation. ``BoundsI`` objects which are static (i.e., ``isStatic()``
returns ``True``) are treated as constants with respect to ``vmap``,
``jit``, and other JAX transforms.
"""


@implements(_galsim.Bounds, lax_description=BOUNDS_LAX_DESCR)
@register_pytree_node_class
class Bounds:
    def __init__(self):
        raise NotImplementedError(
            "Cannot instantiate the base class. Use either BoundsD or BoundsI."
        )

    def _parse_args(self, *args, **kwargs):
        if len(kwargs) == 0:
            if len(args) == 4:
                self._isdefined = True
                self.xmin, self.xmax, self.ymin, self.ymax = args
            elif len(args) == 0:
                self._isdefined = False
                self.xmin = 0
                self.ymin = 0
                self.deltax = 0
                self.deltay = 0
            elif len(args) == 1:
                if isinstance(args[0], Bounds):
                    if isinstance(self, BoundsI) and isinstance(args[0], BoundsD):
                        offset = 1
                    elif isinstance(self, BoundsD) and isinstance(args[0], BoundsI):
                        offset = -1
                    else:
                        offset = 0
                    self._isdefined = args[0]._isdefined
                    self.xmin = args[0].xmin
                    self.deltax = args[0].deltax + offset
                    self.ymin = args[0].ymin
                    self.deltay = args[0].deltay + offset
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
                self.ymin = kwargs.pop("ymin")
            except KeyError:
                raise TypeError(
                    "Keyword arguments, xmin, ymin are required for %s"
                    % (self.__class__.__name__)
                )

            if "xmax" in kwargs and "ymax" in kwargs:
                self.xmax = kwargs.pop("xmax")
                self.ymax = kwargs.pop("ymax")
            elif "deltax" in kwargs and "deltay" in kwargs:
                self.deltax = kwargs.pop("deltax")
                self.deltay = kwargs.pop("deltay")
            else:
                raise TypeError(
                    "Keyword arguments, either (xmax, ymax) "
                    "or (deltax, deltay) are required for %s"
                    % (self.__class__.__name__)
                )

            if kwargs:
                raise TypeError("Got unexpected keyword arguments %s" % kwargs.keys())

        # for simple inputs, we can check if the bounds are valid
        if isinstance(self, BoundsD):
            max_delta = 0
        else:
            max_delta = 1
        if (
            isinstance(self.deltax, (int, float, np.integer, np.floating))
            and isinstance(self.deltay, (int, float, np.integer, np.floating))
            and (self.deltax < max_delta or self.deltay < max_delta)
        ):
            self._isdefined = False

    @implements(_galsim.Bounds.area)
    def area(self):
        return self._area()

    @implements(_galsim.Bounds.withBorder)
    def withBorder(self, dx, dy=None):
        self._check_scalar(dx, "dx")
        if dy is None:
            dy = dx
        else:
            self._check_scalar(dy, "dy")
        return self.__class__(
            xmin=self.xmin - dx,
            deltax=self.deltax + 2 * dx,
            ymin=self.ymin - dy,
            deltay=self.deltay + 2 * dy,
        )

    @property
    @implements(_galsim.Bounds.origin)
    def origin(self):
        return self._pos_class(self.xmin, self.ymin)

    @property
    @implements(_galsim.Bounds.center)
    def center(self):
        if not self.isDefined():
            raise _galsim.GalSimUndefinedBoundsError(
                "center is invalid for an undefined Bounds"
            )
        return self._center

    @property
    @implements(_galsim.Bounds.true_center)
    def true_center(self):
        if not self.isDefined():
            raise _galsim.GalSimUndefinedBoundsError(
                "true_center is invalid for an undefined Bounds"
            )
        return PositionD((self.xmax + self.xmin) / 2.0, (self.ymax + self.ymin) / 2.0)

    @implements(_galsim.Bounds.includes)
    def includes(self, *args):
        if len(args) == 1:
            if isinstance(args[0], Bounds):
                b = args[0]
                return (
                    self.isDefined()
                    & b.isDefined()
                    & (self.xmin <= b.xmin)
                    & (self.xmax >= b.xmax)
                    & (self.ymin <= b.ymin)
                    & (self.ymax >= b.ymax)
                )
            elif isinstance(args[0], Position):
                p = args[0]
                return (
                    self.isDefined()
                    & (self.xmin <= p.x)
                    & (self.ymin <= p.y)
                    & (p.x <= self.xmax)
                    & (p.y <= self.ymax)
                )
            else:
                raise TypeError("Invalid argument %s" % args[0])
        elif len(args) == 2:
            x, y = args
            return (
                self.isDefined()
                & (self.xmin <= float(x))
                & (self.ymin <= float(y))
                & (float(x) <= self.xmax)
                & (float(y) <= self.ymax)
            )
        elif len(args) == 0:
            raise TypeError("include takes at least 1 argument (0 given)")
        else:
            raise TypeError("include takes at most 2 arguments (%d given)" % len(args))

    @implements(_galsim.Bounds.expand)
    def expand(self, factor_x, factor_y=None):
        if factor_y is None:
            factor_y = factor_x
        dx = (self.xmax - self.xmin) * 0.5 * (factor_x - 1.0)
        dy = (self.ymax - self.ymin) * 0.5 * (factor_y - 1.0)
        if isinstance(self, BoundsI):
            dx = jnp.ceil(dx)
            dy = jnp.ceil(dy)
        return self.withBorder(dx, dy)

    @implements(_galsim.Bounds.isDefined)
    def isDefined(self):
        return self._isdefined

    @implements(_galsim.Bounds.getXMin)
    def getXMin(self):
        return self.xmin

    @implements(_galsim.Bounds.getXMax)
    def getXMax(self):
        return self.xmax

    @implements(_galsim.Bounds.getYMin)
    def getYMin(self):
        return self.ymin

    @implements(_galsim.Bounds.getYMax)
    def getYMax(self):
        return self.ymax

    @implements(_galsim.Bounds.shift)
    def shift(self, delta):
        if not isinstance(delta, self._pos_class):
            raise TypeError("delta must be a %s instance" % self._pos_class)
        return self.__class__(
            xmin=self.xmin + delta.x,
            deltax=self.deltax,
            ymin=self.ymin + delta.y,
            deltay=self.deltay,
        )

    def __and__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError("other must be a %s instance" % self.__class__.__name__)
        if not self.isDefined() or not other.isDefined():
            return self.__class__()
        else:
            xmin = jnp.maximum(self.xmin, other.xmin)
            xmax = jnp.minimum(self.xmax, other.xmax)
            ymin = jnp.maximum(self.ymin, other.ymin)
            ymax = jnp.minimum(self.ymax, other.ymax)
            if xmin > xmax or ymin > ymax:
                return self.__class__()
            else:
                return self.__class__(xmin, xmax, ymin, ymax)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            if not other.isDefined():
                return self
            elif self.isDefined():
                xmin = jnp.minimum(self.xmin, other.xmin)
                xmax = jnp.maximum(self.xmax, other.xmax)
                ymin = jnp.minimum(self.ymin, other.ymin)
                ymax = jnp.maximum(self.ymax, other.ymax)
                return self.__class__(xmin, xmax, ymin, ymax)
            else:
                return other
        elif isinstance(other, self._pos_class):
            if self.isDefined():
                xmin = jnp.minimum(self.xmin, other.x)
                xmax = jnp.maximum(self.xmax, other.x)
                ymin = jnp.minimum(self.ymin, other.y)
                ymax = jnp.maximum(self.ymax, other.y)
                return self.__class__(xmin, xmax, ymin, ymax)
            else:
                return self.__class__(other)
        else:
            raise TypeError(
                "other must be either a %s or a %s"
                % (self.__class__.__name__, self._pos_class.__name__)
            )

    def _getinitargs(self):
        if self.isDefined():
            return (self.xmin, self.xmax, self.ymin, self.ymax)
        else:
            return ()

    def __eq__(self, other):
        return self is other or (
            isinstance(other, self.__class__)
            and self._getinitargs() == other._getinitargs()
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(
            (
                self.__class__.__name__,
                ensure_hashable(self.xmin),
                ensure_hashable(self.deltax),
                ensure_hashable(self.ymin),
                ensure_hashable(self.deltay),
            )
        )

    def tree_flatten(self):
        """This function flattens the Bounds into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        if self.isDefined():
            children = (self.xmin, self.deltax, self.ymin, self.deltay)
        else:
            children = tuple()
        # Define auxiliary static data that doesn’t need to be traced
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        if children:
            return cls(
                xmin=children[0],
                deltax=children[1],
                ymin=children[2],
                deltay=children[3],
            )
        else:
            return cls()

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
        if galsim_bounds.isDefined():
            return _cls(
                galsim_bounds.xmin,
                galsim_bounds.xmax,
                galsim_bounds.ymin,
                galsim_bounds.ymax,
            )
        else:
            return _cls()

    def to_galsim(self):
        """Create a galsim `BoundsD/I` from a `jax_galsim.BoundsD/I` object."""
        if isinstance(self, BoundsI):
            gs_class = _galsim.bounds.BoundsI
            cast = int
        else:
            gs_class = _galsim.bounds.BoundsD
            cast = float

        if self.isDefined():
            return gs_class(
                cast(self.xmin),
                cast(self.xmax),
                cast(self.ymin),
                cast(self.ymax),
            )
        else:
            return gs_class()

    def isStatic(self):
        """Returns ``True`` if the ``BoundsI`` instance
        has static, known dimensions and location. Always returns
        ``False`` for ``BoundsD``."""
        return self._isstatic


@implements(_galsim.BoundsD, lax_description=BOUNDS_LAX_DESCR)
@register_pytree_node_class
class BoundsD(Bounds):
    _pos_class = PositionD

    def __init__(self, *args, **kwargs):
        self._isstatic = False
        self._parse_args(*args, **kwargs)
        self.xmin = cast_to_float(self.xmin)
        self.deltax = cast_to_float(self.deltax)
        self.ymin = cast_to_float(self.ymin)
        self.deltay = cast_to_float(self.deltay)

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

    @property
    def xmax(self):
        return self.xmin + self.deltax

    @xmax.setter
    def xmax(self, value):
        self.deltax = value - self.xmin

    @property
    def ymax(self):
        return self.ymin + self.deltay

    @ymax.setter
    def ymax(self, value):
        self.deltay = value - self.ymin

    def _area(self):
        return self.deltax * self.deltay

    @property
    def _center(self):
        return PositionD((self.xmax + self.xmin) / 2.0, (self.ymax + self.ymin) / 2.0)

    def __repr__(self):
        if self.isDefined():
            return "galsim.%s(%r, %r, %r, %r)" % (
                self.__class__.__name__,
                ensure_hashable(self.xmin),
                ensure_hashable(self.xmax),
                ensure_hashable(self.ymin),
                ensure_hashable(self.ymax),
            )
        else:
            return "galsim.%s()" % (self.__class__.__name__)

    def __str__(self):
        if self.isDefined():
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
                ensure_hashable(self.deltax),
                ensure_hashable(self.ymin),
                ensure_hashable(self.deltay),
            )
        )


@implements(_galsim.BoundsI, lax_description=BOUNDS_LAX_DESCR)
@register_pytree_node_class
class BoundsI(Bounds):
    _pos_class = PositionI

    def __init__(self, *args, **kwargs):
        # initial setting to let stuff pass through freely
        self._isstatic = True

        self._parse_args(*args, **kwargs)

        self.deltax = cast_to_float(self.deltax)
        self.deltay = cast_to_float(self.deltay)
        if (self.deltax != int(self.deltax)) or (self.deltay != int(self.deltay)):
            raise TypeError("BoundsI must be initialized with integer values")
        self.deltax = cast_to_int(self.deltax)
        self.deltay = cast_to_int(self.deltay)

        if not (
            isinstance(
                self._xmin,
                (int, float, np.floating, np.integer),
            )
            and isinstance(
                self._ymin,
                (int, float, np.floating, np.integer),
            )
        ):
            self._isstatic = False

        # validate inputs are ints
        self._xmin = check_is_int_then_cast(
            self._xmin, "BoundsI must be initialized with integer values"
        )
        self._ymin = check_is_int_then_cast(
            self._ymin, "BoundsI must be initialized with integer values"
        )

        if self.deltax < 1 and self.deltay < 1:
            self._isdefined = False

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

    def numpyShape(self):
        "A simple utility function to get the numpy shape that corresponds to this `Bounds` object."
        if self.isDefined():
            return self.deltay, self.deltax
        else:
            return 0, 0

    @property
    def xmin(self):
        if self._isstatic:
            return self._xmin
        else:
            return jnp.astype(self._xmin, jnp.int_)

    @xmin.setter
    def xmin(self, value):
        if self._isstatic:
            self._xmin = value
        else:
            self._xmin = jnp.astype(value, jnp.float_)

    @property
    def xmax(self):
        return self.xmin + self.deltax - 1

    @xmax.setter
    def xmax(self, value):
        self.deltax = value - self.xmin + 1

    @property
    def ymin(self):
        if self._isstatic:
            return self._ymin
        else:
            return jnp.astype(self._ymin, jnp.int_)

    @ymin.setter
    def ymin(self, value):
        if self._isstatic:
            self._ymin = value
        else:
            self._ymin = jnp.astype(value, jnp.float_)

    @property
    def ymax(self):
        return self.ymin + self.deltay - 1

    @ymax.setter
    def ymax(self, value):
        self.deltay = value - self.ymin + 1

    def _area(self):
        # Remember the + 1 this time to include the pixels on both edges of the bounds.
        if not self.isDefined():
            return 0
        else:
            return self.deltax * self.deltay

    @property
    def _center(self):
        # Write it this way to make sure the integer rounding goes the same way regardless
        # of whether the values are positive or negative.
        # e.g. (1,10,1,10) -> (6,6)
        #      (-10,-1,-10,-1) -> (-5,-5)
        # Just up and to the right of the true center in both cases.
        return PositionI(
            self.xmin + self.deltax // 2,
            self.ymin + self.deltay // 2,
        )

    def tree_flatten(self):
        """This function flattens the Bounds into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        if self.isDefined():
            if self._isstatic:
                # Define the children nodes of the PyTree that need tracing
                children = tuple()

                # Define auxiliary static data that doesn’t need to be traced
                aux_data = {
                    "xmin": self._xmin,
                    "ymin": self._ymin,
                    "deltax": self.deltax,
                    "deltay": self.deltay,
                }
            else:
                children = (self._xmin, self._ymin)
                # Define auxiliary static data that doesn’t need to be traced
                aux_data = {"deltax": self.deltax, "deltay": self.deltay}
        else:
            children = tuple()
            aux_data = None

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        if aux_data is not None:
            ret = cls.__new__(cls)
            if "xmin" in aux_data and "ymin" in aux_data:
                ret._isstatic = True
                ret._xmin = aux_data["xmin"]
                ret._ymin = aux_data["ymin"]
            else:
                ret._isstatic = False
                ret._xmin = children[0]
                ret._ymin = children[1]
            ret.deltax = aux_data["deltax"]
            ret.deltay = aux_data["deltay"]
            if ret.deltax < 1 and ret.deltay < 1:
                ret._isdefined = False
            else:
                ret._isdefined = True
        else:
            ret = cls()

        return ret

    def __repr__(self):
        if self.isDefined():
            return "galsim.%s(xmin=%r, deltax=%r, ymin=%r, deltay=%r)" % (
                self.__class__.__name__,
                ensure_hashable(self.xmin),
                ensure_hashable(self.deltax),
                ensure_hashable(self.ymin),
                ensure_hashable(self.deltay),
            )
        else:
            return "galsim.%s()" % (self.__class__.__name__)

    def __str__(self):
        if self.isDefined():
            return "galsim.%s(xmin=%s, deltax=%s, ymin=%s, deltay=%s)" % (
                self.__class__.__name__,
                ensure_hashable(self.xmin),
                ensure_hashable(self.deltax),
                ensure_hashable(self.ymin),
                ensure_hashable(self.deltay),
            )
        else:
            return "galsim.%s()" % (self.__class__.__name__)

    def _getinitargs(self):
        if self.isDefined():
            return (self.xmin, self.deltax, self.ymin, self.deltay)
        else:
            return ()

    def __eq__(self, other):
        return self is other or (
            isinstance(other, BoundsI) and self._getinitargs() == other._getinitargs()
        )

    def __hash__(self):
        return hash(
            (
                self.__class__.__name__,
                ensure_hashable(self.xmin),
                ensure_hashable(self.deltax),
                ensure_hashable(self.ymin),
                ensure_hashable(self.deltay),
            )
        )
