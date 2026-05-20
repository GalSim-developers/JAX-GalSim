import equinox
import galsim as _galsim
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from jax_galsim.core.utils import (
    STATIC_SCALAR_TYPES,
    cast_to_float,
    check_is_int_then_cast,
    ensure_hashable,
    implements,
)
from jax_galsim.position import Position, PositionD, PositionI

BOUNDS_LAX_DESCR = """\
The JAX implementation adds a new method, ``isStatic`` to the
``Bounds`` class. If JAX-GalSim detects that a ``BoundsI`` instance
has been instantiated with static, known values, ``isStatic()`` will
return ``True``, otherwise it is ``False``. For ``BoundsD``, ``isStatic()``
always returns ``False``.

``BoundsI`` objects in JAX-Galsim must have a fixed width. To help support
this requirement, JAX-Galsim supports an additional initialization call
``BoundsI(xmin=..., deltax=..., ymin=..., deltay=...)``.

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
        do_isdefined = True

        if len(kwargs) == 0:
            if len(args) == 4:
                self.xmin, self.xmax, self.ymin, self.ymax = args
            elif len(args) == 0:
                do_isdefined = False
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
                    self.xmin = self.xmax = args[0].x
                    self.ymin = self.ymax = args[0].y
                else:
                    raise TypeError(
                        "Single argument to %s must be either a Bounds or a Position"
                        % (self.__class__.__name__)
                    )
            elif len(args) == 2:
                if isinstance(args[0], Position) and isinstance(args[1], Position):
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

        return do_isdefined

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
        if not isinstance(self._isdefined, jnp.ndarray):
            if not self.isDefined():
                raise _galsim.GalSimUndefinedBoundsError(
                    "center is invalid for an undefined Bounds"
                )
        else:
            self._isdefined = equinox.error_if(
                self._isdefined,
                jnp.any(~self._isdefined),
                "center is invalid for an undefined Bounds",
            )
        return self._center

    @property
    @implements(_galsim.Bounds.true_center)
    def true_center(self):
        if not isinstance(self._isdefined, jnp.ndarray):
            if not self.isDefined():
                raise _galsim.GalSimUndefinedBoundsError(
                    "true_center is invalid for an undefined Bounds"
                )
        else:
            self._isdefined = equinox.error_if(
                self._isdefined,
                jnp.any(~self._isdefined),
                "true_center is invalid for an undefined Bounds",
            )
        return PositionD((self.xmax + self.xmin) / 2.0, (self.ymax + self.ymin) / 2.0)

    @implements(_galsim.Bounds.includes)
    def includes(self, *args):
        raise NotImplementedError(
            "Subclasses of `Bounds` must implement the `includes` method!"
        )

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
        raise NotImplementedError(
            "Subclasses of `Bounds` must implement the `__and__` method!"
        )

    def __add__(self, other):
        raise NotImplementedError(
            "Subclasses of `Bounds` must implement the `__add__` method!"
        )

    def __eq__(self, other):
        raise NotImplementedError(
            "Subclasses of `Bounds` must implement the `__eq__` method!"
        )

    def __ne__(self, other):
        raise NotImplementedError(
            "Subclasses of `Bounds` must implement the `__ne__` method!"
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

    def tree_flatten(self):
        """This function flattens the Bounds into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        children = (self.xmin, self.deltax, self.ymin, self.deltay, self._isdefined)
        # Define auxiliary static data that doesn’t need to be traced
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        ret = cls.__new__(cls)
        ret.xmin = children[0]
        ret.deltax = children[1]
        ret.ymin = children[2]
        ret.deltay = children[3]
        ret._isdefined = children[4]
        ret._isstatic = False
        ret._isstaticshape = False

        return ret

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

    def isStaticShape(self):
        """Returns ``True`` if the ``BoundsI`` instance
        has static, known dimensions. Always returns
        ``False`` for ``BoundsD``."""
        return self._isstaticshape


def _bounds_and_op_static(self, other):
    if not self.isDefined() or not other.isDefined():
        return self.__class__()
    else:
        xmin = max(self.xmin, other.xmin)
        xmax = min(self.xmax, other.xmax)
        ymin = max(self.ymin, other.ymin)
        ymax = min(self.ymax, other.ymax)
        if xmin > xmax or ymin > ymax:
            return self.__class__()
        else:
            return self.__class__(xmin, xmax, ymin, ymax)


def _bounds_and_op_dynamic(self, other):
    xmin = jnp.maximum(self.xmin, other.xmin)
    xmax = jnp.minimum(self.xmax, other.xmax)
    ymin = jnp.maximum(self.ymin, other.ymin)
    ymax = jnp.minimum(self.ymax, other.ymax)

    is_defined = self.isDefined() & other.isDefined() & (ymin <= ymax) & (xmin <= xmax)
    xmin = jnp.where(
        is_defined,
        xmin,
        0.0,
    )
    xmax = jnp.where(
        is_defined,
        xmax,
        0.0,
    )
    ymin = jnp.where(
        is_defined,
        ymin,
        0.0,
    )
    ymax = jnp.where(
        is_defined,
        ymax,
        0.0,
    )

    cls = self.__class__
    ret = cls.__new__(cls)
    ret.xmin = xmin
    ret.deltax = xmax - xmin
    ret.ymin = ymin
    ret.deltay = ymax - ymin
    ret._isdefined = is_defined
    ret._isstatic = False
    ret._isstaticshape = False

    return ret


def _bounds_bounds_add_op_static(self, other):
    if not other.isDefined():
        return self
    elif self.isDefined():
        xmin = min(self.xmin, other.xmin)
        xmax = max(self.xmax, other.xmax)
        ymin = min(self.ymin, other.ymin)
        ymax = max(self.ymax, other.ymax)
        return self.__class__(xmin, xmax, ymin, ymax)
    else:
        return other


def _bounds_bounds_add_op_dynamic(self, other, min_delta):
    def _ret_correct_attr(self_isdef, self_attr, other_isdef, other_attr, op):
        return jnp.where(
            ~other_isdef,
            self_attr,
            jnp.where(self_isdef, op(self_attr, other_attr), other_attr),
        )

    xmin = _ret_correct_attr(
        self._isdefined, self.xmin, other._isdefined, other.xmin, jnp.minimum
    )
    xmax = _ret_correct_attr(
        self._isdefined, self.xmax, other._isdefined, other.xmax, jnp.maximum
    )
    ymin = _ret_correct_attr(
        self._isdefined, self.ymin, other._isdefined, other.ymin, jnp.minimum
    )
    ymax = _ret_correct_attr(
        self._isdefined, self.ymax, other._isdefined, other.ymax, jnp.maximum
    )

    cls = self.__class__
    ret = cls.__new__(cls)

    ret.xmin = xmin
    ret.deltax = xmax - xmin + min_delta
    ret.ymin = ymin
    ret.deltay = ymax - ymin + min_delta
    ret._isdefined = jnp.where(
        ~other._isdefined,
        self._isdefined,
        jnp.where(
            self._isdefined,
            (ret.deltax >= min_delta) & (ret.deltay >= min_delta),
            other._isdefined,
        ),
    )
    ret._isstatic = False
    ret._isstaticshape = False

    return ret


def _bounds_pos_add_op_dynamic(self, other, min_delta):
    xmin = jnp.where(
        self._isdefined,
        jnp.minimum(self.xmin, other.x),
        other.x,
    )
    xmax = jnp.where(
        self._isdefined,
        jnp.maximum(self.xmax, other.x),
        other.x,
    )
    ymin = jnp.where(
        self._isdefined,
        jnp.minimum(self.ymin, other.y),
        other.y,
    )
    ymax = jnp.where(
        self._isdefined,
        jnp.maximum(self.ymax, other.y),
        other.y,
    )

    cls = self.__class__
    ret = cls.__new__(cls)

    ret.xmin = xmin
    ret.deltax = xmax - xmin + min_delta
    ret.ymin = ymin
    ret.deltay = ymax - ymin + min_delta
    ret._isdefined = jnp.where(
        self._isdefined,
        (ret.deltax >= min_delta) & (ret.deltay >= min_delta),
        jnp.array(True),
    )
    ret._isstatic = False
    ret._isstaticshape = False

    return ret


@implements(_galsim.BoundsD, lax_description=BOUNDS_LAX_DESCR)
@register_pytree_node_class
class BoundsD(Bounds):
    _pos_class = PositionD

    def __init__(self, *args, **kwargs):
        self._isstatic = False
        self._isstaticshape = False
        do_isdefined = self._parse_args(*args, **kwargs)
        self.xmin = cast_to_float(self.xmin)
        self.deltax = cast_to_float(self.deltax)
        self.ymin = cast_to_float(self.ymin)
        self.deltay = cast_to_float(self.deltay)
        if do_isdefined:
            self._isdefined = (self.deltax >= 0) & (self.deltay >= 0)
        self._isdefined = jnp.array(self._isdefined)

    def _check_scalar(self, x, name):
        try:
            if (
                isinstance(x, jax.Array)
                and x.shape == ()
                and jnp.issubdtype(x.dtype, jnp.floating)
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
                & (self.xmin <= cast_to_float(x))
                & (self.ymin <= cast_to_float(y))
                & (cast_to_float(x) <= self.xmax)
                & (cast_to_float(y) <= self.ymax)
            )
        elif len(args) == 0:
            raise TypeError("include takes at least 1 argument (0 given)")
        else:
            raise TypeError("include takes at most 2 arguments (%d given)" % len(args))

    def __repr__(self):
        # sometimes we will encounter a tracer here
        # and so we suppress any boolean conversion errors
        try:
            if jnp.any(self.isDefined()):
                print_full = True
            else:
                print_full = False
        except Exception:
            print_full = True

        if print_full:
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
        # sometimes we will encounter a tracer here
        # and so we suppress any boolean conversion errors
        try:
            if jnp.any(self.isDefined()):
                print_full = True
            else:
                print_full = False
        except Exception:
            print_full = True

        if print_full:
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

    def _getinitargs(self):
        # defined only for galsim test suite
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    def __eq__(self, other):
        if self is other:
            return jnp.array(True)
        elif isinstance(other, self.__class__):
            return (
                self.isDefined()
                & other.isDefined()
                & (self.xmin == other.xmin)
                & (self.ymin == other.ymin)
                & (self.xmax == other.xmax)
                & (self.ymax == other.ymax)
            ) | ((~self.isDefined()) & (~other.isDefined()))
        else:
            return jnp.array(False)

    def __ne__(self, other):
        return ~self.__eq__(other)

    def __and__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError("other must be a %s instance" % self.__class__.__name__)

        return _bounds_and_op_dynamic(self, other)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return _bounds_bounds_add_op_dynamic(self, other, 0)
        elif isinstance(other, self._pos_class):
            return _bounds_pos_add_op_dynamic(self, other, 0)
        else:
            raise TypeError(
                "other must be either a %s or a %s"
                % (self.__class__.__name__, self._pos_class.__name__)
            )


@implements(_galsim.BoundsI, lax_description=BOUNDS_LAX_DESCR)
@register_pytree_node_class
class BoundsI(Bounds):
    _pos_class = PositionI

    def __init__(self, *args, **kwargs):
        self._parse_args(*args, **kwargs)

        # validate inputs are ints
        self.deltax = check_is_int_then_cast(
            self.deltax, "BoundsI must be initialized with integer values"
        )
        self.deltay = check_is_int_then_cast(
            self.deltay, "BoundsI must be initialized with integer values"
        )
        self.xmin = check_is_int_then_cast(
            self.xmin, "BoundsI must be initialized with integer values"
        )
        self.ymin = check_is_int_then_cast(
            self.ymin, "BoundsI must be initialized with integer values"
        )

        if isinstance(self.deltax, int) and isinstance(self.deltay, int):
            self._isstaticshape = True
        else:
            self._isstaticshape = False

        if (
            isinstance(self.xmin, int)
            and isinstance(self.ymin, int)
            and isinstance(self.deltax, int)
            and isinstance(self.deltay, int)
        ):
            self._isstatic = True
        else:
            self._isstatic = False

        if self.isStaticShape():
            self._isdefined = self.deltax >= 1 and self.deltay >= 1
        else:
            self._isdefined = (self.deltax >= 1) & (self.deltay >= 1)

    def _check_scalar(self, x, name):
        try:
            if (
                isinstance(x, jax.Array)
                and x.shape == ()
                and jnp.issubdtype(x.dtype, jnp.integer)
            ):
                return
            elif x == int(x):
                return
        except (TypeError, ValueError):
            pass
        raise TypeError("%s must be an integer value" % name)

    def numpyShape(self):
        "A simple utility function to get the numpy shape that corresponds to this `Bounds` object."
        if self._isstaticshape:
            if self._isdefined:
                return self.deltay, self.deltax
            else:
                return 0, 0
        else:
            return jax.lax.cond(
                self._isdefined,
                lambda: (self.deltay, self.deltax),
                lambda: (jnp.zeros_like(self.deltay), jnp.zeros_like(self.deltax)),
            )

    @property
    def xmax(self):
        return self.xmin + self.deltax - 1

    @xmax.setter
    def xmax(self, value):
        self.deltax = value - self.xmin + 1

    @property
    def ymax(self):
        return self.ymin + self.deltay - 1

    @ymax.setter
    def ymax(self, value):
        self.deltay = value - self.ymin + 1

    def _area(self):
        # Remember the + 1 this time to include the pixels on both edges of the bounds.
        if self._isstaticshape:
            if self._isdefined:
                return self.deltax * self.deltay
            else:
                return 0
        else:
            return jax.lax.cond(
                self._isdefined,
                lambda: self.deltax * self.deltay,
                lambda: 0.0,
            )

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

    @implements(_galsim.Bounds.includes)
    def includes(self, *args):
        if len(args) == 1:
            if isinstance(args[0], Bounds):
                b = args[0]
                if self.isStatic() and b.isStatic():
                    return (
                        self.isDefined()
                        and b.isDefined()
                        and (self.xmin <= b.xmin)
                        and (self.xmax >= b.xmax)
                        and (self.ymin <= b.ymin)
                        and (self.ymax >= b.ymax)
                    )
                else:
                    return (
                        jnp.array(self.isDefined())
                        & jnp.array(b.isDefined())
                        & jnp.array(self.xmin <= b.xmin)
                        & jnp.array(self.xmax >= b.xmax)
                        & jnp.array(self.ymin <= b.ymin)
                        & jnp.array(self.ymax >= b.ymax)
                    )
            elif isinstance(args[0], Position):
                p = args[0]
                ok_types = STATIC_SCALAR_TYPES
                if (
                    self._isstatic
                    and isinstance(p.x, ok_types)
                    and isinstance(p.y, ok_types)
                ):
                    return (
                        self.isDefined()
                        and (self.xmin <= p.x)
                        and (self.ymin <= p.y)
                        and (p.x <= self.xmax)
                        and (p.y <= self.ymax)
                    )
                else:
                    return (
                        jnp.array(self.isDefined())
                        & jnp.array(self.xmin <= p.x)
                        & jnp.array(self.ymin <= p.y)
                        & jnp.array(p.x <= self.xmax)
                        & jnp.array(p.y <= self.ymax)
                    )
            else:
                raise TypeError("Invalid argument %s" % args[0])
        elif len(args) == 2:
            x, y = args
            x = cast_to_float(x)
            y = cast_to_float(y)
            if self._isstatic and isinstance(x, float) and isinstance(y, float):
                return (
                    self.isDefined()
                    and (self.xmin <= x)
                    and (self.ymin <= y)
                    and (x <= self.xmax)
                    and (y <= self.ymax)
                )
            else:
                return (
                    jnp.array(self.isDefined())
                    & jnp.array(self.xmin <= x)
                    & jnp.array(self.ymin <= y)
                    & jnp.array(x <= self.xmax)
                    & jnp.array(y <= self.ymax)
                )
        elif len(args) == 0:
            raise TypeError("include takes at least 1 argument (0 given)")
        else:
            raise TypeError("include takes at most 2 arguments (%d given)" % len(args))

    def __repr__(self):
        # sometimes we will encounter a tracer here
        # and so we suppress any boolean conversion errors
        try:
            if jnp.any(self.isDefined()):
                print_full = True
            else:
                print_full = False
        except Exception:
            print_full = True

        if print_full:
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
        # sometimes we will encounter a tracer here
        # and so we suppress any boolean conversion errors
        try:
            if jnp.any(self.isDefined()):
                print_full = True
            else:
                print_full = False
        except Exception:
            print_full = True

        if print_full:
            return "galsim.%s(xmin=%s, deltax=%s, ymin=%s, deltay=%s)" % (
                self.__class__.__name__,
                ensure_hashable(self.xmin),
                ensure_hashable(self.deltax),
                ensure_hashable(self.ymin),
                ensure_hashable(self.deltay),
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

    def __eq__(self, other):
        if self is other:
            if self.isStatic() and other.isStatic():
                return True
            else:
                return jnp.array(True)
        elif isinstance(other, self.__class__):
            if self.isStatic() and other.isStatic():
                min_eq = (self.xmin == other.xmin) and (self.ymin == other.ymin)
                self_isdef = self.isDefined()
                other_isdef = other.isDefined()
                shape_eq = (self.deltax == other.deltax) and (
                    self.deltay == other.deltay
                )
                return (self_isdef and other_isdef and shape_eq and min_eq) or (
                    (not self_isdef) and (not other_isdef)
                )
            else:
                min_eq = jnp.array(self.xmin == other.xmin) & jnp.array(
                    self.ymin == other.ymin
                )
                self_isdef = jnp.array(self.isDefined())
                other_isdef = jnp.array(other.isDefined())
                shape_eq = jnp.array(self.deltax == other.deltax) & jnp.array(
                    self.deltay == other.deltay
                )
                return (self_isdef & other_isdef & shape_eq & min_eq) | (
                    (~self_isdef) & (~other_isdef)
                )
        else:
            return False

    def __ne__(self, other):
        if not isinstance(other, self.__class__):
            return True

        if self.isStatic() and other.isStatic():
            return not self.__eq__(other)
        else:
            return ~self.__eq__(other)

    def __and__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError("other must be a %s instance" % self.__class__.__name__)

        if self.isStatic() and other.isStatic():
            return _bounds_and_op_static(self, other)
        else:
            return _bounds_and_op_dynamic(self, other)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            if self.isStatic() and other.isStatic():
                return _bounds_bounds_add_op_static(self, other)
            else:
                return _bounds_bounds_add_op_dynamic(self, other, 1)
        elif isinstance(other, self._pos_class):
            return _bounds_pos_add_op_dynamic(self, other, 1)
        else:
            raise TypeError(
                "other must be either a %s or a %s"
                % (self.__class__.__name__, self._pos_class.__name__)
            )

    def tree_flatten(self):
        """This function flattens the Bounds into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        aux_data = {"isstatic": self._isstatic, "isstaticshape": self._isstaticshape}

        if self._isstatic:
            aux_data["xmin"] = self.xmin
            aux_data["ymin"] = self.ymin

        if self._isstaticshape:
            aux_data["deltax"] = self.deltax
            aux_data["deltay"] = self.deltay
            aux_data["isdefined"] = self._isdefined

        if self._isstatic:
            children = tuple()
        elif self._isstaticshape:
            children = (self.xmin, self.ymin)
        else:
            children = (self.xmin, self.deltax, self.ymin, self.deltay, self._isdefined)

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        ret = cls.__new__(cls)
        ret._isstatic = aux_data["isstatic"]
        ret._isstaticshape = aux_data["isstaticshape"]

        if ret._isstatic:
            ret.xmin = aux_data["xmin"]
            ret.ymin = aux_data["ymin"]
            ret.deltax = aux_data["deltax"]
            ret.deltay = aux_data["deltay"]
            ret._isdefined = aux_data["isdefined"]
        elif ret._isstaticshape:
            ret.xmin = children[0]
            ret.ymin = children[1]
            ret.deltax = aux_data["deltax"]
            ret.deltay = aux_data["deltay"]
            ret._isdefined = aux_data["isdefined"]
        else:
            ret.xmin = children[0]
            ret.deltax = children[1]
            ret.ymin = children[2]
            ret.deltay = children[3]
            ret._isdefined = children[4]

        return ret
