import equinox
import galsim as _galsim
import jax
import jax.numpy as jnp
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
The JAX-GalSim implementation of the ``BoundsI/D`` classes have some key differences
from GalSim.

- ``BoundsI`` instances must have statically known shapes, but may have non-static
  start locations (i.e., ``xmin`` and ``ymin`` may be JAX arrays, traced in JIT operations, etc.).
  This restriction mirrors the JAX restriction that arrays have fixed shapes when traced
  for function transformations like ``jax.vmap``, ``jax.jit``, etc.
- Upon initialization, if a ``BoundsI`` object has a non-static shape, JAX-GalSim will attempt to convert
  it to a static shape by extracting the dimensions from the array via ``.item()``. This operation will
  cause JAX to raise an error if the code is being traced.
- If a ``BoundsI`` object is declared with static ``xmin`` and ``ymin`` values, an error will be raised
  if one attempts to convert those values to non-static values.
- JAX-GalSim does not support the use of the `&/+` dunder methods (i.e., ``__and__`` and ``__add__``)
  for ``BoundsI`` objects when tracing code.
- JAX-Galsim supports an additional initialization signature  ``BoundsI(xmin=..., deltax=..., ymin=..., deltay=...)``
  to help users specify the widths ``deltax`` and ``deltay`` statically at initialization.
- When calling ``jax.vmap``, ``jax.jit`` etc. with ``BoundsI`` objects, ``xmin`` and ``ymin`` are
  traced by JAX. The combination of this feature with statically known shapes allows for code that renders
  objects in fixed sized stamps with variable locations, a common operation.
- For ``BoundsD``, all ``x(y)min(max)`` values are traced as arrays.
- ``Bounds`` objects always return a JAX boolean values for various method calls, except for
  ``BoundsI.isDefined()`` which is always a Python boolean value.
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
                    self.xmin = jnp.minimum(args[0].x, args[1].x)
                    self.xmax = jnp.maximum(args[0].x, args[1].x)
                    self.ymin = jnp.minimum(args[0].y, args[1].y)
                    self.ymax = jnp.maximum(args[0].y, args[1].y)
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
            if not self._isdefined:
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
            if not self._isdefined:
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
        if len(args) == 1:
            if isinstance(args[0], Bounds):
                b = args[0]
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

        return _bounds_and_op_dynamic(self, other)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return _bounds_bounds_add_op_dynamic(self, other)
        elif isinstance(other, self._pos_class):
            return _bounds_pos_add_op_dynamic(self, other)
        else:
            raise TypeError(
                "other must be either a %s or a %s"
                % (self.__class__.__name__, self._pos_class.__name__)
            )

    def __eq__(self, other):
        if self is other:
            return jnp.array(True)
        elif isinstance(other, self.__class__):
            self_isdef = jnp.array(self.isDefined())
            other_isdef = jnp.array(other.isDefined())
            if isinstance(self, BoundsD):
                return (
                    self_isdef
                    & other_isdef
                    & jnp.array(self.xmin == other.xmin)
                    & jnp.array(self.ymin == other.ymin)
                    & jnp.array(self.xmax == other.xmax)
                    & jnp.array(self.ymax == other.ymax)
                ) | ((~self_isdef) & (~other_isdef))
            else:
                return (
                    self_isdef
                    & other_isdef
                    & jnp.array(self.xmin == other.xmin)
                    & jnp.array(self.ymin == other.ymin)
                    & jnp.array(self.deltax == other.deltax)
                    & jnp.array(self.deltay == other.deltay)
                ) | ((~self_isdef) & (~other_isdef))
        else:
            return jnp.array(False)

    def __ne__(self, other):
        return ~self.__eq__(other)

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


def _bounds_and_op_dynamic(self, other):
    xmin = jnp.maximum(self.xmin, other.xmin)
    xmax = jnp.minimum(self.xmax, other.xmax)
    ymin = jnp.maximum(self.ymin, other.ymin)
    ymax = jnp.minimum(self.ymax, other.ymax)

    is_defined = (
        jnp.array(self.isDefined())
        & jnp.array(other.isDefined())
        & jnp.array(ymin <= ymax)
        & jnp.array(xmin <= xmax)
    )
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
    if isinstance(self, BoundsI):
        # we use the class constructor here to ensure we properly convert
        # bounds shape to static ints
        ret = cls(
            xmin=xmin,
            deltax=xmax - xmin + 1,
            ymin=ymin,
            deltay=ymax - ymin + 1,
        )
        # we have to do a conversion to static bools here too
        ret._isdefined = bool(is_defined.item())
    else:
        ret = cls.__new__(cls)
        ret.xmin = xmin
        ret.deltax = xmax - xmin
        ret.ymin = ymin
        ret.deltay = ymax - ymin
        ret._isdefined = is_defined

    return ret


def _bounds_bounds_add_op_dynamic(self, other):
    def _ret_correct_attr(self_isdef, self_attr, other_isdef, other_attr, op):
        return jnp.where(
            ~jnp.array(other_isdef),
            self_attr,
            jnp.where(jnp.array(self_isdef), op(self_attr, other_attr), other_attr),
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
    if isinstance(self, BoundsI):
        # we use the class constructor here to ensure we properly convert
        # bounds shape to static ints
        ret = cls(
            xmin=xmin,
            deltax=xmax - xmin + 1,
            ymin=ymin,
            deltay=ymax - ymin + 1,
        )
        is_defined = jnp.where(
            ~jnp.array(other._isdefined),
            jnp.array(self._isdefined),
            jnp.where(
                jnp.array(self._isdefined),
                jnp.array(ret.deltax >= 1) & jnp.array(ret.deltay >= 1),
                jnp.array(other._isdefined),
            ),
        )
        # we have to do a conversion to static bools here too
        ret._isdefined = bool(is_defined.item())
    else:
        ret = cls.__new__(cls)
        ret.xmin = xmin
        ret.deltax = xmax - xmin
        ret.ymin = ymin
        ret.deltay = ymax - ymin
        ret._isdefined = jnp.where(
            ~jnp.array(other._isdefined),
            jnp.array(self._isdefined),
            jnp.where(
                jnp.array(self._isdefined),
                jnp.array(ret.deltax >= 0) & jnp.array(ret.deltay >= 0),
                jnp.array(other._isdefined),
            ),
        )

    return ret


def _bounds_pos_add_op_dynamic(self, other):
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
    if isinstance(self, BoundsI):
        # we use the class constructor here to ensure we properly convert
        # bounds shape to static ints
        ret = cls(
            xmin=xmin,
            deltax=xmax - xmin + 1,
            ymin=ymin,
            deltay=ymax - ymin + 1,
        )
        is_defined = jnp.where(
            jnp.array(self._isdefined),
            jnp.array(ret.deltax >= 1) & jnp.array(ret.deltay >= 1),
            jnp.array(True),
        )
        # we have to do a conversion to static bools here too
        ret._isdefined = bool(is_defined.item())
    else:
        ret = cls.__new__(cls)
        ret.xmin = xmin
        ret.deltax = xmax - xmin
        ret.ymin = ymin
        ret.deltay = ymax - ymin
        ret._isdefined = jnp.where(
            self._isdefined,
            jnp.array(ret.deltax >= 0) & jnp.array(ret.deltay >= 0),
            jnp.array(True),
        )

    return ret


@implements(_galsim.BoundsD, lax_description=BOUNDS_LAX_DESCR)
@register_pytree_node_class
class BoundsD(Bounds):
    _pos_class = PositionD

    def __init__(self, *args, **kwargs):
        do_isdefined = self._parse_args(*args, **kwargs)
        self.xmin = cast_to_float(jnp.array(self.xmin))
        self.deltax = cast_to_float(jnp.array(self.deltax))
        self.ymin = cast_to_float(jnp.array(self.ymin))
        self.deltay = cast_to_float(jnp.array(self.deltay))
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


@implements(_galsim.BoundsI, lax_description=BOUNDS_LAX_DESCR)
@register_pytree_node_class
class BoundsI(Bounds):
    _pos_class = PositionI

    def __init__(self, *args, **kwargs):
        # we set these variables to disable type checking and conversion
        # for xmin/ymin while we initialize the object
        self._isstatic = True
        self._dotypechecking = False
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

        # attempt to convert widths to static values
        # this will raise if values are being traced
        # we let that error propagate instead of reraising
        # our own.
        if not isinstance(self.deltax, int):
            self.deltax = int(self.deltax.item())
        if not isinstance(self.deltay, int):
            self.deltay = int(self.deltay.item())

        self._isdefined = self.deltax >= 1 and self.deltay >= 1

        # now we compute these properties correctly and turn on type checking
        if isinstance(self._xmin, int) and isinstance(self._ymin, int):
            self._isstatic = True
        else:
            self._isstatic = False
        self._dotypechecking = True

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
        if self._isdefined:
            return self.deltay, self.deltax
        else:
            return 0, 0

    # we store xmin internally as a float even though it is an int
    # so that autodiff works properly (needs floats in general)
    @property
    def xmin(self):
        if self._isstatic:
            return self._xmin
        else:
            return jnp.astype(self._xmin, int)

    @xmin.setter
    def xmin(self, value):
        value = check_is_int_then_cast(value, "BoundsI xmin values must be integers")
        if self._isstatic:
            if self._dotypechecking and isinstance(value, jnp.ndarray):
                raise RuntimeError(
                    "Static `BoundsI` classes cannot be converted to dynamic ones."
                )
            self._xmin = value
        else:
            self._xmin = jnp.astype(value, float)

    @property
    def xmax(self):
        return cast_to_int(self.xmin + self.deltax - 1)

    @xmax.setter
    def xmax(self, value):
        self.deltax = value - self.xmin + 1
        self.deltax = check_is_int_then_cast(
            self.deltax, "BoundsI xmax must be set to an integer value"
        )
        # attempt to convert widths to static values
        # this will raise if values are being traced
        # we let that error propagate instead of reraising
        # our own.
        if not isinstance(self.deltax, int):
            self.deltax = int(self.deltax.item())

    # we store ymin internally as a float even though it is an int
    # so that autodiff works properly (needs floats in general)
    @property
    def ymin(self):
        if self._isstatic:
            return self._ymin
        else:
            return jnp.astype(self._ymin, int)

    @ymin.setter
    def ymin(self, value):
        value = check_is_int_then_cast(value, "BoundsI ymin values must be integers")
        if self._isstatic:
            if self._dotypechecking and isinstance(value, jnp.ndarray):
                raise RuntimeError(
                    "Static `BoundsI` classes cannot be converted to dynamic ones."
                )
            self._ymin = value
        else:
            self._ymin = jnp.astype(value, float)

    @property
    def ymax(self):
        return cast_to_int(self.ymin + self.deltay - 1)

    @ymax.setter
    def ymax(self, value):
        self.deltay = value - self.ymin + 1
        self.deltay = check_is_int_then_cast(
            self.deltay, "BoundsI ymax must be set to an integer value"
        )
        # attempt to convert widths to static values
        # this will raise if values are being traced
        # we let that error propagate instead of reraising
        # our own.
        if not isinstance(self.deltay, int):
            self.deltay = int(self.deltay.item())

    def _area(self):
        # Remember the + 1 this time to include the pixels on both edges of the bounds.
        if not self._isdefined:
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

    def __repr__(self):
        if self._isdefined:
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
        if self._isdefined:
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

    def tree_flatten(self):
        """This function flattens the Bounds into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        aux_data = {"isstatic": self._isstatic}

        # Define the children nodes of the PyTree that need tracing
        if self._isstatic:
            children = tuple()
            aux_data["xmin"] = self._xmin
            aux_data["ymin"] = self._ymin
        else:
            children = (self._xmin, self._ymin)

        # untraced aux data
        aux_data["deltax"] = self.deltax
        aux_data["deltay"] = self.deltay
        aux_data["isdefined"] = self._isdefined

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        ret = cls.__new__(cls)
        if aux_data["isstatic"]:
            ret._xmin = aux_data["xmin"]
            ret._ymin = aux_data["ymin"]
        else:
            ret._xmin = children[0]
            ret._ymin = children[1]
        ret.deltax = aux_data["deltax"]
        ret.deltay = aux_data["deltay"]
        ret._isdefined = aux_data["isdefined"]
        ret._isstatic = aux_data["isstatic"]
        return ret
