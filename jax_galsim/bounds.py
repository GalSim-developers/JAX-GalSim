import galsim as _galsim
import numpy as np
from jax.tree_util import register_pytree_node_class

from jax_galsim.core.utils import (
    ensure_hashable,
    implements,
)
from jax_galsim.position import Position, PositionD, PositionI

BOUNDS_LAX_DESCR = """\
The JAX implementation

  - will not always test whether the bounds are valid
  - will not always test whether BoundsI is initialized with integers
"""


@implements(_galsim.Bounds, lax_description=BOUNDS_LAX_DESCR)
@register_pytree_node_class
class Bounds(_galsim.Bounds):
    def _parse_args(self, *args, **kwargs):
        if len(kwargs) == 0:
            if len(args) == 4:
                self._isdefined = True
                self.xmin, self.xmax, self.ymin, self.ymax = args
            elif len(args) == 0:
                self._isdefined = False
                self.xmin = self.xmax = self.ymin = self.ymax = 0
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

        if not (
            float(self.xmin) <= float(self.xmax)
            and float(self.ymin) <= float(self.ymax)
        ):
            self._isdefined = False

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
                    and b.isDefined()
                    and (self.xmin <= b.xmin)
                    and (self.xmax >= b.xmax)
                    and (self.ymin <= b.ymin)
                    and (self.ymax >= b.ymax)
                )
            elif isinstance(args[0], Position):
                p = args[0]
                return (
                    self.isDefined()
                    and (self.xmin <= p.x)
                    and (p.x <= self.xmax)
                    and (self.ymin <= p.y)
                    and (p.y <= self.ymax)
                )
            else:
                raise TypeError("Invalid argument %s" % args[0])
        elif len(args) == 2:
            x, y = args
            return (
                self.isDefined()
                and (self.xmin <= x)
                and (x <= self.xmax)
                and (self.ymin <= y)
                and (y <= self.ymax)
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
            dx = np.ceil(dx)
            dy = np.ceil(dy)
        return self.withBorder(dx, dy)

    def __and__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError("other must be a %s instance" % self.__class__.__name__)
        if not self.isDefined() or not other.isDefined():
            return self.__class__()
        else:
            xmin = np.maximum(self.xmin, other.xmin)
            xmax = np.minimum(self.xmax, other.xmax)
            ymin = np.maximum(self.ymin, other.ymin)
            ymax = np.minimum(self.ymax, other.ymax)
            if (xmin > xmax) or (ymin > ymax):
                return self.__class__()
            else:
                return self.__class__(xmin, xmax, ymin, ymax)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            if not other.isDefined():
                return self
            elif self.isDefined():
                xmin = np.minimum(self.xmin, other.xmin)
                xmax = np.maximum(self.xmax, other.xmax)
                ymin = np.minimum(self.ymin, other.ymin)
                ymax = np.maximum(self.ymax, other.ymax)
                return self.__class__(xmin, xmax, ymin, ymax)
            else:
                return other
        elif isinstance(other, self._pos_class):
            if self.isDefined():
                xmin = np.minimum(self.xmin, other.x)
                xmax = np.maximum(self.xmax, other.x)
                ymin = np.minimum(self.ymin, other.y)
                ymax = np.maximum(self.ymax, other.y)
                return self.__class__(xmin, xmax, ymin, ymax)
            else:
                return self.__class__(other)
        else:
            raise TypeError(
                "other must be either a %s or a %s"
                % (self.__class__.__name__, self._pos_class.__name__)
            )

    def __repr__(self):
        if self.isDefined():
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
                ensure_hashable(self.xmax),
                ensure_hashable(self.ymin),
                ensure_hashable(self.ymax),
            )
        )

    def tree_flatten(self):
        """This function flattens the Bounds into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        children = ()
        # Define auxiliary static data that doesn’t need to be traced
        if self.isDefined():
            aux_data = (self.xmin, self.xmax, self.ymin, self.ymax)
        else:
            aux_data = ()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        return cls(*aux_data)

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


@implements(_galsim.BoundsD, lax_description=BOUNDS_LAX_DESCR)
@register_pytree_node_class
class BoundsD(Bounds):
    _pos_class = PositionD

    def __init__(self, *args, **kwargs):
        self._parse_args(*args, **kwargs)
        self.xmin = float(self.xmin)
        self.xmax = float(self.xmax)
        self.ymin = float(self.ymin)
        self.ymax = float(self.ymax)

        if not (
            isinstance(
                self.xmin,
                (float, int, np.ndarray, np.int32, np.int64, np.float32, np.float64),
            )
            and isinstance(
                self.xmax,
                (float, int, np.ndarray, np.int32, np.int64, np.float32, np.float64),
            )
            and isinstance(
                self.ymin,
                (float, int, np.ndarray, np.int32, np.int64, np.float32, np.float64),
            )
            and isinstance(
                self.ymax,
                (float, int, np.ndarray, np.int32, np.int64, np.float32, np.float64),
            )
        ):
            raise ValueError(
                "BoundsI/D classes must use python ints/floats or numpy values in JAX-GalSim!"
            )

    def _check_scalar(self, x, name):
        try:
            if (
                isinstance(x, np.ndarray)
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
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    @property
    def _center(self):
        return PositionD((self.xmax + self.xmin) / 2.0, (self.ymax + self.ymin) / 2.0)


@implements(_galsim.BoundsI, lax_description=BOUNDS_LAX_DESCR)
@register_pytree_node_class
class BoundsI(Bounds):
    _pos_class = PositionI

    def __init__(self, *args, **kwargs):
        self._parse_args(*args, **kwargs)
        # for simple inputs, we can check if the bounds are valid ints
        if (
            isinstance(self.xmin, (float, int))
            and isinstance(self.xmax, (float, int))
            and isinstance(self.ymin, (float, int))
            and isinstance(self.ymax, (float, int))
            and (
                self.xmin != int(self.xmin)
                or self.xmax != int(self.xmax)
                or self.ymin != int(self.ymin)
                or self.ymax != int(self.ymax)
            )
        ):
            raise TypeError("BoundsI must be initialized with integer values")

        self.xmin = int(self.xmin)
        self.xmax = int(self.xmax)
        self.ymin = int(self.ymin)
        self.ymax = int(self.ymax)

        if not (
            isinstance(
                self.xmin,
                (float, int, np.ndarray, np.int32, np.int64, np.float32, np.float64),
            )
            and isinstance(
                self.xmax,
                (float, int, np.ndarray, np.int32, np.int64, np.float32, np.float64),
            )
            and isinstance(
                self.ymin,
                (float, int, np.ndarray, np.int32, np.int64, np.float32, np.float64),
            )
            and isinstance(
                self.ymax,
                (float, int, np.ndarray, np.int32, np.int64, np.float32, np.float64),
            )
        ):
            raise ValueError(
                "BoundsI/D classes must use python ints/floats or numpy values in JAX-GalSim!"
            )

    def _check_scalar(self, x, name):
        try:
            if (
                isinstance(x, np.ndarray)
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
            return self.ymax - self.ymin + 1, self.xmax - self.xmin + 1
        else:
            return 0, 0

    def _area(self):
        # Remember the + 1 this time to include the pixels on both edges of the bounds.
        if not self.isDefined():
            return 0
        else:
            return (self.xmax - self.xmin + 1) * (self.ymax - self.ymin + 1)

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
