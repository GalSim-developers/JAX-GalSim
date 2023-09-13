import galsim as _galsim
import jax
import jax.numpy as jnp
from jax._src.numpy.util import _wraps
from jax.tree_util import register_pytree_node_class


@_wraps(_galsim.Position)
class Position(object):
    def __init__(self, *args, **kwargs):
        if len(kwargs) == 0:
            if len(args) == 2:
                self.x, self.y = args
            elif len(args) == 0:
                self.x = self.y = 0
            elif len(args) == 1:
                if isinstance(args[0], (Position,)):
                    self.x = args[0].x
                    self.y = args[0].y
                else:
                    try:
                        self.x, self.y = args[0]
                    except (TypeError, ValueError):
                        raise TypeError(
                            "Single argument to %s must be either a Position "
                            "or a tuple." % self.__class__
                        )
            else:
                raise TypeError(
                    "%s takes at most 2 arguments (%d given)"
                    % (self.__class__, len(args))
                )
        elif len(args) != 0:
            raise TypeError(
                "%s takes x and y as either named or unnamed arguments (given %s, %s)"
                % (self.__class__, args, kwargs)
            )
        else:
            try:
                self.x = kwargs.pop("x")
                self.y = kwargs.pop("y")
            except KeyError:
                raise TypeError(
                    "Keyword arguments x,y are required for %s" % self.__class__
                )
            if kwargs:
                raise TypeError("Got unexpected keyword arguments %s" % kwargs.keys())

    @property
    def array(self):
        """Return the position as a 2-element numpy array."""
        return jnp.array([self.x, self.y])

    def __mul__(self, other):
        return self.__class__(self.x * other, self.y * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        return self.__class__(self.x / other, self.y / other)

    __truediv__ = __div__

    def __neg__(self):
        return self.__class__(-self.x, -self.y)

    def __add__(self, other):
        # from .bounds import Bounds
        # if isinstance(other,Bounds):
        #     return other + self
        if not isinstance(other, Position):
            raise TypeError("Can only add a Position to a %s" % self.__class__.__name__)
        elif isinstance(other, self.__class__):
            return self.__class__(self.x + other.x, self.y + other.y)
        else:
            return PositionD(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        if not isinstance(other, Position):
            raise TypeError(
                "Can only subtract a Position from a %s" % self.__class__.__name__
            )
        elif isinstance(other, self.__class__):
            return self.__class__(self.x - other.x, self.y - other.y)
        else:
            return PositionD(self.x - other.x, self.y - other.y)

    def __repr__(self):
        return "jax_galsim.%s(x=%r, y=%r)" % (self.__class__.__name__, self.x, self.y)

    def __str__(self):
        return "jax_galsim.%s(%s,%s)" % (self.__class__.__name__, self.x, self.y)

    def __eq__(self, other):
        return self is other or (
            isinstance(other, self.__class__)
            and self.x == other.x
            and self.y == other.y
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.__class__.__name__, self.x, self.y))

    def shear(self, shear):
        """Shear the position.

        See the doc string of `galsim.Shear.getMatrix` for more details.

        Parameters:
            shear:    a `galsim.Shear` instance

        Returns:
            a `galsim.PositionD` instance.
        """
        shear_mat = shear.getMatrix()
        shear_pos = jnp.dot(shear_mat, self.array)
        return PositionD(shear_pos[0], shear_pos[1])

    def tree_flatten(self):
        """This function flattens the GSObject into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        children = (self.x, self.y)
        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        del aux_data
        obj = object.__new__(cls)
        obj.x = children[0]
        obj.y = children[1]
        return obj


@_wraps(_galsim.PositionD)
@register_pytree_node_class
class PositionD(Position):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@_wraps(_galsim.PositionI)
@register_pytree_node_class
class PositionI(Position):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Force conversion to int type in this case
        if isinstance(self.x, float):
            self.x = int(self.x)
        elif isinstance(self.x, jax.Array):
            self.x = self.x.astype("int")

        if isinstance(self.y, float):
            self.y = int(self.y)
        elif isinstance(self.y, jax.Array):
            self.y = self.y.astype("int")
