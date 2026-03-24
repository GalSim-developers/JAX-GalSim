import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class ImageIndexer:
    def __init__(self, image):
        self.image = image

    def tree_flatten(self):
        """Flatten the image into a list of values."""
        children = (self.image,)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        obj = object.__new__(cls)
        obj.image = children[0]
        return obj

    def __getitem__(self, *args):
        from jax_galsim import BoundsI, PositionI

        if len(args) == 1:
            args = args[0]
        else:
            raise TypeError("`image.at[index]` got unknown args: %r" % (args,))

        if isinstance(args, BoundsI):
            return ImageIndex(self.image, args)
        elif isinstance(args, PositionI):
            return ImageIndex(self.image, args)
        elif args is Ellipsis:
            return ImageIndex(self.image, self.image.bounds)
        elif isinstance(args, tuple):
            if (
                isinstance(args[0], slice)
                and isinstance(args[1], slice)
                and args[0] == slice(None)
                and args[1] == slice(None)
            ):
                return ImageIndex(self.image, self.image.bounds)
            else:
                return ImageIndex(self.image, PositionI(*args))
        else:
            raise TypeError(
                "`image.at[index]` only accepts BoundsI, PositionI, "
                "tuples, `...`, `:, :`, or `x, y` for the index."
            )


@register_pytree_node_class
class ImageIndex:
    def __init__(self, image, index):
        self.image = image
        self.index = index

    def tree_flatten(self):
        """Flatten the image into a list of values."""
        children = (self.image, self.index)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        obj = object.__new__(cls)
        obj.image = children[0]
        obj.index = children[1]
        return obj

    def set(self, value):
        import galsim as _galsim

        from jax_galsim import BoundsI, PositionI

        if self.image.isconst:
            raise _galsim.GalSimImmutableError(
                "Cannot modify an immutable Image", self.image
            )

        if not self.image.bounds.isDefined():
            raise _galsim.GalSimUndefinedBoundsError(
                "Attempt to set value of an undefined image"
            )

        if isinstance(self.index, PositionI):
            if not self.image.bounds.includes(self.index):
                raise _galsim.GalSimBoundsError(
                    "Attempt to set position not in bounds of image",
                    self.index,
                    self.image.bounds,
                )
            self.image._setValue(self.index.x, self.index.y, value)
        elif isinstance(self.index, BoundsI):
            if not self.image.bounds.includes(self.index):
                raise _galsim.GalSimBoundsError(
                    "Attempt to access subImage not (fully) in image",
                    self.index,
                    self.image.bounds,
                )
            if (
                hasattr(value, "bounds")
                and self.index.numpyShape() != value.bounds.numpyShape()
            ):
                raise _galsim.GalSimIncompatibleValuesError(
                    "Trying to copy images that are not the same shape",
                    self_image=self.image,
                    rhs=value,
                )
            start_inds = (
                self.index.ymin - self.image.ymin,
                self.index.xmin - self.image.xmin,
            )
            self.image._array = jax.lax.dynamic_update_slice(
                self.image.array,
                value.array
                if hasattr(value, "array")
                else jnp.broadcast_to(value, self.index.numpyShape()),
                start_inds,
            )
        else:
            raise TypeError(
                "This error should never be raised. "
                "image.at[index] only accepts BoundsI or PositionI for the index"
            )

        return self.image

    def get(self):
        import galsim as _galsim

        from jax_galsim import BoundsI, PositionI

        if not self.image.bounds.isDefined():
            raise _galsim.GalSimUndefinedBoundsError(
                "Attempt to get value of an undefined image"
            )

        if isinstance(self.index, PositionI):
            if not self.image.bounds.includes(self.index):
                raise _galsim.GalSimBoundsError(
                    "Attempt to access position not in bounds of image.",
                    self.index,
                    self.image.bounds,
                )
            return self.image._getValue(self.index.x, self.index.y)
        elif isinstance(self.index, BoundsI):
            if not self.image.bounds.includes(self.index):
                raise _galsim.GalSimBoundsError(
                    "Attempt to access subImage not (fully) in image",
                    self.index,
                    self.image.bounds,
                )
            start_inds = (
                self.index.ymin - self.image.ymin,
                self.index.xmin - self.image.xmin,
            )
            shape = self.index.numpyShape()
            arr = jax.lax.dynamic_slice(self.image.array, start_inds, shape)
            return self.image.__class__(arr, bounds=self.index, wcs=self.image.wcs)
        else:
            raise TypeError(
                "This error should never be raised. "
                "image.at[index] only accepts BoundsI or PositionI for the index"
            )

    def _op(self, value, func, check_integer=False):
        import galsim as _galsim

        from jax_galsim import BoundsI, Image, PositionI

        if check_integer and not self.image.isinteger:
            raise _galsim.GalSimValueError(
                "Image must have integer values.", self.image
            )

        if check_integer and isinstance(value, Image) and not value.isinteger:
            raise _galsim.GalSimValueError(
                "Image must have integer values.", self.image
            )

        if self.image.isconst:
            raise _galsim.GalSimImmutableError(
                "Cannot modify an immutable Image", self.image
            )

        if not self.image.bounds.isDefined():
            raise _galsim.GalSimUndefinedBoundsError(
                "Attempt to modify to an undefined image"
            )

        if isinstance(self.index, PositionI):
            if not self.image.bounds.includes(self.index):
                raise _galsim.GalSimBoundsError(
                    "Attempt to modify position not in bounds of image.",
                    self.index,
                    self.image.bounds,
                )
            subim = self.image._getValue(self.index.x, self.index.y)
            self.image._setValue(self.index.x, self.index.y, func(subim, value))
        elif isinstance(self.index, BoundsI):
            if not self.image.bounds.includes(self.index):
                raise _galsim.GalSimBoundsError(
                    "Attempt to access subImage not (fully) in image",
                    self.index,
                    self.image.bounds,
                )
            if (
                hasattr(value, "bounds")
                and self.index.numpyShape() != value.bounds.numpyShape()
            ):
                raise _galsim.GalSimIncompatibleValuesError(
                    "Trying to copy images that are not the same shape",
                    self_image=self.image,
                    rhs=value,
                )

            start_inds = (
                self.index.ymin - self.image.ymin,
                self.index.xmin - self.image.xmin,
            )
            shape = self.index.numpyShape()
            subim = jax.lax.dynamic_slice(self.image.array, start_inds, shape)

            self.image._array = jax.lax.dynamic_update_slice(
                self.image.array,
                func(
                    subim,
                    value.array
                    if hasattr(value, "array")
                    else jnp.broadcast_to(value, self.index.numpyShape()),
                ),
                start_inds,
            )
        else:
            raise TypeError(
                "This error should never be raised. "
                "image.at[index] only accepts BoundsI or PositionI for the index"
            )

        return self.image

    def add(self, value):
        return self._op(value, lambda x, y: x + y)

    def subtract(self, value):
        return self._op(value, lambda x, y: x - y)

    def multiply(self, value):
        return self._op(value, lambda x, y: x * y)

    def divide(self, value):
        return self._op(value, lambda x, y: x / y)

    def power(self, value):
        return self._op(value, lambda x, y: x**y)

    def mod(self, value):
        return self._op(value, lambda x, y: x % y, check_integer=True)

    def floor_divide(self, value):
        return self._op(value, lambda x, y: x // y, check_integer=True)

    def bitwise_and(self, value):
        return self._op(value, lambda x, y: x & y, check_integer=True)

    def bitwise_xor(self, value):
        return self._op(value, lambda x, y: x ^ y, check_integer=True)

    def bitwise_or(self, value):
        return self._op(value, lambda x, y: x | y, check_integer=True)
