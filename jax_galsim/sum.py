import galsim as _galsim
import jax.numpy as jnp
import numpy as np
from jax._src.numpy.util import _wraps
from jax.tree_util import register_pytree_node_class

from jax_galsim.gsobject import GSObject
from jax_galsim.gsparams import GSParams


@_wraps(
    _galsim.Add, lax_description="Does not support `ChromaticObject` at this point."
)
def Add(*args, **kwargs):
    return Sum(*args, **kwargs)


@_wraps(
    _galsim.Sum, lax_description="Does not support `ChromaticObject` at this point."
)
@register_pytree_node_class
class Sum(GSObject):
    def __init__(self, *args, gsparams=None, propagate_gsparams=True):
        self._propagate_gsparams = propagate_gsparams

        if len(args) == 0:
            raise TypeError("At least one GSObject must be provided.")
        elif len(args) == 1:
            # 1 argument.  Should be either a GSObject or a list of GSObjects
            if isinstance(args[0], GSObject):
                args = [args[0]]
            elif isinstance(args[0], list) or isinstance(args[0], tuple):
                args = args[0]
            else:
                raise TypeError(
                    "Single input argument must be a GSObject or list of them."
                )
        # else args is already the list of objects

        # Consolidate args for Sums of Sums...
        new_args = []
        for a in args:
            if isinstance(a, Sum):
                new_args.extend(a.params["obj_list"])
            else:
                new_args.append(a)
        args = new_args

        for obj in args:
            if not isinstance(obj, GSObject):
                raise TypeError("Arguments to Sum must be GSObjects, not %s" % obj)

        # Figure out what gsparams to use
        if gsparams is None:
            # If none is given, take the most restrictive combination from the obj_list.
            self._gsparams = GSParams.combine([obj.gsparams for obj in args])
        else:
            # If something explicitly given, then use that.
            self._gsparams = GSParams.check(gsparams)

        # Apply gsparams to all in obj_list.
        if self._propagate_gsparams:
            args = [obj.withGSParams(self._gsparams) for obj in args]

        # Save the list as an attribute, so it can be inspected later if necessary.
        self._params = {"obj_list": args}

    @property
    def obj_list(self):
        """The list of objects being added."""
        return self._params["obj_list"]

    @property
    def flux(self):
        flux_list = jnp.array([obj.flux for obj in self.obj_list])
        return jnp.sum(flux_list)

    def withGSParams(self, gsparams=None, **kwargs):
        """Create a version of the current object with the given gsparams

        .. note::

            Unless you set ``propagate_gsparams=False``, this method will also update the gsparams
            of each object being added.
        """
        if gsparams == self.gsparams:
            return self
        ret = self.__class__(
            self.params["obj_list"],
            gsparams=gsparams,
            propagate_gsparams=self._propagate_gsparams,
        )
        return ret

    def __hash__(self):
        return hash(
            (
                "jax_galsim.Sum",
                tuple(self.obj_list),
                self.gsparams,
                self._propagate_gsparams,
            )
        )

    def __repr__(self):
        return "jax_galsim.Sum(%r, gsparams=%r, propagate_gsparams=%r)" % (
            self.obj_list,
            self.gsparams,
            self._propagate_gsparams,
        )

    def __str__(self):
        str_list = [str(obj) for obj in self.obj_list]
        return "(" + " + ".join(str_list) + ")"

    @property
    def _maxk(self):
        maxk_list = jnp.array([obj.maxk for obj in self.obj_list])
        return jnp.max(maxk_list)

    @property
    def _stepk(self):
        stepk_list = jnp.array([obj.stepk for obj in self.obj_list])
        return jnp.min(stepk_list)

    @property
    def _has_hard_edges(self):
        hard_list = [obj.has_hard_edges for obj in self.obj_list]
        return bool(np.any(hard_list))

    @property
    def _is_axisymmetric(self):
        axi_list = [obj.is_axisymmetric for obj in self.obj_list]
        return bool(np.all(axi_list))

    @property
    def _is_analytic_x(self):
        ax_list = [obj.is_analytic_x for obj in self.obj_list]
        return bool(np.all(ax_list))

    @property
    def _is_analytic_k(self):
        ak_list = [obj.is_analytic_k for obj in self.obj_list]
        return bool(np.all(ak_list))

    @property
    def _centroid(self):
        cen_list = jnp.array([obj.centroid * obj.flux for obj in self.obj_list])
        return jnp.sum(cen_list) / self.flux

    @property
    def _max_sb(self):
        sb_list = jnp.array([obj.max_sb for obj in self.obj_list])
        return jnp.sum(sb_list)

    def _xValue(self, pos):
        xv_list = jnp.array([obj._xValue(pos) for obj in self.obj_list])
        return jnp.sum(xv_list, axis=0)

    def _kValue(self, pos):
        kv_list = jnp.array([obj._kValue(pos) for obj in self.obj_list])
        return jnp.sum(kv_list, axis=0)

    def _drawReal(self, image, jac=None, offset=(0.0, 0.0), flux_scaling=1.0):
        image = self.obj_list[0]._drawReal(image, jac, offset, flux_scaling)
        if len(self.obj_list) > 1:
            for obj in self.obj_list[1:]:
                image += obj._drawReal(image, jac, offset, flux_scaling)
        return image

    def _drawKImage(self, image, jac=None):
        image = self.obj_list[0]._drawKImage(image, jac)
        if len(self.obj_list) > 1:
            for obj in self.obj_list[1:]:
                image += obj._drawKImage(image, jac)
        return image

    def tree_flatten(self):
        """This function flattens the GSObject into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        children = (self.params,)
        # Define auxiliary static data that doesn’t need to be traced
        aux_data = {
            "gsparams": self.gsparams,
            "propagate_gsparams": self._propagate_gsparams,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        return cls(children[0]["obj_list"], **aux_data)
