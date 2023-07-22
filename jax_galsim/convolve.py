import galsim as _galsim
import jax.numpy as jnp
from galsim.errors import galsim_warn
from jax._src.numpy.util import _wraps
from jax.tree_util import register_pytree_node_class

from jax_galsim.gsobject import GSObject
from jax_galsim.gsparams import GSParams


@_wraps(
    _galsim.Convolve,
    lax_description="""Does not support ChromaticConvolutions""",
)
def Convolve(*args, **kwargs):
    if len(args) == 0:
        raise TypeError("At least one GSObject must be provided.")
    elif len(args) == 1:
        if isinstance(args[0], GSObject):
            args = [args[0]]
        elif isinstance(args[0], list) or isinstance(args[0], tuple):
            args = args[0]
        else:
            raise TypeError(
                "Single input argument must be a GSObject, "
                + "or a (possibly mixed) list of them."
            )
    # else args is already the list of objects

    return Convolution(*args, **kwargs)


@_wraps(
    _galsim.Convolution,
    lax_description="""Only supports 'fft' convolution.""",
)
@register_pytree_node_class
class Convolution(GSObject):
    def __init__(self, *args, **kwargs):
        # First check for number of arguments != 0
        if len(args) == 0:
            raise TypeError("At least one GSObject must be provided.")
        elif len(args) == 1:
            if isinstance(args[0], GSObject):
                args = [args[0]]
            elif isinstance(args[0], list) or isinstance(args[0], tuple):
                args = args[0]
            else:
                raise TypeError(
                    "Single input argument must be a GSObject or list of them."
                )
        # else args is already the list of objects

        real_space = kwargs.pop("real_space", None)
        gsparams = kwargs.pop("gsparams", None)
        self._propagate_gsparams = kwargs.pop("propagate_gsparams", True)

        # Make sure there is nothing left in the dict.
        if kwargs:
            raise TypeError(
                "Convolution constructor got unexpected keyword argument(s): %s"
                % kwargs.keys()
            )

        # Check whether to perform real space convolution...
        # Start by checking if all objects have a hard edge.
        hard_edge = True
        for obj in args:
            if not isinstance(obj, GSObject):
                raise TypeError(
                    "Arguments to Convolution must be GSObjects, not %s" % obj
                )
            if not obj.has_hard_edges:
                hard_edge = False

        if real_space is None:
            # The automatic determination is to use real_space if 2 items, both with hard edges.
            if len(args) <= 2:
                real_space = hard_edge
            else:
                real_space = False
        elif bool(real_space) != real_space:
            raise TypeError("real_space must be a boolean")

        # Warn if doing DFT convolution for objects with hard edges
        if not real_space and hard_edge:
            if len(args) == 2:
                galsim_warn(
                    "Doing convolution of 2 objects, both with hard edges. "
                    "This might be more accurate"
                )  ##### and/or faster using real_space=True")
            else:
                galsim_warn(
                    "Doing convolution where all objects have hard edges. "
                    "There might be some inaccuracies due to ringing in k-space."
                )
        if real_space:
            # Can't do real space if nobj > 2
            if len(args) > 2:
                galsim_warn(
                    "Real-space convolution of more than 2 objects is not implemented. "
                    "Switching to DFT method."
                )
                real_space = False

            # Also can't do real space if any object is not analytic, so check for that.
            else:
                for obj in args:
                    if not obj.is_analytic_x:
                        galsim_warn(
                            "A component to be convolved is not analytic in real space. "
                            "Cannot use real space convolution. Switching to DFT method."
                        )
                        real_space = False
                        break

        # Save the construction parameters (as they are at this point) as attributes so they
        # can be inspected later if necessary.
        if bool(real_space):
            raise NotImplementedError("Real space convolutions are not implemented")
        self._real_space = bool(real_space)

        # Figure out what gsparams to use
        if gsparams is None:
            # If none is given, take the most restrictive combination from the obj_list.
            self._gsparams = GSParams.combine([obj.gsparams for obj in args])
        else:
            # If something explicitly given, then use that.
            self._gsparams = GSParams.check(gsparams)

        # Apply gsparams to all in obj_list.
        if self._propagate_gsparams:
            self._obj_list = [obj.withGSParams(self._gsparams) for obj in args]
        else:
            self._obj_list = args

        # Save the list of objects
        self._params = {"obj_list": self._obj_list}

    @property
    def obj_list(self):
        """The list of objects being convolved."""
        return self._obj_list

    @property
    def real_space(self):
        """Whether this `Convolution` should be drawn using real-space convolution rather
        than FFT convolution.
        """
        return self._real_space

    def withGSParams(self, gsparams=None, **kwargs):
        """Create a version of the current object with the given gsparams

        .. note::

            Unless you set ``propagate_gsparams=False``, this method will also update the gsparams
            of each object being convolved.
        """
        if gsparams == self.gsparams:
            return self
        from copy import copy

        ret = copy(self)
        ret._gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        if self._propagate_gsparams:
            ret._obj_list = [obj.withGSParams(ret._gsparams) for obj in self.obj_list]
        return ret

    def __eq__(self, other):
        return self is other or (
            isinstance(other, Convolution)
            and self.obj_list == other.obj_list
            and self.real_space == other.real_space
            and self.gsparams == other.gsparams
            and self._propagate_gsparams == other._propagate_gsparams
        )

    def __hash__(self):
        return hash(
            (
                "galsim.Convolution",
                tuple(self.obj_list),
                self.real_space,
                self.gsparams,
                self._propagate_gsparams,
            )
        )

    def __repr__(self):
        return (
            "galsim.Convolution(%r, real_space=%r, gsparams=%r, propagate_gsparams=%r)"
            % (
                self.obj_list,
                self.real_space,
                self.gsparams,
                self._propagate_gsparams,
            )
        )

    def __str__(self):
        str_list = [str(obj) for obj in self.obj_list]
        s = "galsim.Convolve(%s" % (", ".join(str_list))
        if self.real_space:
            s += ", real_space=True"
        s += ")"
        return s

    # JEC not sure if it is used
    # def _prepareDraw(self):
    #    for obj in self.obj_list:
    #        obj._prepareDraw()

    @property
    def _maxk(self):
        maxk_list = [obj.maxk for obj in self.obj_list]
        return jnp.min(jnp.array(maxk_list))

    @property
    def _stepk(self):
        # This is approximate.  stepk ~ 2pi/R
        # Assume R_final^2 = Sum(R_i^2)
        # So 1/stepk^2 = 1/Sum(1/stepk_i^2)
        inv_stepksq_list = [obj.stepk ** (-2) for obj in self.obj_list]
        return 1.0 / jnp.sqrt(jnp.sum(jnp.array(inv_stepksq_list)))

    @property
    def _has_hard_edges(self):
        return len(self.obj_list) == 1 and self.obj_list[0].has_hard_edges

    @property
    def _is_axisymmetric(self):
        axi_list = [obj.is_axisymmetric for obj in self.obj_list]
        return jnp.alltrue(jnp.array(axi_list)).item()

    @property
    def _is_analytic_x(self):
        if len(self.obj_list) == 1:
            return self.obj_list[0].is_analytic_x
        elif self.real_space and len(self.obj_list) == 2:
            ax_list = [obj.is_analytic_x for obj in self.obj_list]
            return jnp.alltrue(jnp.arry(ax_list)).item()  # return a bool
        else:
            return False

    @property
    def _is_analytic_k(self):
        ak_list = [obj.is_analytic_k for obj in self.obj_list]
        return jnp.alltrue(jnp.array(ak_list)).item()  # return a bool

    @property
    def _centroid(self):
        cen_list = [obj.centroid for obj in self.obj_list]
        return sum(
            cen_list[1:], cen_list[0]
        )  # gives a Position object with x=sum_i x_i and y=sum_i y_i

    @property
    def _flux(self):
        flux_list = [obj.flux for obj in self.obj_list]
        return jnp.prod(jnp.array(flux_list)).item()  # return a float

    @property
    def _positive_flux(self):
        raise NotImplementedError("Not implemented")

    @property
    def _negative_flux(self):
        raise NotImplementedError("Not implemented")

    @property
    def _flux_per_photon(self):
        raise NotImplementedError("Not implemented")

    @property
    def _max_sb(self):
        # This one is probably the least accurate of all the estimates of maxSB.
        # The calculation is based on the exact value for Gaussians.
        #     maxSB = flux / 2pi sigma^2
        # When convolving multiple Gaussians together, the sigma^2 values add:
        #     sigma_final^2 = Sum_i sigma_i^2
        # from which we can calculate
        #     maxSB = flux_final / 2pi sigma_final^2
        # or
        #     maxSB = flux_final / Sum_i (flux_i / maxSB_i)
        #
        # For non-Gaussians, this procedure will tend to produce an over-estimate of the
        # true maximum SB.  Non-Gaussian profiles tend to have peakier parts which get smoothed
        # more than the Gaussian does.  So this is likely to be too high, which is acceptable.
        area_list = [obj.flux / obj.max_sb for obj in self.obj_list]
        return self.flux / jnp.sum(jnp.array(area_list))

    def _xValue(self, pos):
        raise NotImplementedError("Not implemented")

    def _kValue(self, kpos):
        kv_list = [
            obj.kValue(kpos) for obj in self.obj_list
        ]  # In GalSim one uses obj.kValue
        return jnp.prod(jnp.array(kv_list))

    def _drawReal(self, image, jac=None, offset=(0.0, 0.0), flux_scaling=1.0):
        raise NotImplementedError("Not implemented")

    def _shoot(self, photons, rng):
        raise NotImplementedError("Not implemented")

    def _drawKImage(self, image, jac=None):
        image = self.obj_list[0]._drawKImage(image, jac)
        if len(self.obj_list) > 1:
            for obj in self.obj_list[1:]:
                image *= obj._drawKImage(image, jac)
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
            "real_space": self._real_space,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        return cls(children[0]["obj_list"], **aux_data)
