import galsim as _galsim
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from jax_galsim.core.utils import implements
from jax_galsim.gsobject import GSObject
from jax_galsim.gsparams import GSParams
from jax_galsim.position import PositionD
from jax_galsim.random import BaseDeviate


@implements(
    _galsim.Add, lax_description="Does not support `ChromaticObject` at this point."
)
def Add(*args, **kwargs):
    return Sum(*args, **kwargs)


@implements(
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
    @implements(_galsim.Sum.flux)
    def flux(self):
        flux_list = jnp.array([obj.flux for obj in self.obj_list])
        return jnp.sum(flux_list)

    @implements(_galsim.Sum.withGSParams)
    def withGSParams(self, gsparams=None, **kwargs):
        if gsparams == self.gsparams:
            return self
        ret = self.__class__(
            self.params["obj_list"],
            gsparams=GSParams.check(gsparams, self.gsparams, **kwargs),
            propagate_gsparams=self._propagate_gsparams,
        )
        return ret

    def __hash__(self):
        return hash(
            (
                "galsim.Sum",
                tuple(self.obj_list),
                self.gsparams,
                self._propagate_gsparams,
            )
        )

    def __repr__(self):
        return "galsim.Sum(%r, gsparams=%r, propagate_gsparams=%r)" % (
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
        cen_x_arr = jnp.array([obj.centroid.x * obj.flux for obj in self.obj_list])
        cen_y_arr = jnp.array([obj.centroid.y * obj.flux for obj in self.obj_list])
        return PositionD(jnp.sum(cen_x_arr) / self.flux, jnp.sum(cen_y_arr) / self.flux)

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

    @property
    def _positive_flux(self):
        pflux_list = jnp.array([obj.positive_flux for obj in self.obj_list])
        return jnp.sum(pflux_list)

    @property
    def _negative_flux(self):
        nflux_list = jnp.array([obj.negative_flux for obj in self.obj_list])
        return jnp.sum(nflux_list)

    @property
    def _flux_per_photon(self):
        return self._calculate_flux_per_photon()

    def _shoot(self, photons, rng):
        tot_flux = self.positive_flux + self.negative_flux
        fluxes = jnp.array(
            [obj.positive_flux + obj.negative_flux for obj in self.obj_list]
        )
        # for a sum of objects, we use a slightly different approach than galsim did
        # as of version 2.5
        # galsim uses a binomial distribution to compute the number of photons per object
        # we take an equivalent but different approach in order to use fixed size arrays
        # of photons. it means we draw more photons but the code is JIT compilable and a bit simpler
        #
        # this all works as follows:
        #
        #   - for each photon, we draw from a categorical distribution with probabilities
        #     proportional to the total absolute fluxes of the objects.
        #   - we then shoot the photons from each object and rescale the fluxes (see comment below)
        #   - finally, we get the photons that correspond to this object in the cetegorical distribution
        #     and assign them to the photons object there is a special private method on the
        #     PhotonArray that does this assignment
        #
        # one nice thing about this is that the photons come out pre-shuffled and so we don't have
        # to mark them as correlated.
        rng = BaseDeviate(rng)
        key = rng._state.split_one()
        cat_inds = jax.random.choice(
            key,
            len(self.obj_list),
            shape=(len(photons),),
            replace=True,
            p=fluxes / tot_flux,
        )
        for i, obj in enumerate(self.obj_list):
            pa = obj.shoot(photons.size(), rng=rng)
            # now we rescale the fluxes of the photons
            # in galsim, photons end up with a flux that is
            #
            #     fluxes[i] / thisN * tot_flux / photons.size() * thisN / fluxes[i]
            #       = tot_flux / photons.size()
            #
            # our photons start with a flux of
            #
            #     flux[i] / photons.size()
            #
            # so we scale by a factor of
            #
            #     _scale_fac = tot_flux / fluxes[i]
            _scale_fac = tot_flux / fluxes[i]
            pa.scaleFlux(_scale_fac)
            photons._assign_from_categorical_index(cat_inds, i, pa)

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
