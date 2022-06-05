import jax.numpy as jnp

from jax_galsim.gsobject import GSObject
from jax_galsim.gsparams import GSParams

import galsim as _galsim
from jax._src.numpy.util import _wraps
from jax.tree_util import register_pytree_node_class


@_wraps(_galsim.Exponential)
@register_pytree_node_class
class Exponential(GSObject):

    # The half-light-radius is not analytic, but can be calculated numerically
    # by iterative solution of equation:
    #     (re / r0) = ln[(re / r0) + 1] + ln(2)
    _hlr_factor = 1.6783469900166605
    _one_third = 1./3.
    _inv_twopi = 0.15915494309189535

    _has_hard_edges = False
    _is_axisymmetric = True
    _is_analytic_x = True
    _is_analytic_k = True

    def __init__(self, half_light_radius=None, scale_radius=None, flux=1., gsparams=None):

        # Checking gsparams
        gsparams = GSParams.check(gsparams)

        if half_light_radius is not None:
            if scale_radius is not None:
                raise _galsim.GalSimIncompatibleValuesError(
                    "Only one of scale_radius and half_light_radius may be specified",
                    half_light_radius=half_light_radius, scale_radius=scale_radius)
            else:
                super().__init__(half_light_radius=half_light_radius, flux=flux, gsparams=gsparams)

        elif scale_radius is None:
            raise _galsim.GalSimIncompatibleValuesError(
                "Either scale_radius or half_light_radius must be specified",
                half_light_radius=half_light_radius, scale_radius=scale_radius)
        else:
            super().__init__(scale_radius=scale_radius, flux=flux, gsparams=gsparams)

        self._r0 = self.scale_radius
        self._inv_r0 = 1./self._r0
        self._norm = self.flux * Exponential._inv_twopi * self._inv_r0**2

    @property
    def scale_radius(self):
        """The scale radius of the profile.
        """
        if 'half_light_radius' in self.params:
            return self.params['half_light_radius'] / Exponential._hlr_factor
        else:
            return self.params['scale_radius']

    @property
    def half_light_radius(self):
        """The half-light radius of the profile.
        """
        if 'half_light_radius' in self.params:
            return self.params['half_light_radius']
        else:
            return self.params['half_light_radius'] * Exponential._hlr_factor

    def __hash__(self):
        return hash(("galsim.Exponential", self.scale_radius, self.flux, self.gsparams))

    def __repr__(self):
        return 'galsim.Exponential(scale_radius=%r, flux=%r, gsparams=%r)' % (
            self.scale_radius, self.flux, self.gsparams)

    def __str__(self):
        s = 'galsim.Exponential(scale_radius=%s' % self.scale_radius
        if self.flux != 1.0:
            s += ', flux=%s' % self.flux
        s += ')'
        return s

    @property
    def _maxk(self):
        return (self.gsparams.maxk_threshold ** -Exponential._one_third) / self.scale_radius

    @property
    def _stepk(self):
        # Calculate stepk:
        # int( exp(-r) r, r=0..R) = (1 - exp(-R) - Rexp(-R))
        # Fraction excluded is thus (1+R) exp(-R)
        # A fast solution to (1+R)exp(-R) = x:
        # log(1+R) - R = log(x)
        # R = log(1+R) - log(x)
        logx = jnp.log(self.gsparams.folding_threshold)
        R = -logx
        for i in range(3):
            R = jnp.log(1.+R) - logx
        # Make sure it is at least 5 hlr
        # half-light radius = 1.6783469900166605 * r0
        hlr = 1.6783469900166605
        R = jnp.max(jnp.array([R, self.gsparams.stepk_minimum_hlr*hlr]))
        return jnp.pi / R

    @property
    def _max_sb(self):
        return self._norm

    def _xValue(self, pos):
        r = jnp.sqrt(pos.x**2 + pos.y**2)
        return self._norm * jnp.exp(-r * self._inv_r0)

    def _kValue(self, kpos):
        ksqp1 = (kpos.x**2 + kpos.y**2) * self._r0**2 + 1.
        return self._flux / (ksqp1 * jnp.sqrt(ksqp1))

    def withFlux(self, flux):
        return Exponential(scale_radius=self.scale_radius, flux=flux, gsparams=self.gsparams)
