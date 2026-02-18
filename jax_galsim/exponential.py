from functools import lru_cache

import galsim as _galsim
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from jax_galsim.core.draw import draw_by_kValue, draw_by_xValue
from jax_galsim.core.utils import ensure_hashable, implements
from jax_galsim.gsobject import GSObject
from jax_galsim.random import UniformDeviate


@implements(_galsim.Exponential)
@register_pytree_node_class
class Exponential(GSObject):
    # The half-light-radius is not analytic, but can be calculated numerically
    # by iterative solution of equation:
    #     (re / r0) = ln[(re / r0) + 1] + ln(2)
    _hlr_factor = 1.6783469900166605
    _one_third = 1.0 / 3.0
    _inv_twopi = 0.15915494309189535

    _has_hard_edges = False
    _is_axisymmetric = True
    _is_analytic_x = True
    _is_analytic_k = True

    def __init__(
        self, half_light_radius=None, scale_radius=None, flux=1.0, gsparams=None
    ):
        if half_light_radius is not None:
            if scale_radius is not None:
                raise _galsim.GalSimIncompatibleValuesError(
                    "Only one of scale_radius and half_light_radius may be specified",
                    half_light_radius=half_light_radius,
                    scale_radius=scale_radius,
                )
            else:
                super().__init__(
                    scale_radius=half_light_radius / Exponential._hlr_factor,
                    flux=flux,
                    gsparams=gsparams,
                )

        elif scale_radius is None:
            raise _galsim.GalSimIncompatibleValuesError(
                "Either scale_radius or half_light_radius must be specified",
                half_light_radius=half_light_radius,
                scale_radius=scale_radius,
            )
        else:
            super().__init__(scale_radius=scale_radius, flux=flux, gsparams=gsparams)

    @property
    @implements(_galsim.Exponential.scale_radius)
    def scale_radius(self):
        return self.params["scale_radius"]

    @property
    def _r0(self):
        return self.scale_radius

    @property
    def _inv_r0(self):
        return 1.0 / self._r0

    @property
    def _norm(self):
        return self.flux * Exponential._inv_twopi * self._inv_r0**2

    @property
    @implements(_galsim.Exponential.half_light_radius)
    def half_light_radius(self):
        return self.params["scale_radius"] * Exponential._hlr_factor

    def __hash__(self):
        return hash(
            (
                "galsim.Exponential",
                ensure_hashable(self.scale_radius),
                ensure_hashable(self.flux),
                self.gsparams,
            )
        )

    def __repr__(self):
        return "galsim.Exponential(scale_radius=%r, flux=%r, gsparams=%r)" % (
            ensure_hashable(self.scale_radius),
            ensure_hashable(self.flux),
            self.gsparams,
        )

    def __str__(self):
        s = "galsim.Exponential(scale_radius=%s" % ensure_hashable(self.scale_radius)
        s += ", flux=%s" % ensure_hashable(self.flux)
        s += ")"
        return s

    @property
    def _maxk(self):
        _maxk = self.gsparams.maxk_threshold**-Exponential._one_third
        return _maxk / self.scale_radius

    @property
    def _stepk(self):
        # The content of this function is inherited from the GalSim C++ layer
        # https://github.com/GalSim-developers/GalSim/blob/ece3bd32c1ae6ed771f2b489c5ab1b25729e0ea4/src/SBExponential.cpp#L530
        # https://github.com/GalSim-developers/GalSim/blob/ece3bd32c1ae6ed771f2b489c5ab1b25729e0ea4/src/SBExponential.cpp#L97
        # Calculate stepk:
        # int( exp(-r) r, r=0..R) = (1 - exp(-R) - Rexp(-R))
        # Fraction excluded is thus (1+R) exp(-R)
        # A fast solution to (1+R)exp(-R) = x:
        # log(1+R) - R = log(x)
        # R = log(1+R) - log(x)
        logx = jnp.log(self.gsparams.folding_threshold)
        R = -logx
        for i in range(3):
            R = jnp.log(1.0 + R) - logx
        # Make sure it is at least 5 hlr
        # half-light radius = 1.6783469900166605 * r0
        hlr = 1.6783469900166605
        R = jnp.max(jnp.array([R, self.gsparams.stepk_minimum_hlr * hlr]))
        return jnp.pi / R * self._inv_r0

    @property
    def _max_sb(self):
        return self._norm

    def _xValue(self, pos):
        r = jnp.sqrt(pos.x**2 + pos.y**2)
        return self._norm * jnp.exp(-r * self._inv_r0)

    def _kValue(self, kpos):
        ksqp1 = (kpos.x**2 + kpos.y**2) * self._r0**2 + 1.0
        return self.flux / (ksqp1 * jnp.sqrt(ksqp1))

    def _drawReal(self, image, jac=None, offset=(0.0, 0.0), flux_scaling=1.0):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_xValue(self, image, _jac, jnp.asarray(offset), flux_scaling)

    def _drawKImage(self, image, jac=None):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_kValue(self, image, _jac)

    @implements(_galsim.Exponential.withFlux)
    def withFlux(self, flux):
        return Exponential(
            scale_radius=self.scale_radius, flux=flux, gsparams=self.gsparams
        )

    @implements(_galsim.Exponential._shoot)
    def _shoot(self, photons, rng):
        ud = UniformDeviate(rng)

        u = ud.generate(
            photons.x
        )  # this does not fill arrays like in galsim so is safe
        _u_cdf, _cdf = _shoot_cdf(self.gsparams.shoot_accuracy)
        # this interpolation inverts the CDF
        u = jnp.interp(u, _cdf, _u_cdf)
        # this converts from u (see above) to r and scales by the actual size of
        # the object r0.
        r = -jnp.log(1.0 - u) * self._r0

        ang = (
            ud.generate(photons.x) * 2.0 * jnp.pi
        )  # this does not fill arrays like in galsim so is safe
        photons.x = r * jnp.cos(ang)
        photons.y = r * jnp.sin(ang)
        photons.flux = self.flux / photons.size()


@lru_cache(maxsize=8)
def _shoot_cdf(shoot_accuracy):
    """This routine produces a CPU-side cache of the CDF that is embedded
    into JIT-compiled code as needed."""
    # Comments on the math here:
    #
    # We are looking to draw from a distribution that is r * exp(-r).
    # This distribution is the radial PDF of an Exponential profile.
    # The factor of r comes from the area element r * dr.
    #
    # We can compute the CDF of this distribution analytically, but we cannot
    # invert the CDF in closed form. Thus we invert it numerically using a table.
    #
    # One final detail is that we want the inversion to be accurate and are using
    # linear interpolation. Thus we use a change of variables r = -ln(1 - u)
    # to make the CDF more linear and map it's domain to [0, 1) instead of [0, inf).
    #
    # Putting this all together, we get
    #
    #     r * exp(-r) dr = -ln(1-u) (1-u) dr/du du
    #                    = -ln(1-u) (1-u)  * 1 / (1-u)
    #                    = -ln(1-u)
    #
    # The new range of integration is u = 0 to u = 1. Thus the CDF is
    #
    #     CDF = -int_0^u ln(1-u') du'
    #         =  u - (u - 1) ln(1 - u)
    #
    # The final detail is that galsim defines a shoot accuracy and draws photons
    # between r = 0 and rmax = -log(shoot_accuracy). Thus we normalize the CDF to
    # its value at umax = 1 - exp(-rmax) and then finally invert the CDF numerically.
    _rmax = -np.log(shoot_accuracy)
    _umax = 1.0 - np.exp(-_rmax)
    _u_cdf = np.linspace(0, _umax, 10000)
    _cdf = _u_cdf - (_u_cdf - 1) * np.log(1 - _u_cdf)
    _cdf /= _cdf[-1]
    return _u_cdf, _cdf
