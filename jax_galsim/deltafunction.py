import galsim as _galsim
import jax
import jax.numpy as jnp
from jax._src.numpy.util import _wraps
from jax.tree_util import register_pytree_node_class

from jax_galsim.core.draw import draw_by_kValue, draw_by_xValue
from jax_galsim.core.utils import ensure_hashable
from jax_galsim.gsobject import GSObject


@_wraps(_galsim.DeltaFunction)
@register_pytree_node_class
class DeltaFunction(GSObject):
    _opt_params = {"flux": float}

    _mock_inf = (
        1.0e300  # Some arbitrary very large number to use when we need infinity.
    )

    _has_hard_edges = False
    _is_axisymmetric = True
    _is_analytic_x = False
    _is_analytic_k = True

    def __init__(self, flux=1.0, gsparams=None):
        super().__init__(flux=flux, gsparams=gsparams)

    def __hash__(self):
        return hash(("galsim.DeltaFunction", ensure_hashable(self.flux), self.gsparams))

    def __repr__(self):
        return "galsim.DeltaFunction(flux=%r, gsparams=%r)" % (
            ensure_hashable(self.flux),
            self.gsparams,
        )

    def __str__(self):
        s = "galsim.DeltaFunction("
        if self.flux != 1.0:
            s += "flux=%s" % self.flux
        s += ")"
        return s

    @property
    def _maxk(self):
        return DeltaFunction._mock_inf

    @property
    def _stepk(self):
        return DeltaFunction._mock_inf

    @property
    def _max_sb(self):
        return DeltaFunction._mock_inf

    def _xValue(self, pos):
        return jax.lax.cond(
            jnp.array(pos.x == 0.0, dtype=bool)
            & jnp.array(pos.y == 0.0, dtype=bool),
            lambda *a: DeltaFunction._mock_inf,
            lambda *a: 0.0,
        )

    def _kValue(self, kpos):
        # this is a wasteful and fancy way to get the shape to broadcast to
        # to match the input kpos
        return self.flux + kpos.x * (0.0 + 0.0j)

    def _shoot(self, photons, rng):
        flux_per_photon = self.flux / photons.size()
        photons.x = 0.0
        photons.y = 0.0
        photons.flux = flux_per_photon

    def _drawReal(self, image, jac=None, offset=(0.0, 0.0), flux_scaling=1.0):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_xValue(self, image, _jac, jnp.asarray(offset), flux_scaling)

    def _drawKImage(self, image, jac=None):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_kValue(self, image, _jac)

    @_wraps(_galsim.DeltaFunction.withFlux)
    def withFlux(self, flux):
        return DeltaFunction(flux=flux, gsparams=self.gsparams)
