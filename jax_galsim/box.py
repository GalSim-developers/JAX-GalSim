import galsim as _galsim
import jax.numpy as jnp
from jax._src.numpy.util import implements
from jax.tree_util import register_pytree_node_class

from jax_galsim.core.draw import draw_by_kValue, draw_by_xValue
from jax_galsim.core.utils import ensure_hashable
from jax_galsim.gsobject import GSObject
from jax_galsim.random import UniformDeviate


@implements(_galsim.Box)
@register_pytree_node_class
class Box(GSObject):
    _has_hard_edges = True
    _is_axisymmetric = False
    _is_analytic_x = True
    _is_analytic_k = True

    def __init__(self, width, height, flux=1.0, gsparams=None):
        super().__init__(width=width, height=height, flux=flux, gsparams=gsparams)

    @property
    def _minL(self):
        return jnp.minimum(self.width, self.height)

    @property
    def _maxL(self):
        return jnp.maximum(self.width, self.height)

    @property
    def width(self):
        """The width of the `Box`."""
        return self.params["width"]

    @property
    def height(self):
        """The height of the `Box`."""
        return self.params["height"]

    def __hash__(self):
        return hash(
            (
                "galsim.Box",
                ensure_hashable(self.width),
                ensure_hashable(self.height),
                ensure_hashable(self.flux),
                self.gsparams,
            )
        )

    def __repr__(self):
        return "galsim.Box(width=%r, height=%r, flux=%r, gsparams=%r)" % (
            ensure_hashable(self.width),
            ensure_hashable(self.height),
            ensure_hashable(self.flux),
            self.gsparams,
        )

    def __str__(self):
        s = "galsim.Box(width=%s, height=%s" % (
            ensure_hashable(self.width),
            ensure_hashable(self.height),
        )
        if self.flux != 1.0:
            s += ", flux=%s" % ensure_hashable(self.flux)
        s += ")"
        return s

    @property
    def _maxk(self):
        return 2.0 / (self.gsparams.maxk_threshold * self._minL)

    @property
    def _stepk(self):
        return jnp.pi / self._maxL

    @property
    def _max_sb(self):
        return self.flux / (self.width * self.height)

    def _xValue(self, pos):
        norm = self.flux / (self.width * self.height)
        return jnp.where(
            2.0 * jnp.abs(pos.x) < self.width,
            jnp.where(2.0 * jnp.abs(pos.y) < self.height, norm, 0.0),
            0.0,
        )

    def _kValue(self, kpos):
        _wo2pi = self.width / (2.0 * jnp.pi)
        _ho2pi = self.height / (2.0 * jnp.pi)
        return self.flux * jnp.sinc(kpos.x * _wo2pi) * jnp.sinc(kpos.y * _ho2pi)

    def _drawReal(self, image, jac=None, offset=(0.0, 0.0), flux_scaling=1.0):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_xValue(self, image, _jac, jnp.asarray(offset), flux_scaling)

    def _drawKImage(self, image, jac=None):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_kValue(self, image, _jac)

    @implements(_galsim.Box.withFlux)
    def withFlux(self, flux):
        return Box(
            width=self.width, height=self.height, flux=flux, gsparams=self.gsparams
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        return cls(
            children[0]["width"],
            children[0]["height"],
            flux=children[0]["flux"],
            **aux_data
        )

    @implements(_galsim.Box._shoot)
    def _shoot(self, photons, rng):
        ud = UniformDeviate(rng)

        # this does not fill arrays like in galsim
        photons.x = (ud.generate(photons.x) - 0.5) * self.width
        photons.y = (ud.generate(photons.y) - 0.5) * self.height
        photons.flux = self.flux / photons.size()


@implements(_galsim.Pixel)
@register_pytree_node_class
class Pixel(Box):
    def __init__(self, scale, flux=1.0, gsparams=None):
        super(Pixel, self).__init__(
            width=scale, height=scale, flux=flux, gsparams=gsparams
        )

    @property
    def scale(self):
        """The linear scale size of the `Pixel`."""
        return self.width

    def __repr__(self):
        return "galsim.Pixel(scale=%r, flux=%r, gsparams=%r)" % (
            ensure_hashable(self.scale),
            ensure_hashable(self.flux),
            self.gsparams,
        )

    def __str__(self):
        s = "galsim.Pixel(scale=%s" % ensure_hashable(self.scale)
        if self.flux != 1.0:
            s += ", flux=%s" % ensure_hashable(self.flux)
        s += ")"
        return s

    @implements(_galsim.Pixel.withFlux)
    def withFlux(self, flux):
        return Pixel(scale=self.scale, flux=flux, gsparams=self.gsparams)

    def tree_flatten(self):
        children = (self.scale, self.flux)
        aux_data = {"gsparams": self.gsparams}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        return cls(*children, **aux_data)
