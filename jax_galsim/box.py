import jax.numpy as jnp

from jax_galsim.gsobject import GSObject
from jax_galsim.gsparams import GSParams
from jax_galsim.core.draw import draw_by_xValue, draw_by_kValue

import galsim as _galsim
from jax._src.numpy.util import _wraps
from jax.tree_util import register_pytree_node_class

@_wraps(_galsim.Box)
@register_pytree_node_class
class Box(GSObject):

    _has_hard_edges = True
    _is_axisymmetric = False
    _is_analytic_x = True
    _is_analytic_k = True

    def __init__(self, width, height, flux=1., gsparams=None):
        gsparams = GSParams.check(gsparams)

        super().__init__(width=width, height=height, flux=flux, gsparams=gsparams)
        
        self._norm = flux / (width * height)
        self._wo2 = 0.5*width;
        self._ho2 = 0.5*height;
        self._wo2pi = width/(2.*jnp.pi);
        self._ho2pi = height/(2.*jnp.pi);


        self._minL = width
        self._maxL = height
        if width > height:
            self._minL = height
            self._maxL = width


    @property
    def width(self):
        """The width of the `Box`.
        """
        return self.params["width"]

    @property
    def height(self):
        """The height of the `Box`.
        """
        return self.params["height"]

    def __hash__(self):
        return hash(("galsim.Box", self.width, self.height, self.flux, self.gsparams))

    def __repr__(self):
        return 'galsim.Box(width=%r, height=%r, flux=%r, gsparams=%r)'%(
            self.width, self.height, self.flux, self.gsparams)

    def __str__(self):
        s = 'galsim.Box(width=%s, height=%s'%(self.width, self.height)
        if self.flux != 1.0:
            s += ', flux=%s'%self.flux
        s += ')'
        return s

    @property
    def _maxk(self):
        return 2. / (self.gsparams.maxk_threshold * self._minL)

    @property
    def _stepk(self):
        return jnp.pi / self._maxL

    @property
    def _max_sb(self):
        return self._norm

    def _xValue(self, pos):
        return jnp.where(2.*jnp.abs(pos.x) < self.width, 
                         jnp.where(2.*jnp.abs(pos.y) < self.height,
                          self._norm,
                          0.),
                         0.)

    def _kValue(self, kpos):
        return self.flux * jnp.sinc(kpos.x*self._wo2pi)*jnp.sinc(kpos.y*self._ho2pi)


    def _drawReal(self, image, jac=None, offset=(0.0, 0.0), flux_scaling=1.0):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_xValue(self, image, _jac, jnp.asarray(offset), flux_scaling)

    def _drawKImage(self, image, jac=None):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_kValue(self,image, _jac)


    def withFlux(self, flux):
        return Box(width=self.width, height=self.height, flux=flux, gsparams=self.gsparams)

@_wraps(_galsim.Pixel)
@register_pytree_node_class
class Pixel(Box):
    def __init__(self, scale, flux=1., gsparams=None):
        super(Pixel, self).__init__(width=scale, height=scale, flux=flux, gsparams=gsparams)
    
    @property
    def scale(self):
        """The linear scale size of the `Pixel`.
        """
        return self.width
    
    def __repr__(self):
        return 'galsim.Pixel(scale=%r, flux=%r, gsparams=%r)'%(
            self.scale, self.flux, self.gsparams)

    def __str__(self):
        s = 'galsim.Pixel(scale=%s'%self.scale
        if self.flux != 1.0:
            s += ', flux=%s'%self.flux
        s += ')'
        return s

    def withFlux(self, flux):
        return Pixel(scale=self.scale, flux=flux, gsparams=self.gsparams)

