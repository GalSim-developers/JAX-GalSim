import galsim as _galsim
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp
from jax._src.numpy.util import _wraps
from jax.tree_util import Partial as partial
from jax.tree_util import register_pytree_node_class

from jax_galsim.core.draw import draw_by_kValue, draw_by_xValue
from jax_galsim.core.integrate import ClenshawCurtisQuad, quad_integral
from jax_galsim.core.utils import bisect_for_root, ensure_hashable
from jax_galsim.gsobject import GSObject
from jax_galsim.position import PositionD

@jax.jit
def _Knu(nu, x):
    """Modified Bessel 2nd kind """
    return tfp.substrates.jax.math.bessel_kve(nu * 1.0, x) / jnp.exp(jnp.abs(x))

@jax.jit
def _gamma(nu):
    """Gamma(nu) """
    return jnp.exp(jax.lax.lgamma(nu * 1.0))
@jax.jit
def _gammap1(nu):
    """Gamma(nu+1) """
    return _gamma(nu+1.0)

@jax.jit
def fsmallz(z,nu):
    def z2lz(z):
        # z^2 * log(z)
        return jnp.where(z<=1e-16,0.,z*z*jnp.log(z))
    def fnu(z,nu):
        z2 = z*z
        z4 = z2*z2
        c1 = -jnp.power(2.0,-4.0-nu)
        c2 = _gamma(-2.0-nu)
        c3 = c1 * c2 * (8.0 + 4.0*nu + z2) * jnp.power(z,2.0*(1.0+nu))
        c4 = jnp.power(2.0,-5.0+nu)
        c5= _gammap1(nu) /(nu * (-1.0+nu))
        c6 = c4 * c5 * (32.0 * nu * (-1.0+nu) - 8.0 * z2 * (-1.0+nu) + z4)
        return c3 + c6
    def f0(z): 
        #nu = 0
        z2 = z*z
        z4 = z2*z2
        c0 = z2lz(z)# z^2 log(z)
        c1 = 0.30796575782920622441
        c2 = 0.08537071972865077805
        return 1.0 - c1 * z2 - c2 * z4 + c0 * (0.5 + 0.0625 * z2)
    def f1(z):
        #nu = 1
        c1 = 0.10824143945730155610
        z2 = z*z
        z4 = z2*z2
        c0 = z2lz(z) * z2 # z^4*log(z)
        return 2.0 - 0.5 * z2 + c1 * z4 - 0.125 * c0
    def f2(z):
        #nu = 2
        z2 = z*z
        z4 = z2*z2
        return 8.0 - z2 + 0.125 * z4
    def f3(z):
        #nu = 3
        z2 = z*z
        z4 = z2*z2
        return 48.0 - 4 * z2 + 0.25 * z4
    def f4(z):
        #nu = 4
        z2 = z*z
        z4 = z2*z2
        return 384. - 24. * z2 + z4
    return jnp.select(
        [nu == 0, nu==1, nu==2, nu==3, nu==4],
        [f0(z),f1(z),f2(z),f3(z),f4(z)],
        default = fnu(z,nu))
    
@jax.jit
def f(z,nu):
    """ Return z^(nu+1) K_{nu+1}(z) 
        Spergel index nu in [-0.85, 4.]
    """
    return jnp.where(z<=0.1, fsmallz(z,nu), jnp.power(z,nu+1.0) * _Knu(nu+1.0,z))

@jax.jit
def fluxfractionFunc(z,nu, alpha):
    """ Return  z^(nu+1) K_{nu+1}(z) / (2^nu Gamma(nu+1)) - (1-alpha)
    """
    return f(z,nu) / (jnp.power(2.,nu) * _gammap1(nu)) - (1.0 - alpha)

@jax.jit
def calculateFluxRadius(alpha, nu):
    """Return radius R enclosing flux fraction alpha  in unit of the scale radius r0
        
        Method: Solve  F(R/r0=z)/Flux - alpha = 0 using bisection algorithm

        F(R)/F =  int( 1/(2^nu Gamma(nu+1)) (r/r0)^(nu+1) K_nu(r/r0) dr/r0; r=0..R) = alpha
        =>
        z=R/r0 such that
        z^(nu+1) K_{nu+1}(z) / (2^nu Gamma(nu+1)) = 1-alpha
               
        Typical use cases: 
         o alpha = 1/2 => R = Half-Light-Radius, 
         o alpha = 1 - folding-thresold => R used for stepk computation 

         nu: the Spergel index
         
         nb. it is supposed that nu is in [-0.85, 4.0] checked in the Spergel class init 
    """
    return bisect_for_root(partial(fluxfractionFunc,nu=nu,alpha=alpha), 0.0, 5.0, niter=75)

@_wraps(_galsim.Spergel)
@register_pytree_node_class
class Spergel(GSObject):
    _has_hard_edges = False
    _is_axisymmetric = True
    _is_analytic_x = True
    _is_analytic_k = True

    _minimum_nu = -0.85
    _maximum_nu = 4.0

    def __init__(self, 
        nu, 
        half_light_radius=None, 
        scale_radius=None,
        flux=1., 
        gsparams=None
    ):
        # Todo: how to implement this check
        #if self._nu < Spergel._minimum_nu:
        #    raise _galsim.GalSimRangeError("Requested Spergel index is too small",
        #                           self._nu, Spergel._minimum_nu, Spergel._maximum_nu)
        #if self._nu > Spergel._maximum_nu:
        #    raise _galsimGalSimRangeError("Requested Spergel index is too large",
        #                           self._nu, Spergel._minimum_nu, Spergel._maximum_nu)

        # Parse the radius options
        if half_light_radius is not None:
            if scale_radius is not None:
                raise _galsim.GalSimIncompatibleValuesError(
                    "Only one of scale_radius, half_light_radius may be specified",
                    half_light_radius=half_light_radius,
                    scale_radius=scale_radius,
                )
            else:
                super().__init__(
                    nu=nu,
                    scale_radius=half_light_radius/calculateFluxRadius(0.5,nu),
                    flux=flux,
                    gsparams=gsparams,
                )
        elif scale_radius is None:
            raise _galsim.GalSimIncompatibleValuesError(
                "One of scale_radius, half_light_radius must be specified",
                half_light_radius=half_light_radius,
                scale_radius=scale_radius,
            )
        else:
            super().__init__(
                nu=nu,
                scale_radius=scale_radius,
                flux=flux,
                gsparams=gsparams,
            )

    @property
    def nu(self):
        """The Spergel index, nu
        """
        return self._params["nu"]
      

    @property
    def scale_radius(self):
        """The scale radius of this `Spergel` profile."""
        return self.params["scale_radius"]
    
    @property
    def _r0(self):
        return self.scale_radius

    @property
    def _inv_r0(self):
        return 1.0 / self._r0
    
    @property
    def _r0_sq(self):
        return self._r0 * self._r0
    
    @property
    def _inv_r0_sq(self):
        return self._inv_r0 * self._inv_r0

    @property
    def half_light_radius(self):
        """The half-light radius of this `Spergel` profile."""
        return self._r0 * calculateFluxRadius(0.5,self.nu)
    
    @property
    def _shootxnorm(self):
        return 1./(2. * jnp.pi * jnp.power(2.,self.nu) + _gammap1(self.nu))
    
    @property
    def _xnorm(self):
        return self._shootxnorm * self.flux * self._inv_r0_sq

    
    def __hash__(self):
        return hash(
            (
                "galsim.Spergel",
                ensure_hashable(self.nu),
                ensure_hashable(self.scale_radius),
                ensure_hashable(self.flux),
                self.gsparams,
            )
        )
    
    def __repr__(self):
        return (
            "galsim.Spergel(nu=%r, scale_radius=%r, flux=%r, gsparams=%r)"
            % (
                ensure_hashable(self.beta),
                ensure_hashable(self.scale_radius),
                ensure_hashable(self.flux),
                self.gsparams,
            )
        )
    
    def __str__(self):
        s = "galsim.Spergel(nu=%s, half_light_radius=%s" % (
            ensure_hashable(self.nu), 
            ensure_hashable(self.half_light_radius),
        )
        if self.flux != 1.0:
            s += ", flux=%s" % ensure_hashable(self.flux)
        s += ')'
        return s
    
    @property
    def _maxk(self):
        """(1+ (k r0)^2)^(-1-nu) = maxk_threshold """
        return jnp.sqrt(jnp.power(self.gsparams.maxk_threshold,-1./(1.+self.nu)) - 1.0) / self._r0
    
    @property
    def _stepk(self):
        R = calculateFluxRadius(1.0 - self.gsparams.folding_threshold, self.nu) * self._r0
        # Go to at least 5*hlr
        R = jnp.maximum(R, self.gsparams.stepk_minimum_hlr * self.half_light_radius)
        return jnp.pi / R
    
    @property
    def _max_sb(self):
        return self._norm