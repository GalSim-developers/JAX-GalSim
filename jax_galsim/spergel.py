import galsim as _galsim
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp
from jax._src.numpy.util import _wraps
from jax.tree_util import Partial as partial
from jax.tree_util import register_pytree_node_class

from jax_galsim.core.draw import draw_by_kValue, draw_by_xValue
from jax_galsim.core.utils import bisect_for_root, ensure_hashable
from jax_galsim.gsobject import GSObject


@jax.jit
def _Knu(nu, x):
    """Modified Bessel 2nd kind"""
    return tfp.substrates.jax.math.bessel_kve(nu * 1.0, x) / jnp.exp(jnp.abs(x))


@jax.jit
def _gamma(nu):
    """Gamma(nu)"""
    return jnp.exp(jax.lax.lgamma(nu * 1.0))


@jax.jit
def _gammap1(nu):
    """Gamma(nu+1)"""
    return _gamma(nu + 1.0)


@jax.jit
def z2lz(z):
    """return z^2 * log(z)"""
    return jnp.where(z <= 1e-40, 0.0, z * z * jnp.log(z))


@jax.jit
def f0(z):
    """K_0[z] z -> 0  O(z^4)"""
    z2 = z * z
    z4 = z2 * z2
    c0 = 0.11593151565841244881
    c1 = 0.27898287891460311220
    c2 = 0.025248929932162694513
    return c0 + c1 * z2 + c2 * z4 - jnp.power(1.0 + 0.125 * z2, 2.0) * jnp.log(z)


@jax.jit
def f1(z):
    """z^1 K_1[z] z -> 0  O(z^4)"""
    z2 = z * z
    z4 = z2 * z2
    c0 = z2lz(z)  # z^2 log(z)
    c1 = 0.30796575782920622441
    c2 = 0.08537071972865077805
    return 1.0 - c1 * z2 - c2 * z4 + c0 * (0.5 + 0.0625 * z2)


@jax.jit
def f2(z):
    """z^2 K_2[z] z -> 0  O(z^4)"""
    c1 = 0.10824143945730155610
    z2 = z * z
    z4 = z2 * z2
    c0 = z2lz(z) * z2  # z^4*log(z)
    return 2.0 - 0.5 * z2 + c1 * z4 - 0.125 * c0


@jax.jit
def f3(z):
    """z^3 K_3[z] z -> 0  O(z^4)"""
    z2 = z * z
    z4 = z2 * z2
    return 8.0 - z2 + 0.125 * z4


@jax.jit
def f4(z):
    """z^4 K_4[z] z -> 0 O(z^4)"""
    z2 = z * z
    z4 = z2 * z2
    return 48.0 - 4 * z2 + 0.25 * z4


@jax.jit
def f5(z):
    """z^5 K_5[z] z -> 0 O(z^4)"""
    z2 = z * z
    z4 = z2 * z2
    return 384.0 - 24.0 * z2 + z4


@jax.jit
def fsmallz_nu(z, nu):
    def fnu(z, nu):
        """z^nu K_nu[z] z -> 0 O(z^4) z > 0"""
        z2 = z * z
        z4 = z2 * z2
        c1 = jnp.power(2.0, -6.0 - nu)
        c2 = _gamma(-2.0 - nu)
        c3 = _gamma(-2.0 + nu)
        c4 = jnp.power(z, 2.0 * nu)
        c5 = z4 * 8.0 * z2 * (2.0 + nu) + 32.0 * (1.0 + nu) * (2.0 + nu)
        c6 = z2 * (16.0 + z2 - 8.0 * nu)
        return c1 * (c4 * c5 * c2 + jnp.power(4.0, nu) * (c6 + 32.0 * _gamma(nu)))

    return jnp.select(
        [nu == 0, nu == 1, nu == 2, nu == 3, nu == 4],
        [f0(z), f1(z), f2(z), f3(z), f4(z)],
        default=fnu(z, nu),
    )


@jax.jit
def fz_nu(z, nu):
    """z^nu K_nu[z], z > 0"""
    return jnp.where(z <= 1.0e-10, fsmallz_nu(z, nu), jnp.power(z, nu) * _Knu(nu, z))


@jax.jit
def fsmallz_nup1(z, nu):
    def fnu(z, nu):
        """z^(nu+1) K_(nu+1)[z] z -> 0"""
        z2 = z * z
        z4 = z2 * z2
        c1 = -jnp.power(2.0, -4.0 - nu)
        c2 = _gamma(-2.0 - nu)
        c3 = c1 * c2 * (8.0 + 4.0 * nu + z2) * jnp.power(z, 2.0 * (1.0 + nu))
        c4 = jnp.power(2.0, -5.0 + nu)
        c5 = _gammap1(nu) / (nu * (-1.0 + nu))
        c6 = c4 * c5 * (32.0 * nu * (-1.0 + nu) - 8.0 * z2 * (-1.0 + nu) + z4)
        return c3 + c6

    return jnp.select(
        [nu == 0, nu == 1, nu == 2, nu == 3, nu == 4],
        [f1(z), f2(z), f3(z), f4(z), f5(z)],
        default=fnu(z, nu),
    )


@jax.jit
def fz_nup1(z, nu):
    """Return z^(nu+1) K_{nu+1}(z)
    Spergel index nu in [-0.85, 4.]
    """
    return jnp.where(
        z <= 1.0e-10, fsmallz_nup1(z, nu), jnp.power(z, nu + 1.0) * _Knu(nu + 1.0, z)
    )


@jax.jit
def fluxfractionFunc(z, nu, alpha):
    """Return  z^(nu+1) K_{nu+1}(z) / (2^nu Gamma(nu+1)) - (1-alpha)"""
    return fz_nup1(z, nu) / (jnp.power(2.0, nu) * _gammap1(nu)) - (1.0 - alpha)


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
    return bisect_for_root(
        partial(fluxfractionFunc, nu=nu, alpha=alpha), 0.0, 5.0, niter=75
    )


@_wraps(
    _galsim.Spergel,
    lax_description="""
    The JAX version uses the following profile
        .. math::
        
            I(r) = flux \times \left(2\pi 2^\nu \Gamma(1+\nu) r_0^2\right)^{-1} 
              \times \left(\frac{r}{r_0}\right)^\nu K_\nu\left(\frac{r}{r_0}\right)

    with the following Fourier expression
        .. math::

            \hat{I}(k) = flux / (1 + (k r_0)^2)^{1+\nu}

    where :math:`r_0` is the ``scale_radius``, and :math: `\nu` mandatory to be in [-0.85,4.0] 
    """,
)
@register_pytree_node_class
class Spergel(GSObject):
    _has_hard_edges = False
    _is_axisymmetric = True
    _is_analytic_x = True
    _is_analytic_k = True

    _minimum_nu = -0.85
    _maximum_nu = 4.0

    def __init__(
        self, nu, half_light_radius=None, scale_radius=None, flux=1.0, gsparams=None
    ):
        # Todo: how to implement this check
        # if self._nu < Spergel._minimum_nu:
        #    raise _galsim.GalSimRangeError("Requested Spergel index is too small",
        #                           self._nu, Spergel._minimum_nu, Spergel._maximum_nu)
        # if self._nu > Spergel._maximum_nu:
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
                    scale_radius=half_light_radius / calculateFluxRadius(0.5, nu),
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
        """The Spergel index, nu"""
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
        return self._r0 * calculateFluxRadius(0.5, self.nu)

    @property
    def _shootxnorm(self):
        """Normalization for photon shooting"""
        return 1.0 / (2.0 * jnp.pi * jnp.power(2.0, self.nu) + _gammap1(self.nu))

    @property
    def _xnorm(self):
        """Normalization of xValue"""
        return self._shootxnorm * self.flux * self._inv_r0_sq

    @property
    def _xnorm0(self):
        """return z^nu K_nu(z) for z=0"""
        return jax.lax.select(
            self.nu > 0, _gamma(self.nu) * jnp.power(2.0, self.nu - 1.0), jnp.inf
        )

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
        return "galsim.Spergel(nu=%r, scale_radius=%r, flux=%r, gsparams=%r)" % (
            ensure_hashable(self.beta),
            ensure_hashable(self.scale_radius),
            ensure_hashable(self.flux),
            self.gsparams,
        )

    def __str__(self):
        s = "galsim.Spergel(nu=%s, half_light_radius=%s" % (
            ensure_hashable(self.nu),
            ensure_hashable(self.half_light_radius),
        )
        if self.flux != 1.0:
            s += ", flux=%s" % ensure_hashable(self.flux)
        s += ")"
        return s

    @property
    def _maxk(self):
        """(1+ (k r0)^2)^(-1-nu) = maxk_threshold"""
        return (
            jnp.sqrt(
                jnp.power(self.gsparams.maxk_threshold, -1.0 / (1.0 + self.nu)) - 1.0
            )
            / self._r0
        )

    @property
    def _stepk(self):
        R = (
            calculateFluxRadius(1.0 - self.gsparams.folding_threshold, self.nu)
            * self._r0
        )
        # Go to at least 5*hlr
        R = jnp.maximum(R, self.gsparams.stepk_minimum_hlr * self.half_light_radius)
        return jnp.pi / R

    @property
    def _max_sb(self):
        # from SBSpergelImpl.h
        return jnp.abs(self._xnorm) * self._xnorm0

    @jax.jit
    def _xValue(self, pos):
        r = jnp.sqrt(pos.x**2 + pos.y**2) * self._inv_r0
        res = jnp.where(r == 0, self._xnorm0(self.nu), fz_nu(r, self.nu))
        return self._xnorm * res

    @jax.jit
    def _kValue(self, kpos):
        ksq = (kpos.x**2 + kpos.y**2) * self._r0_sq
        return self.flux * jnp.power(1.0 + ksq, 1.0 + self.nu)

    def _drawReal(self, image, jac=None, offset=(0.0, 0.0), flux_scaling=1.0):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_xValue(self, image, _jac, jnp.asarray(offset), flux_scaling)

    def _drawKImage(self, image, jac=None):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_kValue(self, image, _jac)

    @_wraps(_galsim.Spergel.withFlux)
    def withFlux(self, flux):
        return Spergel(
            nu=self.nu,
            scale_radius=self.scale_radius,
            flux=flux,
            gsparams=self.gsparams,
        )

    @_wraps(_galsim.Spergel._shoot)
    def _shoot(self, photons, rng):
        raise NotImplementedError("Shooting photons is not yet implemented")
