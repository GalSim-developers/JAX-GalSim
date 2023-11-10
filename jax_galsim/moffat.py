import galsim as _galsim
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp
from jax._src.numpy.util import _wraps
from jax.tree_util import Partial as partial
from jax.tree_util import register_pytree_node_class

from jax_galsim.core.bessel import j0
from jax_galsim.core.draw import draw_by_kValue, draw_by_xValue
from jax_galsim.core.integrate import ClenshawCurtisQuad, quad_integral
from jax_galsim.core.utils import bisect_for_root, ensure_hashable
from jax_galsim.gsobject import GSObject
from jax_galsim.position import PositionD


@jax.jit
def _Knu(nu, x):
    """Modified Bessel 2nd kind for Untruncated Moffat"""
    return tfp.substrates.jax.math.bessel_kve(nu * 1.0, x) / jnp.exp(jnp.abs(x))


@jax.jit
def _MoffatIntegrant(x, k, beta):
    """For truncated Hankel used in truncated Moffat"""
    return x * jnp.power(1 + x**2, -beta) * j0(k * x)


def _xMoffatIntegrant(k, beta, rmax, quad):
    return quad_integral(partial(_MoffatIntegrant, k=k, beta=beta), 0.0, rmax, quad)


@jax.jit
def _hankel(k, beta, rmax):
    quad = ClenshawCurtisQuad.init(150)
    g = partial(_xMoffatIntegrant, beta=beta, rmax=rmax, quad=quad)
    return jax.vmap(g)(k)


@jax.jit
def _MoffatCalculateSRFromHLR(re, rm, beta):
    """
    The basic equation that is relevant here is the flux of a Moffat profile
    out to some radius.

    flux(R) = int( (1+r^2/rd^2 )^(-beta) 2pi r dr, r=0..R )
            = (pi rd^2 / (beta-1)) (1 - (1+R^2/rd^2)^(1-beta) )
    For now, we can ignore the first factor.  We call the second factor fluxfactor below,
    or in this function f(R).
    We are given two values of R for which we know that the ratio of their fluxes is 1/2:
    f(re) = 0.5 * f(rm)

    nb1. rd aka r0 aka the scale radius
    nb2. In GalSim definition rm = 0 (ex. no truncated Moffat) means in reality rm=+Inf.
         BUT the case rm==0 is already done, so HERE rm != 0
    """

    # fix loop iteration is faster and reach eps=1e-6 (single precision)
    def body(i, xcur):
        x = (1 + jnp.power(1 + (rm / xcur) ** 2, 1 - beta)) / 2
        x = jnp.power(x, 1 / (1 - beta))
        x = jnp.sqrt(x - 1)
        return re / x

    rd = jax.lax.fori_loop(0, 1000, body, re)

    return rd


@_wraps(_galsim.Moffat)
@register_pytree_node_class
class Moffat(GSObject):
    _is_axisymmetric = True
    _is_analytic_x = True
    _is_analytic_k = True

    def __init__(
        self,
        beta,
        scale_radius=None,
        half_light_radius=None,
        fwhm=None,
        trunc=0.0,
        flux=1.0,
        gsparams=None,
    ):
        # notice that trunc==0. means no truncated Moffat.
        # let define beta_thr a threshold to trigger the truncature
        self._beta_thr = 1.1

        # Parse the radius options
        if half_light_radius is not None:
            if scale_radius is not None or fwhm is not None:
                raise _galsim.GalSimIncompatibleValuesError(
                    "Only one of scale_radius, half_light_radius, or fwhm may be specified",
                    half_light_radius=half_light_radius,
                    scale_radius=scale_radius,
                    fwhm=fwhm,
                )
            else:
                super().__init__(
                    beta=beta,
                    scale_radius=(
                        jax.lax.select(
                            trunc > 0,
                            _MoffatCalculateSRFromHLR(half_light_radius, trunc, beta),
                            half_light_radius
                            / jnp.sqrt(jnp.power(0.5, 1.0 / (1.0 - beta)) - 1.0),
                        )
                    ),
                    trunc=trunc,
                    flux=flux,
                    gsparams=gsparams,
                )
        elif fwhm is not None:
            if scale_radius is not None:
                raise _galsim.GalSimIncompatibleValuesError(
                    "Only one of scale_radius, half_light_radius, or fwhm may be specified",
                    half_light_radius=half_light_radius,
                    scale_radius=scale_radius,
                    fwhm=fwhm,
                )
            else:
                super().__init__(
                    beta=beta,
                    scale_radius=fwhm / (2.0 * jnp.sqrt(2.0 ** (1.0 / beta) - 1.0)),
                    trunc=trunc,
                    flux=flux,
                    gsparams=gsparams,
                )
        elif scale_radius is None:
            raise _galsim.GalSimIncompatibleValuesError(
                "One of scale_radius, half_light_radius, or fwhm must be specified",
                half_light_radius=half_light_radius,
                scale_radius=scale_radius,
                fwhm=fwhm,
            )
        else:
            super().__init__(
                beta=beta,
                scale_radius=scale_radius,
                trunc=trunc,
                flux=flux,
                gsparams=gsparams,
            )

    @property
    def beta(self):
        """The beta parameter of this `Moffat` profile."""
        return self._params["beta"]

    @property
    def trunc(self):
        """The truncation radius (if any) of this `Moffat` profile."""
        return self._params["trunc"]

    @property
    def scale_radius(self):
        """The scale radius of this `Moffat` profile."""
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
    def _maxRrD(self):
        """maxR/rd ; fluxFactor Integral of total flux in terms of 'rD' units."""
        return jax.lax.select(
            self.trunc > 0.0,
            self.trunc * self._inv_r0,
            jnp.sqrt(
                jnp.power(self.gsparams.xvalue_accuracy, 1.0 / (1.0 - self.beta)) - 1.0
            ),
        )

    @property
    def _maxR(self):
        """maximum r"""
        return self._maxRrD * self._r0

    @property
    def _maxRrD_sq(self):
        return self._maxRrD * self._maxRrD

    @property
    def _fluxFactor(self):
        return jax.lax.select(
            self.trunc > 0.0,
            1.0 - jnp.power(1 + self._maxRrD * self._maxRrD, (1.0 - self.beta)),
            1.0,
        )

    @property
    def half_light_radius(self):
        """The half-light radius of this `Moffat` profile."""
        return self._r0 * jnp.sqrt(
            jnp.power(1.0 - 0.5 * self._fluxFactor, 1.0 / (1.0 - self.beta)) - 1.0
        )

    @property
    def fwhm(self):
        """The FWHM of this `Moffat` profle."""
        return self._r0 * (2.0 * jnp.sqrt(2.0 ** (1.0 / self.beta) - 1.0))

    @property
    def _norm(self):
        """Normalisation f(x) (trunc=0)"""
        return self.flux * (self.beta - 1) / (jnp.pi * self._fluxFactor * self._r0**2)

    @property
    def _knorm(self):
        """Normalisation f(k) (trunc = 0, k=0)"""
        return self.flux

    @property
    def _knorm_bis(self):
        """Normalisation f(k) (trunc = 0; k=/= 0)"""
        x1 = self.flux * 4
        x2 = jnp.power(2.0, self.beta)
        x3 = jnp.exp(jax.lax.lgamma(self.beta - 1.0))
        return x1 / (x2 * x3)

    def __hash__(self):
        return hash(
            (
                "galsim.Moffat",
                ensure_hashable(self.beta),
                ensure_hashable(self.scale_radius),
                ensure_hashable(self.trunc),
                ensure_hashable(self.flux),
                self.gsparams,
            )
        )

    def __repr__(self):
        return (
            "galsim.Moffat(beta=%r, scale_radius=%r, trunc=%r, flux=%r, gsparams=%r)"
            % (
                ensure_hashable(self.beta),
                ensure_hashable(self.scale_radius),
                ensure_hashable(self.trunc),
                ensure_hashable(self.flux),
                self.gsparams,
            )
        )

    def __str__(self):
        s = "galsim.Moffat(beta=%s, scale_radius=%s" % (
            ensure_hashable(self.beta),
            ensure_hashable(self.scale_radius),
        )
        if self.trunc != 0.0:
            s += ", trunc=%s" % ensure_hashable(self.trunc)
        if self.flux != 1.0:
            s += ", flux=%s" % ensure_hashable(self.flux)
        s += ")"
        return s

    @property
    def _prefactor(self):
        return 2.0 * (self.beta - 1.0) / (self._fluxFactor)

    @jax.jit
    def _maxk_func(self, k):
        return (
            jnp.abs(self._kValue(PositionD(x=k, y=0)).real / self.flux)
            - self.gsparams.maxk_threshold
        )

    @property
    @jax.jit
    def _maxk(self):
        return bisect_for_root(partial(self._maxk_func), 0.0, 1e5, niter=75)

    @property
    def _stepk_lowbeta(self):
        # implicit trunc>0 => _maxR= trunc
        #    then flux never converges (or nearly so),
        #   => so just use truncation radius
        return jnp.pi / self._maxR

    @property
    def _stepk_highbeta(self):
        # ignore the 1 in (1+R^2), so approximately
        R = (
            jnp.power(self.gsparams.folding_threshold, 0.5 / (1.0 - self.beta))
            * self._r0
        )
        R = jnp.minimum(R, self._maxR)
        # at least R should be 5 HLR
        R5hlr = self.gsparams.stepk_minimum_hlr * self.half_light_radius
        R = jnp.maximum(R, R5hlr)
        return jnp.pi / R

    @property
    def _stepk(self):
        """The fractional flux out to radius R is (if not truncated)
        1 - (1+(R/rd)^2)^(1-beta)
        So solve (1+(R/rd)^2)^(1-beta) = folding_threshold
        """
        return jax.lax.select(
            self.beta <= self._beta_thr, self._stepk_lowbeta, self._stepk_highbeta
        )

    @property
    def _has_hard_edges(self):
        return self.trunc != 0.0

    @property
    def _max_sb(self):
        return self._norm

    @jax.jit
    def _xValue(self, pos):
        rsq = (pos.x**2 + pos.y**2) * self._inv_r0_sq
        # trunc if r>maxR with r0 scaled version
        return jnp.where(
            rsq > self._maxRrD_sq, 0.0, self._norm * jnp.power(1.0 + rsq, -self.beta)
        )

    def _kValue_untrunc(self, k):
        """Non truncated version of _kValue"""
        return jnp.where(
            k > 0,
            self._knorm_bis * jnp.power(k, self.beta - 1.0) * _Knu(self.beta - 1.0, k),
            self._knorm,
        )

    def _kValue_trunc(self, k):
        """Truncated version of _kValue"""
        return jnp.where(
            k <= 50.0,
            self._knorm * self._prefactor * _hankel(k, self.beta, self._maxRrD),
            0.0,
        )

    @jax.jit
    def _kValue(self, kpos):
        """computation of the Moffat response in k-space with switch of truncated/untracated case
        kpos can be a scalar or a vector (typically, scalar for debug and 2D considering an image)
        """
        k = jnp.sqrt((kpos.x**2 + kpos.y**2) * self._r0_sq)
        out_shape = jnp.shape(k)
        k = jnp.atleast_1d(k)
        res = jax.lax.cond(
            self.trunc > 0,
            lambda x: self._kValue_trunc(x),
            lambda x: self._kValue_untrunc(x),
            k,
        )
        return res.reshape(out_shape)

    def _drawReal(self, image, jac=None, offset=(0.0, 0.0), flux_scaling=1.0):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_xValue(self, image, _jac, jnp.asarray(offset), flux_scaling)

    def _drawKImage(self, image, jac=None):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_kValue(self, image, _jac)

    @_wraps(_galsim.Moffat.withFlux)
    def withFlux(self, flux):
        return Moffat(
            beta=self.beta,
            scale_radius=self.scale_radius,
            trunc=self.trunc,
            flux=flux,
            gsparams=self.gsparams,
        )
