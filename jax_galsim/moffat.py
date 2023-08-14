from functools import partial

import jax
import jax.numpy as jnp

import jax.scipy as jsc


from jax_galsim.gsobject import GSObject
from jax_galsim.gsparams import GSParams
from jax_galsim.core.draw import draw_by_xValue, draw_by_kValue

from jax_galsim.core.bessel import j0
from jax_galsim.core.integrate import ClenshawCurtisQuad, quad_integral

import galsim as _galsim
from jax._src.numpy.util import _wraps
from jax.tree_util import register_pytree_node_class

import tensorflow_probability as tfp


def _Knu(nu, x):
    """Modified Bessel 2nd kind for Untruncated Moffat"""
    return tfp.substrates.jax.math.bessel_kve(nu, x) / jnp.exp(jnp.abs(x))


#
def MoffatIntegrant(x, k, beta):
    """For truncated Hankel used in truncated Moffat"""
    return x * jnp.power(1 + x**2, -beta) * j0(k * x)


def _xMoffatIntegrant(k, beta, rmax, quad):
    return quad_integral(partial(MoffatIntegrant, k=k, beta=beta), 0.0, rmax, quad)


def MoffatCalculateSRFromHLR(re, rm, beta, Nloop=1000):
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
    # JEC Where to do these checking?
    # assert rm != 0.0, f"MoffatCalculateSRFromHLR: rm=={rm} should be done elsewhere"
    #
    # assert (
    #    rm > jnp.sqrt(2.0) * re
    # ), f"MoffatCalculateSRFromHLR: Cannot find a scaled radius: rm={rm}, sqrt(2)*re={jnp.sqrt(2.) * re}"

    ## fix loop iteration is faster and reach eps=1e-6 (single precision)
    def body(i, xcur):
        xnew = re / jnp.sqrt(
            jnp.power(
                (1 + jnp.power(1 + (rm / xcur) ** 2, 1 - beta)) / 2, 1 / (1 - beta)
            )
            - 1
        )
        return xnew

    rd = jax.lax.fori_loop(0, Nloop, body, re)

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

        gsparams = GSParams.check(gsparams)

        # let define beta_thr a threshold to trigger the truncature
        self._beta_thr = 1.1

        # JEC Where to do these checking?
        # if trunc == 0.0 and beta <= _beta_thr:
        #    raise _galsim.GalSimRangeError(
        #        f"Moffat profiles with beta <= {_beta_thr} must be truncated",
        #        beta,
        #        _beta_thr,
        #    )
        # if trunc < 0.0:
        #    raise _galsim.GalSimRangeError("Moffat trunc must be >= 0", trunc, 0.0)

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
                    half_light_radius=half_light_radius,
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
                    beta=beta, fwhm=fwhm, trunc=trunc, flux=flux, gsparams=gsparams
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
        return self.params["beta"]

    @property
    def trunc(self):
        """The truncation radius (if any) of this `Moffat` profile."""
        return self.params["trunc"]

    @property
    def scale_radius(self):
        """The scale radius of this `Moffat` profile."""
        if "half_light_radius" in self.params:
            hlr = self.params["half_light_radius"]
            return jax.lax.select(
                self.trunc > 0,
                MoffatCalculateSRFromHLR(hlr, self.trunc, self.beta),
                hlr / jnp.sqrt(jnp.power(0.5, 1.0 / (1.0 - self.beta)) - 1.0),
            )
        elif "fwhm" in self.params:
            return self.params["fwhm"] / (
                2.0 * jnp.sqrt(2.0 ** (1.0 / self.beta) - 1.0)
            )
        else:
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
        if "half_light_radius" in self.params:
            return self.params["half_light_radius"]
        else:
            return self._r0 * jnp.sqrt(
                jnp.power(1.0 - 0.5 * self._fluxFactor, 1.0 / (1.0 - self.beta)) - 1.0
            )

    @property
    def fwhm(self):
        """The FWHM of this `Moffat` profle."""
        if "fwhm" in self.params:
            return self.params["fwhm"]
        else:
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
        return (
            self.flux
            * 4.0
            / (jnp.power(2.0, self.beta) * jnp.exp(jax.lax.lgamma(self.beta - 1.0)))
        )

    def __hash__(self):
        return hash(
            (
                "galsim.Moffat",
                self.beta,
                self.scale_radius,
                self.trunc,
                self.flux,
                self.gsparams,
            )
        )

    def __repr__(self):
        return (
            "galsim.Moffat(beta=%r, scale_radius=%r, trunc=%r, flux=%r, gsparams=%r)"
            % (self.beta, self.scale_radius, self.trunc, self.flux, self.gsparams)
        )

    def __str__(self):
        s = "galsim.Moffat(beta=%s, scale_radius=%s" % (self.beta, self.scale_radius)
        if self.trunc != 0.0:
            s += ", trunc=%s" % self.trunc
        if self.flux != 1.0:
            s += ", flux=%s" % self.flux
        s += ")"
        return s

    @property
    def _maxk_untrunc(self):
        """untruncated Moffat maxK"""
        ## JEC 8/7/23: new code w/o while_loop. Notice that some test using GalSim code itself leads to NaN for beta>=5.5
        ## (this should be investigated with GalSim Tests)

        ## JEC 21/1/23: see issue #1208 in GalSim github as it seems there is an error
        ## in the expression used.

        ##The 2D Fourier Transform of f(r)=C (1+(r/rd)^2)^(-beta) leads
        ## C rd^2 = Flux (beta-1)/pi (no truc)
        ## and
        ## f(k) = C rd^2 int_0^infty (1+x^2)^(-beta) J_0(krd x) x dx
        ##      = 2 F (k rd /2)^(\beta-1) K[beta-1, k rd]/Gamma[beta-1]
        ## with k->\infty asymptotic behavior
        ## f(k)/F \approx sqrt(pi)/Gamma(beta-1) e^(-k') (k'/2)^(beta -3/2) with k' = k rd
        ## So we solve f(maxk)/F = thr  (aka maxk_threshold  in  gsparams.py)
        ## leading to the iterative search of
        ## let alpha = -log(thr Gamma(beta-1)/sqrt(pi))
        ## k = (\beta -3/2)\log(k/2) + alpha
        ## starting with k = alpha
        ##
        def body(i, val):
            # decode val
            kcur, alpha = val
            knew = (self.beta - 0.5) * jnp.log(kcur) + alpha  ## GalSim code
            ## knew = (self.beta -1.5)* jnp.log(kcur/2) + alpha # My code
            return knew, alpha

        ## alpha = -jnp.log(self.gsparams.maxk_threshold * jnp.exp(jsc.special.gammaln(self._beta-1))/jnp.sqrt(jnp.pi) ) # My code

        alpha = -jnp.log(
            self.gsparams.maxk_threshold
            * jnp.power(2.0, self.beta - 0.5)
            * jnp.exp(jsc.special.gammaln(self.beta - 1))
            / (2 * jnp.sqrt(jnp.pi))
        )  ## Galsim code

        val_init = (
            alpha,
            alpha,
        )
        val = jax.lax.fori_loop(0, 5, body, val_init)
        maxk, alpha = val
        return maxk / self._r0

    def _hankel1(self, k):
        return partial(
            _xMoffatIntegrant,
            beta=self.beta,
            rmax=self._maxRrD,
            quad=ClenshawCurtisQuad.init(150),
        )

    @property
    def _prefactor(self):
        return 2.0 * (self.beta - 1.0) / (self._fluxFactor)

    def _hankel(self, k):
        return self._hankel1(k) * self._prefactor

    def _v_hankel(self, k):
        return jax.jit(jax.vmap(self._hankel))

    @property
    def _maxk_trunc(self):
        """truncated Moffat maxK"""
        maxk_val = (
            self.gsparams.maxk_threshold
        )  # a for gaussian profile... this is f(k_max)/Flux = maxk_threshold
        dk = self.gsparams.table_spacing * jnp.sqrt(
            jnp.sqrt(self.gsparams.kvalue_accuracy / 10.0)
        )
        ki = jnp.arange(
            0.0, 50.0, dk
        )  # 50 is a max (GalSim) but it may be lowered if necessary
        fki = self._v_hankel(ki)
        maxk = ki[jnp.abs(fki) > maxk_val][-1]
        return maxk / self._r0

    @property
    def _maxk(self):
        return jax.lax.select(self.trunc > 0, self._maxk_trunc, self._maxk_untrunc)

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
        if R > self._maxR:
            R = self._maxR
        # at least R should be 5 HLR
        R5hlr = self.gsparams.stepk_minimum_hlr * self.half_light_radius
        if R < R5hlr:
            R = R5hlr
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

    def _xValue(self, pos):
        rsq = (pos.x**2 + pos.y**2) * self._inv_r0_sq
        # trunc if r>maxR with r0 scaled version
        return jax.lax.select(
            rsq > self._maxRrD_sq, 0.0, self._norm * jnp.power(1.0 + rsq, -self.beta)
        )

    def _kValue_untrunc(self, kpos):
        """Non truncated version of _kValue"""
        k = jnp.sqrt((kpos.x**2 + kpos.y**2) * self._r0_sq)

        return jax.lax.select(
            k == 0,
            self._knorm,
            self._knorm_bis * jnp.power(k, self.beta - 1.0) * _Knu(self.beta - 1, k),
        )

    def _kvalue_trunc(self, kpos):
        """truncated version of _kValue"""
        ksq = (kpos.x**2 + kpos.y**2) * self._r0_sq
        k = jnp.sqrt(ksq)
        return jax.lax.select(k > 50.0, 0.0, self._knorm * self._hankel(k))

    def _kValue(self, kpos):
        return jax.lax.select(
            self.trunc > 0, self._kvalue_trunc(kpos), self._kValue_untrunc(kpos)
        )

    def _drawReal(self, image, jac=None, offset=(0.0, 0.0), flux_scaling=1.0):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_xValue(self, image, _jac, jnp.asarray(offset), flux_scaling)

    def _drawKImage(self, image, jac=None):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_kValue(self, image, _jac)

    def withFlux(self, flux):
        return Moffat(
            beta=self.beta,
            scale_radius=self.scale_radius,
            trunc=self.trunc,
            flux=flux,
            gsparams=self.gsparams,
        )

    # @classmethod
    # def tree_unflatten(cls, aux_data, children):
    #    """Recreates an instance of the class from flatten representation"""
    #    obj = object.__new__(cls)
    #    return obj
