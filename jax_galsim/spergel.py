import galsim as _galsim
import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial
from jax.tree_util import register_pytree_node_class

from jax_galsim.bessel import kv
from jax_galsim.core.draw import draw_by_kValue, draw_by_xValue
from jax_galsim.core.utils import bisect_for_root, ensure_hashable, implements
from jax_galsim.gsobject import GSObject
from jax_galsim.random import UniformDeviate
from jax_galsim.utilities import lazy_property


@jax.jit
def gamma(x):
    """Gamma(x)"""
    x = x * 1.0
    return jnp.exp(jax.lax.lgamma(x))


@jax.jit
def _gamma(nu):
    """Gamma(nu) with care for integer nu in [0,5]"""
    return jnp.select(
        [nu == 0, nu == 1, nu == 2, nu == 3, nu == 4, nu == 5],
        [jnp.inf, 1.0, 1.0, 2.0, 6.0, 24.0],
        default=gamma(nu),
    )


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
    """K_0[z] with z -> 0  O(z^4)"""
    z2 = z * z
    z4 = z2 * z2
    c0 = 0.11593151565841244881
    c1 = 0.27898287891460311220
    c2 = 0.025248929932162694513
    return c0 + c1 * z2 + c2 * z4 - jnp.power(1.0 + 0.125 * z2, 2.0) * jnp.log(z)


@jax.jit
def f1(z):
    """z^1 K_1[z] with z -> 0  O(z^4)"""
    z2 = z * z
    z4 = z2 * z2
    c0 = z2lz(z)  # z^2 log(z)
    c1 = 0.30796575782920622441
    c2 = 0.08537071972865077805
    return 1.0 - c1 * z2 - c2 * z4 + c0 * (0.5 + 0.0625 * z2)


@jax.jit
def f2(z):
    """z^2 K_2[z] with z -> 0  O(z^4)"""
    c1 = 0.10824143945730155610
    z2 = z * z
    z4 = z2 * z2
    c0 = z2lz(z) * z2  # z^4*log(z)
    return 2.0 - 0.5 * z2 + c1 * z4 - 0.125 * c0


@jax.jit
def f3(z):
    """z^3 K_3[z] with z -> 0  O(z^4)"""
    z2 = z * z
    z4 = z2 * z2
    return 8.0 - z2 + 0.125 * z4


@jax.jit
def f4(z):
    """z^4 K_4[z] with z -> 0 O(z^4)"""
    z2 = z * z
    z4 = z2 * z2
    return 48.0 - 4 * z2 + 0.25 * z4


@jax.jit
def f5(z):
    """z^5 K_5[z] with z -> 0 O(z^4)"""
    z2 = z * z
    z4 = z2 * z2
    return 384.0 - 24.0 * z2 + z4


@jax.jit
def fsmallz_nu(z, nu):
    def fnu(z, nu):
        """z^nu K_nu[z] with z -> 0 O(z^4) z > 0"""
        nu += 1.0e-10  # to garanty that nu is not an integer
        z2 = z * z
        z4 = z2 * z2
        c1 = jnp.power(2.0, -6.0 - nu)
        c2 = _gamma(-2.0 - nu)
        c3 = _gamma(-2.0 + nu)
        c4 = jnp.power(z, 2.0 * nu)
        c5 = z4 * 8.0 * z2 * (2.0 + nu) + 32.0 * (1.0 + nu) * (2.0 + nu)
        c6 = z2 * (16.0 + z2 - 8.0 * nu) * c3
        return c1 * (c4 * c5 * c2 + jnp.power(4.0, nu) * (c6 + 32.0 * _gamma(nu)))

    return jnp.select(
        [nu == 0, nu == 1, nu == 2, nu == 3, nu == 4],
        [f0(z), f1(z), f2(z), f3(z), f4(z)],
        default=fnu(z, nu),
    )


@jax.jit
def fz_nu(z, nu):
    """z^nu K_nu[z] with z > 0"""
    return jnp.where(z <= 1.0e-10, fsmallz_nu(z, nu), jnp.power(z, nu) * kv(nu, z))


@jax.jit
def fsmallz_nup1(z, nu):
    def fnu(z, nu):
        """z^(nu+1) K_(nu+1)[z] with  z -> 0"""
        z2 = z * z
        z4 = z2 * z2
        c1 = -jnp.power(2.0, -4.0 - nu)
        c2 = _gamma(-2.0 - nu)
        c3 = c1 * c2 * (8.0 + 4.0 * nu + z2) * jnp.power(z, 2.0 * (1.0 + nu))
        c4 = jnp.power(2.0, nu)
        c5 = _gammap1(nu)
        c6 = c4 * c5 * (1.0 - 0.25 * z2 / nu + z4 * 0.03125 / (nu * (nu - 1.0)))
        return c3 + c6

    return jnp.select(
        [nu == 0, nu == 1, nu == 2, nu == 3, nu == 4],
        [f1(z), f2(z), f3(z), f4(z), f5(z)],
        default=fnu(z, nu),
    )


@jax.jit
def fz_nup1(z, nu):
    """z^(nu+1) K_{nu+1}(z)"""
    return jnp.where(
        z <= 1.0e-10, fsmallz_nup1(z, nu), jnp.power(z, nu + 1.0) * kv(nu + 1.0, z)
    )


@jax.jit
def fluxfractionFunc(z, nu, alpha):
    """1 - z^(nu+1) K_{nu+1}(z) / (2^nu Gamma(nu+1)) - alpha"""
    return 1.0 - fz_nup1(z, nu) / (jnp.power(2.0, nu) * _gammap1(nu)) - alpha


@jax.jit
def reducedfluxfractionFunc(z, nu, norm):
    """(1 - z^(nu+1) K_{nu+1}(z) / (2^nu Gamma(nu+1)))/norm"""
    return fluxfractionFunc(z, nu, alpha=0.0) / norm


@jax.jit
def calculateFluxRadius(alpha, nu, zmin=0.0, zmax=40.0):
    """Return radius R enclosing flux fraction alpha in unit of the scale radius r0

    Method: Solve  F(R/r0=z)/Flux - alpha = 0 using bisection algorithm

    F(R)/F =  int( 1/(2^nu Gamma(nu+1)) (r/r0)^(nu+1) K_nu(r/r0) dr/r0; r=0..R) = alpha
    =>
    z=R/r0 such that
    1 - z^(nu+1) K_{nu+1}(z) / (2^nu Gamma(nu+1)) = alpha

    Typical use cases:
     o alpha = 1/2 => R = Half-Light-Radius,
     o alpha = 1 - folding-thresold => R used for stepk computation

     nu: the Spergel index

     nb. it is supposed that nu is in [-0.85, 4.0] checked in the Spergel class init
    """
    return bisect_for_root(
        partial(fluxfractionFunc, nu=nu, alpha=alpha), zmin, zmax, niter=75,
    )


def _spergel_hlr_pade(x):
    """A Pseudo-Pade approximation for the HLR of the Spergel profile as a function of nu.

    See dev/notebooks/spergel_hlr_flux_radius_approx.ipynb for code to generate this routine.
    """
    # fmt: off
    pm = 1.2571513771129166 + x * (
        3.7059053890269102 + x * (
            2.8577090425861944 + x * (
                -0.30570486567039273 + x * (
                    0.6589831675940833 + x * (
                        3.375577680133867 + x * (
                            2.8143565844741403 + x * (
                                0.9292378858457211 + x * (
                                    0.12096941981286179 + x * (
                                        0.004206502758293099
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    qm = 1.0 + x * (
        2.1939178810491837 + x * (
            0.8281034080784796 + x * (
                -0.5163329765186994 + x * (
                    0.9164871490929886 + x * (
                        1.8988551389326231 + x * (
                            1.042688817291684 + x * (
                                0.22580140592548198 + x * (
                                    0.01681923980317362 + x * (
                                        0.00018168506955933716
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    # fmt: on
    return pm / qm


@implements(
    _galsim.Spergel,
    lax_description=r"""The fully normalized Spergel profile (used in both standard GalSim and JAX-GalSim) is
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
        self,
        nu,
        scale_radius=None,
        half_light_radius=None,
        flux=1.0,
        gsparams=None,
    ):
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
                    scale_radius=half_light_radius / _spergel_hlr_pade(nu),
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
    @implements(_galsim.spergel.Spergel.nu)
    def nu(self):
        return self._params["nu"]

    @property
    @implements(_galsim.spergel.Spergel.scale_radius)
    def scale_radius(self):
        return self.params["scale_radius"]

    @property
    def _r0(self):
        return self.scale_radius

    @lazy_property
    def _inv_r0(self):
        return 1.0 / self._r0

    @lazy_property
    def _r0_sq(self):
        return self._r0 * self._r0

    @lazy_property
    def _inv_r0_sq(self):
        return self._inv_r0 * self._inv_r0

    @lazy_property
    @implements(_galsim.spergel.Spergel.half_light_radius)
    def half_light_radius(self):
        return self._r0 * _spergel_hlr_pade(self.nu)

    @lazy_property
    def _shootxnorm(self):
        """Normalization for photon shooting"""
        return 1.0 / (2.0 * jnp.pi * jnp.power(2.0, self.nu) * _gammap1(self.nu))

    @lazy_property
    def _xnorm(self):
        """Normalization of xValue"""
        return self._shootxnorm * self.flux * self._inv_r0_sq

    @lazy_property
    def _xnorm0(self):
        """return z^nu K_nu(z) for z=0"""
        return jax.lax.select(
            self.nu > 0, _gamma(self.nu) * jnp.power(2.0, self.nu - 1.0), jnp.inf
        )

    @implements(_galsim.spergel.Spergel.calculateFluxRadius)
    def calculateFluxRadius(self, f):
        return self._r0 * calculateFluxRadius(f, self.nu)

    @implements(_galsim.spergel.Spergel.calculateIntegratedFlux)
    def calculateIntegratedFlux(self, r):
        return fluxfractionFunc(r / self._r0, self.nu, 0.0)

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
            ensure_hashable(self.nu),
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

    @lazy_property
    def _maxk(self):
        """(1+ (k r0)^2)^(-1-nu) = maxk_threshold"""
        res = jnp.power(self.gsparams.maxk_threshold, -1.0 / (1.0 + self.nu)) - 1.0
        return jnp.sqrt(res) / self._r0

    @lazy_property
    def _stepk(self):
        R = calculateFluxRadius(1.0 - self.gsparams.folding_threshold, self.nu)
        R *= self._r0
        # Go to at least 5*hlr
        R = jnp.maximum(R, self.gsparams.stepk_minimum_hlr * self.half_light_radius)
        return jnp.pi / R

    @lazy_property
    def _max_sb(self):
        # from SBSpergelImpl.h
        return jnp.abs(self._xnorm) * self._xnorm0

    @jax.jit
    def _xValue(self, pos):
        r = jnp.sqrt(pos.x**2 + pos.y**2) * self._inv_r0
        res = jnp.where(r == 0, self._xnorm0, fz_nu(r, self.nu))
        return self._xnorm * res

    @jax.jit
    def _kValue(self, kpos):
        ksq = (kpos.x**2 + kpos.y**2) * self._r0_sq
        return self.flux * jnp.power(1.0 + ksq, -1.0 - self.nu)

    def _drawReal(self, image, jac=None, offset=(0.0, 0.0), flux_scaling=1.0):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_xValue(self, image, _jac, jnp.asarray(offset), flux_scaling)

    def _drawKImage(self, image, jac=None):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_kValue(self, image, _jac)

    @implements(_galsim.Spergel.withFlux)
    def withFlux(self, flux):
        return Spergel(
            nu=self.nu,
            scale_radius=self.scale_radius,
            flux=flux,
            gsparams=self.gsparams,
        )

    @lazy_property
    def _shoot_pos_cdf(self):
        zmax = calculateFluxRadius(
            1.0 - self.gsparams.shoot_accuracy, self.nu, zmax=30.0
        )
        flux_max = fluxfractionFunc(zmax, self.nu, alpha=0.0)
        preducedfluxfractionFunc = partial(
            reducedfluxfractionFunc, nu=self.nu, norm=flux_max
        )
        z_cdf = jnp.linspace(0, zmax, 10_000)
        cdf = preducedfluxfractionFunc(z_cdf)
        return z_cdf, cdf

    def _shoot_pos(self, u):
        # shoot r in case of nu>0
        z_cdf, cdf = self._shoot_pos_cdf
        z = jnp.interp(u, cdf, z_cdf)  # linear inversion of the CDF
        r = z * self._r0
        return r

    @lazy_property
    def _shoot_neg_cdf(self):
        # comment:
        # In the Galsim code the profile below rmin is linearized such that
        # call zmin = rmin/r0 such that
        # Int_0^zmin 2pi u x I(u) du = shoot_accuracy
        # Then let (a,b) such that
        # 1) Int_0^zmin 2pi u x (a + b u) du = shoot_accuracy
        # 2) a + b zmin = zmin^nu K_nu(zmin)
        # Now, noticing that
        # I(z) = z^nu  K_nu(z) / (2pi 2^nu Gamma(nu+1)) = z^nu  K_nu(z)/(2 pi Nnu)
        # there is a problem with eq. 1 as we would have expected
        # 1b) Int_0^zmin 2pi u x (a + b u)/(2 pi Nnu) du = shoot_accuracy
        # so the corrFact is there to signal the changement in this implementation

        zmax = calculateFluxRadius(
            1.0 - self.gsparams.shoot_accuracy, self.nu, zmax=30.0
        )
        flux_target = self.gsparams.shoot_accuracy
        shoot_rmin = calculateFluxRadius(flux_target, self.nu)
        knur = fz_nu(shoot_rmin, self.nu)

        corrFact = self._shootxnorm  # this is the correct normalisation
        b = knur - flux_target / (jnp.pi * shoot_rmin * shoot_rmin * corrFact)
        b = 3.0 * b / shoot_rmin
        a = knur - shoot_rmin * b

        def cumulflux(z, a, b, zmin, nu, norm=1.0):
            flux_min = a / 3.0 * zmin * zmin * zmin + b / 2.0 * zmin * zmin
            c1 = fz_nup1(zmin, nu)
            res = jnp.where(
                z <= zmin,
                a / 3.0 * z * z * z + b / 2.0 * z * z,
                flux_min + c1 - fz_nup1(z, nu),
            )
            return res / norm

        flux_max = cumulflux(zmax, a, b, shoot_rmin, self.nu)

        preducedfluxfractionFunc = partial(
            cumulflux, a=a, b=b, zmin=shoot_rmin, nu=self.nu, norm=flux_max
        )
        z_cdf = jnp.linspace(0, zmax, 10_000)
        cdf = preducedfluxfractionFunc(z_cdf)
        return z_cdf, cdf

    def _shoot_neg(self, u):
        # shoot r in case of  nu<=0
        z_cdf, cdf = self._shoot_neg_cdf
        z = jnp.interp(u, cdf, z_cdf)  # linear inversion of the CDF
        r = z * self._r0
        return r

    @implements(_galsim.Spergel._shoot)
    def _shoot(self, photons, rng):
        ud = UniformDeviate(rng)
        u = ud.generate(photons.x)
        r = jax.lax.select(self.nu > 0, self._shoot_pos(u), self._shoot_neg(u))
        ang = ud.generate(photons.x) * 2.0 * jnp.pi
        photons.x = r * jnp.cos(ang)
        photons.y = r * jnp.sin(ang)
        photons.flux = self.flux / photons.size()
