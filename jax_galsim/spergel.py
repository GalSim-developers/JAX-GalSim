import galsim as _galsim
import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial
from jax.tree_util import register_pytree_node_class

from jax_galsim.bessel import kv
from jax_galsim.core.draw import draw_by_kValue, draw_by_xValue
from jax_galsim.core.utils import cast_to_float, ensure_hashable, implements
from jax_galsim.gsobject import GSObject
from jax_galsim.random import UniformDeviate


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
def fz_nu(nu, z):
    """z^nu K_nu[z] with z > 0"""
    return jnp.power(z, nu) * kv(nu, z)


@jax.jit
def fluxfractionFunc(z, nu, alpha):
    """1 - z^(nu+1) K_{nu+1}(z) / (2^nu Gamma(nu+1)) - alpha"""
    return 1.0 - fz_nu(nu + 1.0, z) / (jnp.power(2.0, nu) * _gammap1(nu)) - alpha


@jax.jit
def reducedfluxfractionFunc(z, nu, norm):
    """(1 - z^(nu+1) K_{nu+1}(z) / (2^nu Gamma(nu+1)))/norm"""
    return fluxfractionFunc(z, nu, alpha=0.0) / norm


# code here is from JAX osurce for testing custom_root
# used under license:
# Copyright 2022 The JAX Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def _binary_search(func, x0, low=0.0, high=40.0):
    del x0  # unused

    def cond(state):
        low, high = state
        midpoint = 0.5 * (low + high)
        return (low < midpoint) & (midpoint < high)

    def body(state):
        low, high = state
        midpoint = 0.5 * (low + high)
        update_upper = func(midpoint) > 0
        low = jnp.where(update_upper, low, midpoint)
        high = jnp.where(update_upper, midpoint, high)
        return (low, high)

    solution, _ = jax.lax.while_loop(cond, body, (low, high))
    return solution


# end of code from jax source


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
    return jax.lax.custom_root(
        partial(fluxfractionFunc, nu=nu, alpha=alpha),
        20.0,
        _binary_search,
        lambda f, y: y / f(1.0),
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


@jax.jit
def _spergel_hlr_binary_search_plus_pade_init(nu):
    """Return radius R enclosing flux fraction 0.5 in unit of the scale radius r0"""
    z = _spergel_hlr_pade(nu)

    def _hlr_bs(f, x0):
        return _binary_search(f, x0, low=x0 - 1, high=x0 + 1)

    return jax.lax.custom_root(
        partial(fluxfractionFunc, nu=nu, alpha=0.5),
        z,
        _hlr_bs,
        lambda f, y: y / f(1.0),
    )


LAX_SPERGEL_DESCRIPTION = r"""
The fully normalized Spergel profile (used in both standard GalSim and JAX-GalSim) is

.. math::
    I(r) = flux \times \left(2\pi 2^\nu \Gamma(1+\nu) r_0^2\right)^{-1} \times \left(\frac{r}{r_0}\right)^\nu K_\nu\left(\frac{r}{r_0}\right)

with the following Fourier expression

.. math::
    \hat{I}(k) = flux / (1 + (k r_0)^2)^{1+\nu}

where :math:`r_0` is the ``scale_radius``, and :math:`\nu` mandatory to be in [-0.85,4.0]

The JAX-GalSim implementation does not support autodiff with respect to :math:`\nu` for
real-space evaluations.
"""


@implements(_galsim.Spergel, lax_description=LAX_SPERGEL_DESCRIPTION)
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
        nu = cast_to_float(nu)

        # Parse the radius options
        if half_light_radius is not None:
            if scale_radius is not None:
                raise _galsim.GalSimIncompatibleValuesError(
                    "Only one of scale_radius, half_light_radius may be specified",
                    half_light_radius=half_light_radius,
                    scale_radius=scale_radius,
                )
            else:
                # for python floats, we can use galsim on the CPU-side to do this
                # quickly as long as we ensure it is done at compile time.
                if isinstance(nu, float):
                    with jax.ensure_compile_time_eval():
                        hlr = _galsim.Spergel(nu, scale_radius=1).half_light_radius
                else:
                    hlr = _spergel_hlr_binary_search_plus_pade_init(nu)

                super().__init__(
                    nu=nu,
                    scale_radius=half_light_radius / hlr,
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
    @implements(_galsim.spergel.Spergel.half_light_radius)
    def half_light_radius(self):
        return self._r0 * _spergel_hlr_pade(self.nu)

    @property
    def _shootxnorm(self):
        """Normalization for photon shooting"""
        return 1.0 / (2.0 * jnp.pi * jnp.power(2.0, self.nu) * _gammap1(self.nu))

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
            s += ", flux=%s" % (ensure_hashable(self.flux),)
        s += ")"
        return s

    @property
    def _maxk(self):
        """(1+ (k r0)^2)^(-1-nu) = maxk_threshold"""
        res = jnp.power(self.gsparams.maxk_threshold, -1.0 / (1.0 + self.nu)) - 1.0
        return jnp.sqrt(res) / self._r0

    @property
    def _stepk(self):
        R = calculateFluxRadius(1.0 - self.gsparams.folding_threshold, self.nu)
        R *= self._r0
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
        res = jnp.where(r == 0, self._xnorm0, fz_nu(jax.lax.stop_gradient(self.nu), r))
        res = self._xnorm * res
        return res

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

    @property
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

    @property
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
        knur = fz_nu(self.nu, shoot_rmin)

        corrFact = self._shootxnorm  # this is the correct normalisation
        b = knur - flux_target / (jnp.pi * shoot_rmin * shoot_rmin * corrFact)
        b = 3.0 * b / shoot_rmin
        a = knur - shoot_rmin * b

        def cumulflux(z, a, b, zmin, nu, norm=1.0):
            flux_min = a / 3.0 * zmin * zmin * zmin + b / 2.0 * zmin * zmin
            c1 = fz_nu(nu + 1.0, zmin)
            res = jnp.where(
                z <= zmin,
                a / 3.0 * z * z * z + b / 2.0 * z * z,
                flux_min + c1 - fz_nu(nu + 1.0, z),
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
