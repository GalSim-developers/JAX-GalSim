import jax.numpy as jnp

from jax_galsim.gsobject import GSObject
from jax_galsim.gsparams import GSParams

import galsim as _galsim
from jax._src.numpy.util import _wraps
from jax.tree_util import register_pytree_node_class


@_wraps(_galsim.Gaussian)
@register_pytree_node_class
class Gaussian(GSObject):
    # The FWHM of a Gaussian is 2 sqrt(2 ln2) sigma
    _fwhm_factor = 2.3548200450309493
    # The half-light-radius is sqrt(2 ln2) sigma
    _hlr_factor = 1.1774100225154747
    # 1/(2pi)
    _inv_twopi = 0.15915494309189535

    _has_hard_edges = False
    _is_axisymmetric = True
    _is_analytic_x = True
    _is_analytic_k = True

    def __init__(
        self, half_light_radius=None, sigma=None, fwhm=None, flux=1.0, gsparams=None
    ):
        # Checking gsparams
        gsparams = GSParams.check(gsparams)

        if fwhm is not None:
            if sigma is not None or half_light_radius is not None:
                raise _galsim.GalSimIncompatibleValuesError(
                    "Only one of sigma, fwhm, and half_light_radius may be specified",
                    fwhm=fwhm,
                    sigma=sigma,
                    half_light_radius=half_light_radius,
                )
            else:
                super().__init__(fwhm=fwhm, flux=flux, gsparams=gsparams)
        elif half_light_radius is not None:
            if sigma is not None:
                raise _galsim.GalSimIncompatibleValuesError(
                    "Only one of sigma, fwhm, and half_light_radius may be specified",
                    fwhm=fwhm,
                    sigma=sigma,
                    half_light_radius=half_light_radius,
                )
            else:
                super().__init__(
                    half_light_radius=half_light_radius, flux=flux, gsparams=gsparams
                )
        elif sigma is None:
            raise _galsim.GalSimIncompatibleValuesError(
                "One of sigma, fwhm, and half_light_radius must be specified",
                fwhm=fwhm,
                sigma=sigma,
                half_light_radius=half_light_radius,
            )
        else:
            super().__init__(sigma=sigma, flux=flux, gsparams=gsparams)

        self._sigsq = self.sigma**2
        self._inv_sigsq = 1.0 / self._sigsq
        self._norm = self.flux * self._inv_sigsq * Gaussian._inv_twopi

    @property
    def sigma(self):
        """The sigma of this Gaussian profile"""
        if "fwhm" in self.params:
            return self.params["fwhm"] / Gaussian._fwhm_factor
        elif "half_light_radius" in self.params:
            return self.params["half_light_radius"] / Gaussian._hlr_factor
        else:
            return self.params["sigma"]

    @property
    def half_light_radius(self):
        """The half-light radius of this Gaussian profile"""
        return self.sigma * Gaussian._hlr_factor

    @property
    def fwhm(self):
        """The FWHM of this Gaussian profile"""
        return self.sigma * Gaussian._fwhm_factor

    def __hash__(self):
        return hash(("galsim.Gaussian", self.sigma, self.flux, self.gsparams))

    def __repr__(self):
        return "galsim.Gaussian(sigma=%r, flux=%r, gsparams=%r)" % (
            self.sigma,
            self.flux,
            self.gsparams,
        )

    def __str__(self):
        s = "galsim.Gaussian(sigma=%s" % self.sigma
        if self.flux != 1.0:
            s += ", flux=%s" % self.flux
        s += ")"
        return s

    @property
    def _maxk(self):
        return jnp.sqrt(-2.0 * jnp.log(self.gsparams.maxk_threshold)) / self.sigma

    @property
    def _stepk(self):
        R = jnp.sqrt(-2.0 * jnp.log(self.gsparams.folding_threshold))
        # Bounding the value of R based on gsparams
        R = jnp.maximum(R, self.gsparams.stepk_minimum_hlr * Gaussian._hlr_factor)
        return jnp.pi / (R * self.sigma)

    @property
    def _max_sb(self):
        return self._norm

    def _xValue(self, pos):
        rsq = pos.x**2 + pos.y**2
        return self._norm * jnp.exp(-0.5 * rsq * self._inv_sigsq)

    def _kValue(self, kpos):
        ksq = (kpos.x**2 + kpos.y**2) * self._sigsq
        return self.flux * jnp.exp(-0.5 * ksq)

    def withFlux(self, flux):
        return Gaussian(sigma=self.sigma, flux=flux, gsparams=self.gsparams)
