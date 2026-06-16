import os

import galsim as _galsim
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from jax_galsim.core.utils import ensure_hashable, implements


@implements(
    _galsim.SED,
    lax_description="""\
JAX-GalSim supports array-backed SEDs only.  The wavelength grid is static
and the flux array may be traced.  File input, string expressions, units,
LookupTable metadata, thinning, and magnitude/zeropoint helpers are not
implemented.
""",
)
@register_pytree_node_class
class SED:
    def __init__(self, wave, flux, redshift=0.0):
        if isinstance(wave, (str, bytes, os.PathLike)) or isinstance(
            flux, (str, bytes, os.PathLike)
        ):
            raise NotImplementedError(
                "JAX-GalSim SED supports array-backed SEDs only; file input, "
                "string expressions, and flux-type strings are not implemented."
            )

        self._wave = jnp.asarray(wave, dtype=float)  # static, not traced
        self._flux = jnp.asarray(flux)  # traced
        self._redshift = jnp.asarray(redshift, dtype=float)

        if self._wave.ndim != 1 or len(self._wave) < 2:
            raise ValueError("wave must be a 1-D array with at least 2 elements.")
        if self._flux.shape != self._wave.shape:
            raise ValueError("flux must have the same shape as wave.")

    @property
    @implements(
        None,
        lax_description="JAX-GalSim-specific static wavelength grid for array-backed SEDs.",
    )
    def wave(self):
        return self._wave

    @property
    @implements(
        None,
        lax_description="JAX-GalSim-specific traced flux array for array-backed SEDs.",
    )
    def flux(self):
        return self._flux

    @property
    @implements(
        None,
        lax_description="JAX-GalSim-specific traced redshift scalar for array-backed SEDs.",
    )
    def redshift(self):
        return self._redshift

    @property
    @implements(
        None,
        lax_description="JAX-GalSim-specific observed blue wavelength limit.",
    )
    def blue_limit(self):
        return float(self._wave[0]) * (1.0 + float(self._redshift))

    @property
    @implements(
        None,
        lax_description="JAX-GalSim-specific observed red wavelength limit.",
    )
    def red_limit(self):
        return float(self._wave[-1]) * (1.0 + float(self._redshift))

    @implements(_galsim.SED.__call__)
    def __call__(self, wave):
        wave = jnp.asarray(wave, dtype=float)
        # Convert observed wavelength to rest-frame before interpolating
        wave_rest = wave / (1.0 + self._redshift)
        return jnp.interp(wave_rest, self._wave, self._flux, left=0.0, right=0.0)

    @implements(_galsim.SED.calculateFlux)
    def calculateFlux(self, bandpass, n_waves=512):
        waves = jnp.linspace(bandpass.blue_limit, bandpass.red_limit, n_waves)
        return jnp.trapezoid(self(waves) * bandpass(waves), waves)

    @implements(_galsim.SED.atRedshift)
    def atRedshift(self, redshift):
        return SED(self._wave, self._flux, redshift)

    @implements(_galsim.SED.__mul__)
    def __mul__(self, other):
        if isinstance(other, SED):
            # Evaluate other SED on self's rest-frame grid and multiply
            other_flux = jnp.interp(
                self._wave * (1.0 + self._redshift),
                other._wave * (1.0 + other._redshift),
                other._flux,
                left=0.0,
                right=0.0,
            )
            return SED(self._wave, self._flux * other_flux, self._redshift)
        from jax_galsim.gsobject import GSObject

        if isinstance(other, GSObject):
            from jax_galsim.chromatic import SimpleChromaticTransformation

            return SimpleChromaticTransformation(other, self)
        from jax_galsim.chromatic import ChromaticObject

        if isinstance(other, ChromaticObject):
            return other * self
        return SED(self._wave, self._flux * other, self._redshift)

    @implements(getattr(_galsim.SED, "__rmul__", None))
    def __rmul__(self, other):
        return self.__mul__(other)

    @implements(getattr(_galsim.SED, "__truediv__", None))
    def __truediv__(self, other):
        return SED(self._wave, self._flux / other, self._redshift)

    @implements(_galsim.SED.__add__)
    def __add__(self, other):
        if isinstance(other, SED):
            other_flux = jnp.interp(
                self._wave * (1.0 + self._redshift),
                other._wave * (1.0 + other._redshift),
                other._flux,
                left=0.0,
                right=0.0,
            )
            return SED(self._wave, self._flux + other_flux, self._redshift)
        return SED(self._wave, self._flux + other, self._redshift)

    # ------------------------------------------------------------------
    # JAX pytree interface
    # ------------------------------------------------------------------

    def tree_flatten(self):
        """Flatten for JAX tracing.

        ``flux`` and ``redshift`` are traced children.
        ``wave`` is static auxiliary data (the interpolation grid).
        """
        children = (self._flux, self._redshift)
        # wave must be hashable for JAX cache keys; store as tuple of floats.
        aux_data = {"wave": tuple(self._wave.tolist())}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            wave=jnp.asarray(aux_data["wave"], dtype=float),
            flux=children[0],
            redshift=children[1],
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"SED(wave=[{self._wave[0]:.1f}, ..., {self._wave[-1]:.1f}] nm, "
            f"n_wave={len(self._wave)}, redshift={float(self._redshift):.4f})"
        )

    def __eq__(self, other):
        if not isinstance(other, SED):
            return False
        return (
            jnp.array_equal(self._wave, other._wave)
            & jnp.array_equal(self._flux, other._flux)
            & jnp.array_equal(self._redshift, other._redshift)
        )

    def __hash__(self):
        return hash(
            (
                "galsim.SED",
                ensure_hashable(self._wave),
                ensure_hashable(self._flux),
                ensure_hashable(self._redshift),
            )
        )
