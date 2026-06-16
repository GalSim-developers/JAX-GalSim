import os

import galsim as _galsim
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from jax_galsim.core.interpolate import akima_interp, akima_interp_coeffs
from jax_galsim.core.utils import cast_to_float, ensure_hashable, implements


@implements(
    _galsim.Bandpass,
    lax_description="""\
JAX-GalSim supports array-backed bandpasses only.  The wavelength grid is
static and the throughput array may be traced.  File input, string
expressions, zeropoints, units, and LookupTable metadata are not implemented.
""",
)
@register_pytree_node_class
class Bandpass:
    def __init__(self, wave, throughput, blue_limit=None, red_limit=None):
        if isinstance(wave, (str, bytes, os.PathLike)) or isinstance(
            throughput, (str, bytes, os.PathLike)
        ):
            raise NotImplementedError(
                "JAX-GalSim Bandpass supports array-backed bandpasses only; "
                "file input, string expressions, and unit strings are not implemented."
            )

        self._wave = jnp.asarray(wave, dtype=float)
        self._throughput = jnp.asarray(throughput)

        if self._wave.ndim != 1 or len(self._wave) < 2:
            raise ValueError("wave must be a 1-D array with at least 2 elements.")
        if self._throughput.shape != self._wave.shape:
            raise ValueError("throughput must have the same shape as wave.")

        self._blue_limit = (
            cast_to_float(blue_limit)
            if blue_limit is not None
            else cast_to_float(self._wave[0])
        )
        self._red_limit = (
            cast_to_float(red_limit)
            if red_limit is not None
            else cast_to_float(self._wave[-1])
        )

    @property
    @implements(
        None,
        lax_description="JAX-GalSim-specific static wavelength grid for array-backed bandpasses.",
    )
    def wave(self):
        return self._wave

    @property
    @implements(
        None,
        lax_description="JAX-GalSim-specific traced throughput array for array-backed bandpasses.",
    )
    def throughput(self):
        return self._throughput

    @property
    @implements(
        None,
        lax_description="JAX-GalSim-specific blue wavelength limit.",
    )
    def blue_limit(self):
        return self._blue_limit

    @property
    @implements(
        None,
        lax_description="JAX-GalSim-specific red wavelength limit.",
    )
    def red_limit(self):
        return self._red_limit

    @property
    @implements(_galsim.Bandpass.effective_wavelength)
    def effective_wavelength(self):
        return self.calculateEffectiveWavelength()

    @implements(
        _galsim.Bandpass.calculateEffectiveWavelength,
        lax_description=(
            "The JAX implementation returns a traced scalar when throughput is traced."
        ),
    )
    def calculateEffectiveWavelength(self, precise=False):
        if precise:
            raise NotImplementedError(
                "JAX-GalSim Bandpass does not support precise=True."
            )
        waves = jnp.linspace(self._blue_limit, self._red_limit, 512)
        throughput = self(waves)
        norm = jnp.trapezoid(throughput, waves)
        midpoint = 0.5 * (self._blue_limit + self._red_limit)
        safe_norm = jnp.where(norm > 0, norm, 1.0)
        return jnp.where(
            norm > 0,
            jnp.trapezoid(waves * throughput, waves) / safe_norm,
            midpoint,
        )

    @implements(_galsim.Bandpass.__call__)
    def __call__(self, wave):
        wave = jnp.asarray(wave, dtype=float)
        if len(self._wave) >= 5:
            coeffs = akima_interp_coeffs(self._wave, self._throughput, use_jax=True)
            throughput = akima_interp(wave, self._wave, self._throughput, coeffs)
        else:
            throughput = jnp.interp(
                wave, self._wave, self._throughput, left=0.0, right=0.0
            )
        in_band = (wave >= self._blue_limit) & (wave <= self._red_limit)
        return jnp.where(in_band, throughput, 0.0)

    @implements(_galsim.Bandpass.__mul__)
    def __mul__(self, other):
        if isinstance(other, Bandpass):
            wave = jnp.unique(jnp.concatenate([self._wave, other._wave]))
            t = jnp.interp(wave, self._wave, self._throughput, left=0.0, right=0.0)
            t2 = jnp.interp(wave, other._wave, other._throughput, left=0.0, right=0.0)
            blue = max(self._blue_limit, other._blue_limit)
            red = min(self._red_limit, other._red_limit)
            return Bandpass(wave, t * t2, blue_limit=blue, red_limit=red)
        return Bandpass(
            self._wave,
            self._throughput * other,
            self._blue_limit,
            self._red_limit,
        )

    @implements(getattr(_galsim.Bandpass, "__rmul__", None))
    def __rmul__(self, other):
        return self.__mul__(other)

    @implements(_galsim.Bandpass.truncate)
    def truncate(self, blue_limit=None, red_limit=None):
        blue = blue_limit if blue_limit is not None else self._blue_limit
        red = red_limit if red_limit is not None else self._red_limit
        return Bandpass(
            self._wave,
            self._throughput,
            blue_limit=blue,
            red_limit=red,
        )

    @classmethod
    @implements(
        None,
        lax_description=(
            "Construct a unit-throughput tophat bandpass on a static JAX wavelength grid."
        ),
    )
    def tophat(cls, blue_limit, red_limit, n_wave=100):
        """Uniform throughput = 1 between blue_limit and red_limit."""
        wave = jnp.linspace(blue_limit, red_limit, n_wave)
        return cls(wave, jnp.ones(n_wave))

    def tree_flatten(self):
        children = (self._throughput,)
        aux_data = {
            "wave": tuple(self._wave.tolist()),
            "blue_limit": ensure_hashable(self._blue_limit),
            "red_limit": ensure_hashable(self._red_limit),
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            wave=jnp.asarray(aux_data["wave"], dtype=float),
            throughput=children[0],
            blue_limit=aux_data["blue_limit"],
            red_limit=aux_data["red_limit"],
        )

    def __repr__(self):
        return (
            f"Bandpass(wave=[{self._blue_limit:.1f}, {self._red_limit:.1f}] nm, "
            f"lam_eff={float(self.effective_wavelength):.1f} nm)"
        )

    def __eq__(self, other):
        if not isinstance(other, Bandpass):
            return False
        return (
            jnp.array_equal(self._wave, other._wave)
            & jnp.array_equal(self._throughput, other._throughput)
            & jnp.array_equal(self._blue_limit, other._blue_limit)
            & jnp.array_equal(self._red_limit, other._red_limit)
        )

    def __hash__(self):
        return hash(
            (
                "galsim.Bandpass",
                ensure_hashable(self._wave),
                ensure_hashable(self._throughput),
                ensure_hashable(self._blue_limit),
                ensure_hashable(self._red_limit),
            )
        )
