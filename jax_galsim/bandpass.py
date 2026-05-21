"""Bandpass filter representation for chromatic simulations."""

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Bandpass:
    """Wavelength-dependent throughput of an observing bandpass.

    Represents the combined throughput of optics, filter, and detector
    as a function of wavelength.  The throughput array is a JAX-traced
    parameter, so gradients can flow through it if needed (e.g. for
    filter-design optimisation).

    Parameters
    ----------
    wave : array_like
        Wavelength array **in nanometers**, strictly increasing.
        Treated as static (not traced by JAX).
    throughput : array_like
        Dimensionless throughput ∈ [0, 1] at each wavelength.
        Treated as a JAX-traced parameter.
    blue_limit : float, optional
        Override the short-wavelength cut-off in nm.  Defaults to
        ``wave[0]``.
    red_limit : float, optional
        Override the long-wavelength cut-off in nm.  Defaults to
        ``wave[-1]``.

    Examples
    --------
    Construct from arrays::

        >>> import jax.numpy as jnp
        >>> from jax_galsim.bandpass import Bandpass
        >>> wave = jnp.linspace(550, 700, 100)
        >>> bp = Bandpass(wave, jnp.ones(100))
        >>> float(bp(625.0))
        1.0
        >>> float(bp.effective_wavelength)
        625.0

    Top-hat filter between 600 nm and 700 nm::

        >>> wave = jnp.array([500., 600., 600.001, 700., 700.001, 800.])
        >>> thru = jnp.array([0.,   0.,   1.,      1.,   0.,      0.])
        >>> bp = Bandpass(wave, thru)
    """

    def __init__(self, wave, throughput, blue_limit=None, red_limit=None):
        self._wave = jnp.asarray(wave, dtype=float)
        self._throughput = jnp.asarray(throughput)

        if self._wave.ndim != 1 or len(self._wave) < 2:
            raise ValueError("wave must be a 1-D array with at least 2 elements.")
        if len(self._throughput) != len(self._wave):
            raise ValueError("throughput must have the same length as wave.")

        self._blue_limit = float(blue_limit) if blue_limit is not None else float(self._wave[0])
        self._red_limit = float(red_limit) if red_limit is not None else float(self._wave[-1])

        # Precompute effective wavelength at construction time so it is a
        # concrete Python float and can be used as a static value under JIT.
        _w = jnp.linspace(self._blue_limit, self._red_limit, 512)
        _t = jnp.interp(_w, self._wave, self._throughput)
        _norm = jnp.trapezoid(_t, _w)
        self._effective_wavelength_val = (
            float(jnp.trapezoid(_w * _t, _w) / _norm)
            if float(_norm) > 0
            else float(0.5 * (self._blue_limit + self._red_limit))
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def wave(self):
        """Wavelength grid in nm (JAX array, static)."""
        return self._wave

    @property
    def throughput(self):
        """Throughput array (JAX array, traced)."""
        return self._throughput

    @property
    def blue_limit(self):
        """Short-wavelength cut-off in nm."""
        return self._blue_limit

    @property
    def red_limit(self):
        """Long-wavelength cut-off in nm."""
        return self._red_limit

    @property
    def effective_wavelength(self):
        """Flux-weighted mean wavelength (concrete Python float, safe under JIT)."""
        return self._effective_wavelength_val

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def __call__(self, wave):
        """Evaluate throughput at wavelength(s) in nm.

        Returns 0 outside the defined range.

        Parameters
        ----------
        wave : float or array_like
            Wavelength(s) in nm.

        Returns
        -------
        jnp.ndarray
            Throughput at the requested wavelengths.
        """
        wave = jnp.asarray(wave, dtype=float)
        return jnp.interp(wave, self._wave, self._throughput, left=0.0, right=0.0)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __mul__(self, other):
        """Multiply throughput by a scalar or another Bandpass."""
        if isinstance(other, Bandpass):
            # Union wavelength grid
            wave = jnp.unique(jnp.concatenate([self._wave, other._wave]))
            t = jnp.interp(wave, self._wave, self._throughput, left=0.0, right=0.0)
            t2 = jnp.interp(wave, other._wave, other._throughput, left=0.0, right=0.0)
            blue = max(self._blue_limit, other._blue_limit)
            red = min(self._red_limit, other._red_limit)
            return Bandpass(wave, t * t2, blue_limit=blue, red_limit=red)
        return Bandpass(self._wave, self._throughput * other,
                        self._blue_limit, self._red_limit)

    def __rmul__(self, other):
        return self.__mul__(other)

    def truncate(self, blue_limit=None, red_limit=None):
        """Return a new Bandpass with tighter wavelength limits."""
        blue = blue_limit if blue_limit is not None else self._blue_limit
        red = red_limit if red_limit is not None else self._red_limit
        return Bandpass(self._wave, self._throughput,
                        blue_limit=blue, red_limit=red)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def tophat(cls, blue_limit, red_limit, n_wave=100):
        """Uniform throughput = 1 between blue_limit and red_limit."""
        wave = jnp.linspace(blue_limit, red_limit, n_wave)
        return cls(wave, jnp.ones(n_wave))

    # ------------------------------------------------------------------
    # JAX pytree interface
    # ------------------------------------------------------------------

    def tree_flatten(self):
        children = (self._throughput,)
        aux_data = {
            "wave": tuple(self._wave.tolist()),
            "blue_limit": self._blue_limit,
            "red_limit": self._red_limit,
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

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"Bandpass(wave=[{self._blue_limit:.1f}, {self._red_limit:.1f}] nm, "
            f"lam_eff={self.effective_wavelength:.1f} nm)"
        )
