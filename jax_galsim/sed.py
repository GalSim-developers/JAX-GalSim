"""Spectral Energy Distribution (SED) for chromatic profiles.

Designed for JAX compatibility: the flux array is a traced parameter,
so gradients flow through SED values (e.g. from DSPS outputs).
"""

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class SED:
    """Spectral Energy Distribution.

    Represents flux density as a function of wavelength, designed for
    full JAX compatibility. The ``flux`` array is a JAX-traced parameter,
    enabling gradients through SED parameters (e.g. outputs of DSPS).

    The wavelength grid (``wave``) is treated as static auxiliary data:
    it defines the interpolation structure and is not differentiated.

    Parameters
    ----------
    wave : array_like
        Wavelength array **in nanometers**. Must be strictly increasing.
        Treated as static (not traced by JAX).
    flux : array_like
        Flux density at each wavelength. Treated as a JAX-traced parameter.
        Units are arbitrary but must be consistent across the simulation
        (typically photons / nm / cm² / s for spectral SEDs, or
        dimensionless for shape-only profiles).
    redshift : float, optional
        Cosmological redshift applied to the SED.  The observed wavelength
        grid is shifted to ``wave_obs = wave_rest * (1 + redshift)``.
        Default 0.

    Examples
    --------
    Basic construction from arrays::

        >>> import jax.numpy as jnp
        >>> from jax_galsim.sed import SED
        >>> wave = jnp.linspace(300, 1100, 512)   # nm
        >>> flux = jnp.ones(512)
        >>> sed = SED(wave, flux)
        >>> float(sed(550.0))
        1.0

    DSPS workflow — flux is a traced JAX array::

        >>> flux = dsps_model(params)             # JAX array
        >>> sed = SED(dsps_wave_nm, flux)
        >>> image = chromatic_galaxy.drawImage(bandpass)  # differentiable

    Redshifted SED::

        >>> sed_z = SED(wave, flux, redshift=0.5)
        >>> sed_z(825.0)   # queries rest-frame 550 nm
    """

    def __init__(self, wave, flux, redshift=0.0):
        self._wave = jnp.asarray(wave, dtype=float)  # static, not traced
        self._flux = jnp.asarray(flux)               # traced
        self._redshift = jnp.asarray(redshift, dtype=float)

        if self._wave.ndim != 1 or len(self._wave) < 2:
            raise ValueError("wave must be a 1-D array with at least 2 elements.")
        if len(self._flux) != len(self._wave):
            raise ValueError("flux must have the same length as wave.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def wave(self):
        """Rest-frame wavelength grid in nm (JAX array, static)."""
        return self._wave

    @property
    def flux(self):
        """Flux density array (JAX array, traced)."""
        return self._flux

    @property
    def redshift(self):
        """Redshift (JAX scalar)."""
        return self._redshift

    @property
    def blue_limit(self):
        """Shortest observed wavelength in nm."""
        return float(self._wave[0]) * (1.0 + float(self._redshift))

    @property
    def red_limit(self):
        """Longest observed wavelength in nm."""
        return float(self._wave[-1]) * (1.0 + float(self._redshift))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def __call__(self, wave):
        """Evaluate flux density at observed wavelength(s) in nm.

        Uses linear interpolation; returns 0 outside the defined range.

        Parameters
        ----------
        wave : float or array_like
            Observed wavelength(s) in nm.

        Returns
        -------
        jnp.ndarray
            Flux density at the requested wavelengths.
        """
        wave = jnp.asarray(wave, dtype=float)
        # Convert observed wavelength to rest-frame before interpolating
        wave_rest = wave / (1.0 + self._redshift)
        return jnp.interp(wave_rest, self._wave, self._flux, left=0.0, right=0.0)

    # ------------------------------------------------------------------
    # Flux through a bandpass
    # ------------------------------------------------------------------

    def calculateFlux(self, bandpass, n_waves=512):
        """Integrate SED through a bandpass: ``∫ SED(λ) × BP(λ) dλ``.

        Parameters
        ----------
        bandpass : Bandpass
            Observing bandpass.
        n_waves : int, optional
            Number of quadrature points. Default 512.

        Returns
        -------
        jnp.ndarray
            Scalar flux value.
        """
        waves = jnp.linspace(bandpass.blue_limit, bandpass.red_limit, n_waves)
        return jnp.trapezoid(self(waves) * bandpass(waves), waves)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def withRedshift(self, redshift):
        """Return a copy of this SED at a new redshift."""
        return SED(self._wave, self._flux, redshift)

    def __mul__(self, other):
        """Multiply SED by a scalar or another SED.

        SED × scalar scales all flux values.
        SED × SED multiplies flux densities (both evaluated on ``self``'s grid).
        """
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
            from jax_galsim.chromatic import Chromatic

            return Chromatic(other, self)
        from jax_galsim.chromatic import ChromaticObject

        if isinstance(other, ChromaticObject):
            return other * self
        return SED(self._wave, self._flux * other, self._redshift)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return SED(self._wave, self._flux / other, self._redshift)

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
            and jnp.array_equal(self._flux, other._flux)
            and jnp.array_equal(self._redshift, other._redshift)
        )
