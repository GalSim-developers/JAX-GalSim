"""Chromatic (wavelength-dependent) profiles for jax_galsim.

Architecture overview
---------------------
Every chromatic object exposes:

* ``evaluateAtWavelength(wave) -> GSObject``
  Returns the monochromatic profile at wavelength *wave* (nm).  The
  returned GSObject may carry wavelength-dependent parameters as JAX
  traced values, so the function is vmappable.

* ``drawImage(bandpass, n_waves=64, **kwargs) -> Image``
  Integrates the profile over *bandpass* using a static wavelength grid
  of *n_waves* points (trapezoid rule).  ``n_waves`` is static at
  JIT-compile time; everything else may be traced.

Hierarchy
---------
::

    ChromaticObject            base class, non-separable draw by default
    ├── Chromatic              GSObject × SED  (separable, fast path)
    ├── ChromaticAtmosphere    Gaussian PSF with FWHM ∝ λ^alpha
    └── ChromaticConvolution   convolution of any chromatic objects

Separable vs non-separable
--------------------------
*Separable* means ``I(x, y, λ) = g(x, y) × h(λ)``.  For a separable
object the integration reduces to a single monochromatic draw:

    flux = ∫ SED(λ) × BP(λ) dλ
    image = g(x, y) drawn with total flux

*Non-separable* objects (e.g. ChromaticAtmosphere) must evaluate the
k-space image at every wavelength sample and integrate:

    K_eff(k) = ∫ K(k, λ) × BP(λ) dλ
    image    = IFFT[ K_eff ]

JAX compatibility
-----------------
* All arithmetic uses ``jax.numpy``.
* Wavelength grids are **static** (fixed-size) to allow ``jax.jit``.
* ``jax.vmap`` vectorises the per-wavelength k-value computation.
* ``jax.grad`` flows through SED flux arrays (e.g. DSPS outputs).
"""

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from jax_galsim.gsparams import GSParams
from jax_galsim.position import PositionD


def _pixel_scale_from_kwargs(kwargs):
    """Return pixel scale in world units (arcsec/pixel) as a concrete float.

    Reads ``scale`` from kwargs; falls back to reading the WCS object if
    the ``wcs`` kwarg is provided, or 1.0 otherwise.  The value from
    ``kwargs['scale']`` is always a Python float (user-supplied literal),
    so this is safe to call inside ``jax.jit``.
    """
    if "scale" in kwargs:
        return float(kwargs["scale"])
    if "wcs" in kwargs:
        wcs = kwargs["wcs"]
        if hasattr(wcs, "_scale"):
            return float(wcs._scale)
    return 1.0


def _make_setup_image(profile, kwargs):
    """Build draw target without running full ``drawImage`` setup when size is fixed."""
    from jax_galsim.image import Image

    image = kwargs.get("image", None)
    if image is not None:
        return Image(image=image)

    image_kwargs = {
        "dtype": kwargs.get("dtype", None),
        "scale": kwargs.get("scale", None),
        "wcs": kwargs.get("wcs", None),
    }
    image_kwargs = {k: v for k, v in image_kwargs.items() if v is not None}

    bounds = kwargs.get("bounds", None)
    if bounds is not None:
        return Image(bounds=bounds, **image_kwargs)

    nx = kwargs.get("nx", None)
    ny = kwargs.get("ny", None)
    if nx is not None and ny is not None:
        return Image(nx, ny, **image_kwargs)

    return profile.drawImage(setup_only=True, **kwargs)


def _next_pow2(n):
    n = int(n)
    out = 1
    while out < n:
        out *= 2
    return out


def _fix_fft_size_for_image(profile, image):
    """Pin FFT size for JIT-safe setup when output bounds are already fixed."""
    nrow, ncol = image.array.shape
    n = max(128, _next_pow2(2 * max(nrow, ncol)))
    n = min(n, profile.gsparams.maximum_fft_size)
    return profile.withGSParams(minimum_fft_size=n, maximum_fft_size=n)


def _static_kcoords(kimage, wrap_size, pixel_scale):
    """Return k-space pixel centers as a JAX array with static shape."""
    nrow, ncol = kimage.array.shape
    x = jnp.arange(ncol, dtype=float)
    y = jnp.arange(nrow, dtype=float) - wrap_size // 2
    kx, ky = jnp.meshgrid(x, y)
    dk = 2.0 * jnp.pi / (wrap_size * pixel_scale)
    return jnp.stack([kx.ravel(), ky.ravel()], axis=-1) * dk


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class ChromaticObject:
    """Base class for wavelength-dependent profiles.

    Subclasses must override :meth:`evaluateAtWavelength`.  The default
    :meth:`drawImage` uses a k-space trapezoid integration that works for
    any subclass; separable subclasses override it with a faster path.
    """

    #: Set True in separable subclasses.
    _separable: bool = False

    def __init__(self, obj=None):
        if obj is None:
            self._base_obj = None
            return

        from jax_galsim.gsobject import GSObject

        if not isinstance(obj, GSObject):
            raise TypeError("ChromaticObject requires a GSObject.")
        self._base_obj = obj
        self._separable = True

    @property
    def separable(self):
        """True if the profile factors as g(x,y) × h(λ)."""
        return self._separable

    # ------------------------------------------------------------------
    # Interface that subclasses must implement
    # ------------------------------------------------------------------

    def evaluateAtWavelength(self, wave):
        """Return the monochromatic GSObject at wavelength *wave* (nm).

        This method must be JAX-traceable: all internal computations
        should use ``jax.numpy``, and the returned GSObject's parameters
        may be abstract JAX tracers.

        Parameters
        ----------
        wave : float or jax scalar
            Wavelength in nm.

        Returns
        -------
        GSObject
        """
        if getattr(self, "_base_obj", None) is not None:
            return self._base_obj
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement evaluateAtWavelength."
        )

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def drawImage(self, bandpass, n_waves=64, **kwargs):
        """Draw the bandpass-integrated image.

        Parameters
        ----------
        bandpass : Bandpass
            Observing bandpass.
        n_waves : int
            Number of wavelength samples for numerical integration.
            Must be a **static** integer (fixed at JIT compile time).
        **kwargs
            Forwarded to the underlying ``GSObject.drawImage`` calls.
            Typical keys: ``nx``, ``ny``, ``scale``, ``method``, etc.

        Returns
        -------
        Image
        """
        if self._separable:
            return self._drawImage_separable(bandpass, n_waves, **kwargs)
        return self._drawImage_nonseparable(bandpass, n_waves, **kwargs)

    # ------------------------------------------------------------------
    # Separable fast path
    # ------------------------------------------------------------------

    def _drawImage_separable(self, bandpass, n_waves, **kwargs):
        """FFT draw, scaling by total SED×BP flux (traced-safe under JIT).

        Design: split into two phases.

        **Phase 1 - static setup**:
        Build image bounds, k-grid, and base k-values using the unit-flux
        spatial profile.  All shape parameters must be concrete Python
        scalars at this stage (true for ``Chromatic`` where the spatial
        profile has static params); the SED flux is NOT evaluated here.

        **Phase 2 — traced computation**:
        Compute total_flux = ∫ SED(λ)×BP(λ) dλ (may be a JAX traced
        value, e.g. DSPS output), multiply into k-values, IFFT.
        """
        from jax_galsim.box import Pixel
        from jax_galsim.convolve import Convolve
        from jax_galsim.image import Image

        wave_eff = bandpass.effective_wavelength  # concrete Python float
        pixel_scale = _pixel_scale_from_kwargs(kwargs)

        # ------------------------------------------------------------------
        # Phase 1: concrete setup. Under jit this runs during tracing.
        # ------------------------------------------------------------------
        with jax.disable_jit():
            # Unit-flux spatial profile — shape params are concrete Python floats
            spatial_prof = self._static_spatial_profile(wave_eff)
            image = _make_setup_image(spatial_prof, kwargs)
            spatial_prof = _fix_fft_size_for_image(spatial_prof, image)
            original_center = image.center
            original_wcs = image.wcs
            image.setCenter(0, 0)

            pixel = Pixel(scale=pixel_scale, gsparams=spatial_prof.gsparams)
            prof_conv = Convolve([spatial_prof, pixel], gsparams=spatial_prof.gsparams)
            kimage, wrap_size = prof_conv.drawFFT_makeKImage(image)

            kcoords = _static_kcoords(kimage, wrap_size, pixel_scale)

            # Static k-values (unit flux, no SED scaling).
            kvals_static = jax.vmap(
                lambda k: prof_conv._kValue(PositionD(k[0], k[1]))
            )(kcoords)

            # Apply the same -0.5 pixel centering correction that gsobject._adjust_offset
            # uses for even-sized images (avoids 0.5-pixel offset vs non-chromatic draws).
            img_shape = image.array.shape   # (ny, nx); unchanged by setCenter
            dx_corr = -0.5 * pixel_scale * ((img_shape[1] + 1) % 2)
            dy_corr = -0.5 * pixel_scale * ((img_shape[0] + 1) % 2)
            phase_corr = jnp.exp(
                -1j * (kcoords[:, 0] * dx_corr + kcoords[:, 1] * dy_corr)
            )
            kvals_static = kvals_static * phase_corr

            kshape = kimage.array.shape
            kbounds = kimage.bounds
            kwcs = kimage.wcs

        # ------------------------------------------------------------------
        # Phase 2: traced computation — integrate SED × bandpass
        # ------------------------------------------------------------------
        waves = jnp.linspace(bandpass.blue_limit, bandpass.red_limit, n_waves)
        weights = jax.vmap(lambda w: self._sed_value(w) * bandpass(w))(waves)
        total_flux = jnp.trapezoid(weights, waves)

        # Scale pre-computed k-values by traced total flux
        kvals = kvals_static * total_flux
        karray = kvals.reshape(kshape).astype(kimage.dtype)
        eff_kimage = Image(
            array=karray, bounds=kbounds, wcs=kwcs, _check_bounds=False
        )
        prof_conv.drawFFT_finish(image, eff_kimage, wrap_size, add_to_image=False)

        image.shift(original_center)
        image.wcs = original_wcs
        return image

    def _static_spatial_profile(self, wave_eff):
        """Return unit-flux spatial profile at *wave_eff* with static params.

        *wave_eff* must be a concrete Python float (use
        ``bandpass.effective_wavelength``).  Subclasses override this when
        they have a dedicated static spatial object (e.g. ``Chromatic`` has
        ``self.obj``).  The default calls ``evaluateAtWavelength`` with a
        Python float — works when all shape params are Python scalars.
        """
        return self.evaluateAtWavelength(float(wave_eff)).withFlux(1.0)

    def _sed_value(self, wave):
        """SED flux density at *wave*.  Subclasses override if needed."""
        if getattr(self, "_base_obj", None) is not None:
            return jnp.ones((), dtype=float)
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Non-separable k-space integration
    # ------------------------------------------------------------------

    def _drawImage_nonseparable(self, bandpass, n_waves, **kwargs):
        """Integrate in k-space: K_eff = ∫ K(k,λ) × BP(λ) dλ, then IFFT.

        Mirrors the drawFFT pipeline of GSObject.drawImage exactly, split
        into a concrete setup phase and a traced computation phase.
        """
        from jax_galsim.box import Pixel
        from jax_galsim.convolve import Convolve
        from jax_galsim.image import Image

        wave_eff = bandpass.effective_wavelength  # static Python float
        pixel_scale = _pixel_scale_from_kwargs(kwargs)

        # ------------------------------------------------------------------
        # Phase 1: concrete setup. Under jit this runs during tracing.
        # ------------------------------------------------------------------
        with jax.disable_jit():
            prof0 = self._static_spatial_profile(wave_eff)
            image = _make_setup_image(prof0, kwargs)
            prof0 = _fix_fft_size_for_image(prof0, image)
            original_center = image.center
            original_wcs = image.wcs
            image.setCenter(0, 0)

            pixel = Pixel(scale=pixel_scale, gsparams=prof0.gsparams)
            prof0_conv = Convolve([prof0, pixel], gsparams=prof0.gsparams)
            kimage, wrap_size = prof0_conv.drawFFT_makeKImage(image)

            kcoords = _static_kcoords(kimage, wrap_size, pixel_scale)

            pixel_kvals = jax.vmap(
                lambda k: pixel._kValue(PositionD(k[0], k[1]))
            )(kcoords)

            # Apply the -0.5 pixel centering correction for even-sized images.
            img_shape = image.array.shape
            dx_corr = -0.5 * pixel_scale * ((img_shape[1] + 1) % 2)
            dy_corr = -0.5 * pixel_scale * ((img_shape[0] + 1) % 2)
            phase_corr = jnp.exp(
                -1j * (kcoords[:, 0] * dx_corr + kcoords[:, 1] * dy_corr)
            )
            pixel_kvals = pixel_kvals * phase_corr

            kshape = kimage.array.shape
            kbounds = kimage.bounds
            kwcs = kimage.wcs

        # ------------------------------------------------------------------
        # Phase 2: traced computation
        # ------------------------------------------------------------------
        waves = jnp.linspace(bandpass.blue_limit, bandpass.red_limit, n_waves)

        def kvals_at_wave(wave):
            prof = self.evaluateAtWavelength(wave)
            kv = jax.vmap(
                lambda k: prof._kValue(PositionD(k[0], k[1]))
            )(kcoords)
            return kv * bandpass(wave)

        all_kvals = jax.vmap(kvals_at_wave)(waves)   # (n_waves, n_k)
        eff_kvals = jnp.trapezoid(all_kvals, waves, axis=0)
        eff_kvals = eff_kvals * pixel_kvals

        eff_karray = eff_kvals.reshape(kshape).astype(kimage.dtype)
        eff_kimage = Image(
            array=eff_karray,
            bounds=kbounds,
            wcs=kwcs,
            _check_bounds=False,
        )

        # IFFT back to pixel space
        prof0_conv.drawFFT_finish(image, eff_kimage, wrap_size, add_to_image=False)

        # Restore original center and WCS (same as drawImage does after drawFFT)
        image.shift(original_center)
        image.wcs = original_wcs

        return image

    # ------------------------------------------------------------------
    # Operator overloads
    # ------------------------------------------------------------------

    def __add__(self, other):
        return ChromaticSum([self, other])

    def __radd__(self, other):
        return ChromaticSum([other, self])

    def __mul__(self, other):
        from jax_galsim.sed import SED

        if isinstance(other, SED):
            base_obj = getattr(self, "_base_obj", None)
            if base_obj is None:
                raise TypeError("Only achromatic ChromaticObject wrappers can be multiplied by SED.")
            return Chromatic(base_obj, other)
        return _ScaledChromaticObject(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)


class _ScaledChromaticObject(ChromaticObject):
    """Chromatic object with wavelength-independent flux scaling."""

    def __init__(self, obj, scale):
        self.obj = obj
        self.scale = scale
        self._separable = obj._separable

    def evaluateAtWavelength(self, wave):
        return self.obj.evaluateAtWavelength(wave).withScaledFlux(self.scale)

    def _static_spatial_profile(self, wave_eff):
        return self.obj._static_spatial_profile(wave_eff)

    def _sed_value(self, wave):
        return self.obj._sed_value(wave) * self.scale


# ---------------------------------------------------------------------------
# ChromaticSum — sum of chromatic objects
# ---------------------------------------------------------------------------


class ChromaticSum(ChromaticObject):
    """Sum of two or more chromatic profiles.

    The combined SED is the sum of all component SEDs.  Drawing
    evaluates each component at every wavelength and sums.

    Parameters
    ----------
    obj_list : list of ChromaticObject
    """

    _separable = False  # conservative; optimised later if all are separable

    def __init__(self, *args):
        if len(args) == 0:
            raise TypeError("At least one object must be provided.")
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            self.obj_list = list(args[0])
        else:
            self.obj_list = list(args)
        self._separable = all(o._separable for o in self.obj_list)

    def evaluateAtWavelength(self, wave):
        from jax_galsim.sum import Sum

        return Sum([o.evaluateAtWavelength(wave) for o in self.obj_list])

    def drawImage(self, bandpass, n_waves=64, **kwargs):
        # Draw each component and sum
        images = [obj.drawImage(bandpass, n_waves=n_waves, **kwargs)
                  for obj in self.obj_list]
        result = images[0]
        for img in images[1:]:
            result._array = result._array + img._array
        return result


# ---------------------------------------------------------------------------
# Chromatic — separable GSObject × SED
# ---------------------------------------------------------------------------


@register_pytree_node_class
class Chromatic(ChromaticObject):
    """Separable chromatic profile: a GSObject multiplied by an SED.

    The spatial profile is fixed; wavelength dependence enters only
    through the SED flux scaling.

    ``I(x, y, λ) = g(x, y) × SED(λ)``

    Drawing a ``Chromatic`` through a bandpass reduces to a single
    monochromatic draw at the effective wavelength with total flux
    ``∫ SED(λ) × BP(λ) dλ``.

    Parameters
    ----------
    obj : GSObject
        Normalised spatial profile (flux = 1 by convention, though
        any flux is allowed and will be multiplied by the SED).
    sed : SED
        Spectral energy distribution.

    Examples
    --------
    ::

        >>> from jax_galsim import Gaussian
        >>> from jax_galsim.sed import SED
        >>> from jax_galsim.bandpass import Bandpass
        >>> import jax.numpy as jnp

        >>> wave = jnp.linspace(500, 900, 256)
        >>> sed = SED(wave, jnp.ones(256))
        >>> bp  = Bandpass.tophat(550, 750)
        >>> gal = Gaussian(half_light_radius=0.5) * sed
        >>> img = gal.drawImage(bp, scale=0.2, nx=64, ny=64)
    """

    _separable = True

    def __init__(self, obj, sed):
        self.obj = obj
        self.sed = sed

    def evaluateAtWavelength(self, wave):
        """Return the GSObject scaled to SED(wave)."""
        return self.obj.withScaledFlux(self.sed(wave))

    def _sed_value(self, wave):
        return self.sed(wave)

    def _static_spatial_profile(self, wave_eff):
        """Return self.obj (unit flux) — shape params are always static."""
        return self.obj.withFlux(1.0)

    # JAX pytree: obj and sed are both pytrees
    def tree_flatten(self):
        children = (self.obj, self.sed)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(obj=children[0], sed=children[1])

    def __repr__(self):
        return f"Chromatic({self.obj!r}, {self.sed!r})"


# ---------------------------------------------------------------------------
# ChromaticAtmosphere — seeing PSF with wavelength-dependent FWHM
# ---------------------------------------------------------------------------


@register_pytree_node_class
class ChromaticAtmosphere(ChromaticObject):
    """Atmospheric PSF with a power-law wavelength-dependent FWHM.

    The PSF profile at wavelength λ is a Gaussian (or Moffat) with:

        FWHM(λ) = fwhm_ref × (λ / lam_ref)^alpha

    For Kolmogorov turbulence, the expected scaling is α ≈ −0.2.

    This profile carries a **flat (dimensionless) SED**: ``SED(λ) = 1``.
    The physical SED is typically attached to the galaxy component via
    :class:`Chromatic`, and passed to :class:`ChromaticConvolution`.

    Parameters
    ----------
    fwhm_ref : float
        FWHM in arcseconds at the reference wavelength.
    lam_ref : float
        Reference wavelength in nm.
    alpha : float, optional
        Power-law index.  Default −0.2 (Kolmogorov).
    profile : {'gaussian', 'moffat'}
        Profile type.  Default ``'gaussian'``.
    moffat_beta : float, optional
        Moffat β parameter (only used when ``profile='moffat'``).
        Default 4.765 (typical for atmospheric seeing).
    gsparams : GSParams, optional

    Examples
    --------
    ::

        >>> from jax_galsim.chromatic import ChromaticAtmosphere
        >>> psf = ChromaticAtmosphere(fwhm_ref=0.7, lam_ref=700.0, alpha=-0.2)
        >>> prof = psf.evaluateAtWavelength(550.0)   # Gaussian at 550 nm
    """

    _separable = False

    def __init__(
        self,
        fwhm_ref,
        lam_ref,
        alpha=-0.2,
        profile="gaussian",
        moffat_beta=4.765,
        gsparams=None,
    ):
        # All shape params stored as Python floats (static, not JAX-traced).
        # This enables jax.jit compatibility without pinning the FFT size.
        # To differentiate through fwhm_ref, use gsparams with fixed fft_size:
        #   GSParams(minimum_fft_size=N, maximum_fft_size=N)
        self._fwhm_ref = float(fwhm_ref)
        self._lam_ref = float(lam_ref)
        self._alpha = float(alpha)
        self._profile = profile
        self._moffat_beta = float(moffat_beta)
        self._gsparams = GSParams.check(gsparams)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fwhm_ref(self):
        return self._fwhm_ref

    @property
    def lam_ref(self):
        return self._lam_ref

    @property
    def alpha(self):
        return self._alpha

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def evaluateAtWavelength(self, wave):
        """Return the PSF profile (unit flux) at wavelength *wave* (nm).

        FWHM is scaled as ``fwhm_ref × (wave / lam_ref)^alpha``.
        When *wave* is a JAX tracer (inside vmap/jit), fwhm is also traced.
        """
        fwhm = self._fwhm_ref * (wave / self._lam_ref) ** self._alpha

        if self._profile == "gaussian":
            from jax_galsim.gaussian import Gaussian

            return Gaussian(fwhm=fwhm, flux=1.0, gsparams=self._gsparams)

        elif self._profile == "moffat":
            from jax_galsim.moffat import Moffat

            return Moffat(
                fwhm=fwhm,
                beta=self._moffat_beta,
                flux=1.0,
                gsparams=self._gsparams,
            )
        else:
            raise ValueError(
                f"Unknown profile type '{self._profile}'. "
                "Expected 'gaussian' or 'moffat'."
            )

    def _sed_value(self, wave):
        """Flat SED: unit flux at all wavelengths."""
        return jnp.ones((), dtype=float)

    # ------------------------------------------------------------------
    # JAX pytree interface
    # ------------------------------------------------------------------

    def tree_flatten(self):
        # No traced children — all params are static Python scalars.
        children = ()
        aux_data = {
            "fwhm_ref": self._fwhm_ref,
            "lam_ref": self._lam_ref,
            "alpha": self._alpha,
            "profile": self._profile,
            "moffat_beta": self._moffat_beta,
            "gsparams": self._gsparams,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            fwhm_ref=aux_data["fwhm_ref"],
            lam_ref=aux_data["lam_ref"],
            alpha=aux_data["alpha"],
            profile=aux_data["profile"],
            moffat_beta=aux_data["moffat_beta"],
            gsparams=aux_data["gsparams"],
        )

    def __repr__(self):
        return (
            f"ChromaticAtmosphere(fwhm_ref={float(self._fwhm_ref):.3f}, "
            f"lam_ref={self._lam_ref:.0f} nm, alpha={self._alpha:.2f}, "
            f"profile={self._profile!r})"
        )


# ---------------------------------------------------------------------------
# ChromaticConvolution — convolution of chromatic objects
# ---------------------------------------------------------------------------


class ChromaticConvolution(ChromaticObject):
    """Convolution of multiple chromatic profiles.

    Computes the bandpass-integrated image of the convolution:

        K_eff(k) = ∫ ∏_i K_i(k, λ) × BP(λ) dλ

    where each ``K_i(k, λ)`` is the k-space value of the i-th component
    at wavelength λ.

    **Optimised case**: when all components except one are separable
    (e.g. galaxy × SED convolved with a chromatic PSF), the separable
    profiles are extracted and convolved with the wavelength-integrated
    effective PSF.  This avoids multiplying the same galaxy k-image at
    every wavelength sample.

    Parameters
    ----------
    obj_list : list of ChromaticObject or GSObject
        Components to convolve.  Plain ``GSObject`` instances are wrapped
        automatically in a flat-SED :class:`Chromatic`.

    Examples
    --------
    ::

        >>> from jax_galsim import Gaussian, Convolve
        >>> from jax_galsim.chromatic import Chromatic, ChromaticAtmosphere, ChromaticConvolution
        >>> from jax_galsim.sed import SED
        >>> from jax_galsim.bandpass import Bandpass
        >>> import jax.numpy as jnp

        >>> wave = jnp.linspace(300, 1100, 512)
        >>> flux = jnp.exp(-0.5 * ((wave - 700) / 150) ** 2)  # Gaussian SED
        >>> sed = SED(wave, flux)
        >>> bp  = Bandpass.tophat(550, 800)

        >>> gal = Gaussian(half_light_radius=0.5) * sed
        >>> psf = ChromaticAtmosphere(fwhm_ref=0.7, lam_ref=700.0, alpha=-0.2)
        >>> final = ChromaticConvolution([gal, psf])
        >>> img = final.drawImage(bp, scale=0.2, nx=64, ny=64)
    """

    _separable = False

    def __init__(self, *args, **kwargs):
        from jax_galsim.gsobject import GSObject

        if len(args) == 0:
            raise TypeError("At least one object must be provided.")
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            obj_list = list(args[0])
        else:
            obj_list = list(args)

        real_space = kwargs.pop("real_space", None)
        self._gsparams = GSParams.check(kwargs.pop("gsparams", None))
        self._propagate_gsparams = kwargs.pop("propagate_gsparams", True)
        if kwargs:
            raise TypeError(
                "ChromaticConvolution constructor got unexpected keyword argument(s): %s"
                % kwargs.keys()
            )
        if real_space:
            raise NotImplementedError("Real-space chromatic convolutions are not implemented")

        # Wrap plain GSObjects with a flat SED so they fit the interface
        processed = []
        for obj in obj_list:
            if isinstance(obj, ChromaticConvolution):
                processed.extend(obj.obj_list)
                continue
            if isinstance(obj, GSObject):
                from jax_galsim.sed import SED

                wave_stub = jnp.array([100.0, 2000.0])
                flat_sed = SED(wave_stub, jnp.ones(2))
                processed.append(Chromatic(obj, flat_sed))
            else:
                processed.append(obj)
        self.obj_list = processed

    @property
    def gsparams(self):
        return self._gsparams

    def withGSParams(self, gsparams=None, **kwargs):
        ret = self.__class__.__new__(self.__class__)
        ret.obj_list = self.obj_list
        ret._gsparams = GSParams.check(gsparams, self._gsparams, **kwargs)
        ret._propagate_gsparams = self._propagate_gsparams
        return ret

    def evaluateAtWavelength(self, wave):
        """Return the convolved monochromatic profile at *wave* (nm)."""
        from jax_galsim.convolve import Convolve

        return Convolve([o.evaluateAtWavelength(wave) for o in self.obj_list])

    # ------------------------------------------------------------------
    # Drawing with the optimised separable split
    # ------------------------------------------------------------------

    def drawImage(self, bandpass, n_waves=64, **kwargs):
        """Draw the integrated image, exploiting separability where possible.

        Algorithm:

        1. Separate components into *separable* (galaxy × SED) and
           *non-separable* (chromatic PSF).
        2. Build the combined weight function:
           ``w(λ) = ∏_sep SED_i(λ) × BP(λ)``
        3. In k-space, integrate non-separable components:
           ``K_nonsep_eff(k) = ∫ ∏_nonsep K_i(k,λ) × w(λ) dλ``
        4. Multiply by separable component k-values (λ-independent after
           extracting their SED into the weight).
        5. Include pixel convolution, IFFT to real space.

        All-separable case falls back to a single monochromatic draw.
        """
        from jax_galsim.box import Pixel
        from jax_galsim.convolve import Convolve
        from jax_galsim.image import Image

        sep_objs = [o for o in self.obj_list if o._separable]
        nonsep_objs = [o for o in self.obj_list if not o._separable]

        wave_eff = bandpass.effective_wavelength  # static Python float
        waves = jnp.linspace(bandpass.blue_limit, bandpass.red_limit, n_waves)
        pixel_scale = _pixel_scale_from_kwargs(kwargs)

        # -------------------------------------------------------------------
        # Phase 1: concrete setup. Under jit this runs during tracing.
        #
        # Shape parameters (sigma, fwhm) must NOT be JIT-traced inputs here.
        # If they are, pin the FFT size via gsparams(min/max_fft_size=N).
        # -------------------------------------------------------------------
        with jax.disable_jit():
            if not nonsep_objs:
                # All-separable: use static spatial profiles (avoids traced SED)
                spatial_profs = [o._static_spatial_profile(wave_eff) for o in sep_objs]
                grid_prof = Convolve(spatial_profs)
            else:
                # Mixed: sep objects use static profiles; nonsep objects are evaluated
                # at wave_eff (their shape params must be concrete at this point).
                fiducial_profs = [
                    o._static_spatial_profile(wave_eff) if o._separable
                    else o.evaluateAtWavelength(wave_eff)
                    for o in self.obj_list
                ]
                grid_prof = Convolve(fiducial_profs)

            image = _make_setup_image(grid_prof, kwargs)
            grid_prof = _fix_fft_size_for_image(grid_prof, image)
            original_center = image.center
            original_wcs = image.wcs
            image.setCenter(0, 0)

            pixel = Pixel(scale=pixel_scale)
            grid_prof_conv = Convolve([grid_prof, pixel], gsparams=grid_prof.gsparams)
            kimage, wrap_size = grid_prof_conv.drawFFT_makeKImage(image)

            # k-space coordinates with static shape.
            kcoords = _static_kcoords(kimage, wrap_size, pixel_scale)
            n_k = kcoords.shape[0]

            # Pixel k-values.
            pixel_kvals = jax.vmap(
                lambda k: pixel._kValue(PositionD(k[0], k[1]))
            )(kcoords)

            # Match gsobject._adjust_offset: even-sized images need a -0.5 pixel
            # true-center correction before the FFT draw.  Apply it as a k-space
            # phase so chromatic draws align with monochromatic drawImage.
            img_shape = image.array.shape   # (ny, nx); unchanged by setCenter
            dx_corr = -0.5 * pixel_scale * ((img_shape[1] + 1) % 2)
            dy_corr = -0.5 * pixel_scale * ((img_shape[0] + 1) % 2)
            phase_corr = jnp.exp(
                -1j * (kcoords[:, 0] * dx_corr + kcoords[:, 1] * dy_corr)
            )

            if not nonsep_objs:
                # Pre-compute k-values of the full (spatial+pixel) convolution
                base_kvals = jax.vmap(
                    lambda k: grid_prof_conv._kValue(PositionD(k[0], k[1]))
                )(kcoords) * phase_corr
            else:
                pixel_kvals = pixel_kvals * phase_corr

                # Pre-compute k-values of separable components (unit flux each)
                sep_kvals = jnp.ones(n_k, dtype=complex)
                for o in sep_objs:
                    prof_sep = o._static_spatial_profile(wave_eff)
                    sep_kvals = sep_kvals * jax.vmap(
                        lambda k: prof_sep._kValue(PositionD(k[0], k[1]))
                    )(kcoords)

            kshape = kimage.array.shape
            kbounds = kimage.bounds
            kwcs = kimage.wcs

        # -------------------------------------------------------------------
        # Phase 2: traced computation (JAX-traced values allowed here)
        # -------------------------------------------------------------------
        if not nonsep_objs:
            # All-separable: integrate SED × bandpass → total flux, then scale
            def combined_weight(wave):
                w = bandpass(wave)
                for o in sep_objs:
                    w = w * o._sed_value(wave)
                return w

            total_flux = jnp.trapezoid(jax.vmap(combined_weight)(waves), waves)
            kvals = base_kvals * total_flux

        else:
            # Mixed sep + nonsep: integrate nonsep k-values weighted by sep SED × BP
            def sep_weight(wave):
                w = bandpass(wave)
                for o in sep_objs:
                    w = w * o._sed_value(wave)
                return w

            def kvals_nonsep_at_wave(wave):
                kv = jnp.ones(n_k, dtype=complex)
                for o in nonsep_objs:
                    prof = o.evaluateAtWavelength(wave)
                    kv = kv * jax.vmap(
                        lambda k: prof._kValue(PositionD(k[0], k[1]))
                    )(kcoords)
                return kv * sep_weight(wave)

            all_kvals = jax.vmap(kvals_nonsep_at_wave)(waves)   # (n_waves, n_k)
            eff_kvals = jnp.trapezoid(all_kvals, waves, axis=0)  # (n_k,)

            # Multiply by separable spatial k-values and pixel convolution
            kvals = eff_kvals * sep_kvals * pixel_kvals

        karray = kvals.reshape(kshape).astype(kimage.dtype)
        eff_kimage = Image(
            array=karray, bounds=kbounds, wcs=kwcs, _check_bounds=False
        )
        grid_prof_conv.drawFFT_finish(image, eff_kimage, wrap_size, add_to_image=False)

        image.shift(original_center)
        image.wcs = original_wcs
        return image

    def __repr__(self):
        inner = ", ".join(repr(o) for o in self.obj_list)
        return f"ChromaticConvolution([{inner}])"


# ---------------------------------------------------------------------------
# Monkey-patch GSObject.__mul__ to return Chromatic when multiplied by SED
# ---------------------------------------------------------------------------

def _gsobject_mul_sed(self, other):
    """Allow ``gsobject * sed → Chromatic(gsobject, sed)``."""
    from jax_galsim.sed import SED

    if isinstance(other, SED):
        return Chromatic(self, other)
    # Fall through to original implementation (flux scaling)
    return self.withScaledFlux(other)


def _gsobject_rmul_sed(self, other):
    return _gsobject_mul_sed(self, other)


# Apply the patch once at import time
def _patch_gsobject():
    from jax_galsim.gsobject import GSObject

    GSObject.__mul__ = _gsobject_mul_sed
    GSObject.__rmul__ = _gsobject_rmul_sed


_patch_gsobject()
