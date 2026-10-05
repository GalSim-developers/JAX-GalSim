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
    ├── SimpleChromaticTransformation  GSObject × SED  (separable, fast path)
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

import galsim as _galsim
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from jax_galsim.box import Pixel
from jax_galsim.core.utils import cast_to_float, implements
from jax_galsim.gaussian import Gaussian
from jax_galsim.gsobject import GSObject
from jax_galsim.gsparams import GSParams
from jax_galsim.image import Image
from jax_galsim.moffat import Moffat
from jax_galsim.position import PositionD
from jax_galsim.sed import SED
from jax_galsim.sum import Sum


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


def _setup_wavelength_from_bandpass(bandpass):
    """Return compile-time wavelength for FFT sizing, independent of throughput."""
    with jax.ensure_compile_time_eval():
        return float(cast_to_float(0.5 * (bandpass.blue_limit + bandpass.red_limit)))


def _make_setup_image(profile, kwargs):
    """Build draw target without running full ``drawImage`` setup when size is fixed."""
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


def _next_pow2(number):
    number = int(number)
    power = 1
    while power < number:
        power *= 2
    return power


def _fix_fft_size_for_image(profile, image):
    """Pin FFT size for JIT-safe setup when output bounds are already fixed."""
    row_count, column_count = image.array.shape
    fft_size = max(128, _next_pow2(2 * max(row_count, column_count)))
    fft_size = min(fft_size, profile.gsparams.maximum_fft_size)
    return profile.withGSParams(minimum_fft_size=fft_size, maximum_fft_size=fft_size)


def _static_kcoords(kimage, wrap_size, pixel_scale):
    """Return k-space pixel centers as a JAX array with static shape."""
    row_count, column_count = kimage.array.shape
    x_index = jnp.arange(column_count, dtype=float)
    y_index = jnp.arange(row_count, dtype=float) - wrap_size // 2
    kx_grid, ky_grid = jnp.meshgrid(x_index, y_index)
    delta_k = 2.0 * jnp.pi / (wrap_size * pixel_scale)
    return jnp.stack([kx_grid.ravel(), ky_grid.ravel()], axis=-1) * delta_k


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


@implements(
    _galsim.ChromaticObject,
    lax_description="""\
JAX-GalSim implements the differentiable FFT drawing subset.  Chromatic
transformations, interpolation, photon shooting, and full SED unit semantics
are not implemented.
""",
)
class ChromaticObject:
    #: Set True in separable subclasses.
    _separable: bool = False

    def __init__(self, obj=None):
        if obj is None:
            self._base_obj = None
            return

        if not isinstance(obj, GSObject):
            raise TypeError("ChromaticObject requires a GSObject.")
        self._base_obj = obj
        self._separable = True

    @property
    @implements(
        getattr(_galsim.ChromaticObject, "separable", None),
        lax_description="JAX-GalSim-specific separability flag for chromatic drawing.",
    )
    def separable(self):
        return self._separable

    @property
    @implements(_galsim.ChromaticObject.gsparams)
    def gsparams(self):
        base_obj = getattr(self, "_base_obj", None)
        if base_obj is not None:
            return base_obj.gsparams
        return GSParams.default

    # ------------------------------------------------------------------
    # Interface that subclasses must implement
    # ------------------------------------------------------------------

    @implements(_galsim.ChromaticObject.evaluateAtWavelength)
    def evaluateAtWavelength(self, wave):
        if getattr(self, "_base_obj", None) is not None:
            return self._base_obj
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement evaluateAtWavelength."
        )

    @implements(_galsim.ChromaticObject.withGSParams)
    def withGSParams(self, gsparams=None, **kwargs):
        base_obj = getattr(self, "_base_obj", None)
        if base_obj is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement withGSParams."
            )
        gsparams = GSParams.check(gsparams, base_obj.gsparams, **kwargs)
        return ChromaticObject(base_obj.withGSParams(gsparams))

    @implements(_galsim.ChromaticObject.calculateFlux)
    def calculateFlux(self, bandpass, n_waves=512):
        waves = jnp.linspace(bandpass.blue_limit, bandpass.red_limit, n_waves)
        fluxes = jax.vmap(
            lambda wave: self.evaluateAtWavelength(wave).flux * bandpass(wave)
        )(waves)
        return jnp.trapezoid(fluxes, waves)

    @implements(_galsim.ChromaticObject.atRedshift)
    def atRedshift(self, redshift):
        return self

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    @implements(
        _galsim.ChromaticObject.drawImage,
        lax_description="JAX-GalSim supports FFT drawing through array-backed Bandpass objects.",
    )
    def drawImage(self, bandpass, n_waves=64, **kwargs):
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
        scalars at this stage (true for ``SimpleChromaticTransformation`` where the spatial
        profile has static params); the SED flux is NOT evaluated here.

        **Phase 2 — traced computation**:
        Compute total_flux = ∫ SED(λ)×BP(λ) dλ (may be a JAX traced
        value, e.g. DSPS output), multiply into k-values, IFFT.
        """
        # Keep this local to avoid a circular import with jax_galsim.convolve.
        from jax_galsim.convolve import Convolve

        wave_eff = _setup_wavelength_from_bandpass(bandpass)
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
            kvals_static = jax.vmap(lambda k: prof_conv._kValue(PositionD(k[0], k[1])))(
                kcoords
            )

            # Apply the same -0.5 pixel centering correction that gsobject._adjust_offset
            # uses for even-sized images (avoids 0.5-pixel offset vs non-chromatic draws).
            img_shape = image.array.shape  # (ny, nx); unchanged by setCenter
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
        eff_kimage = Image(array=karray, bounds=kbounds, wcs=kwcs, _check_bounds=False)
        prof_conv.drawFFT_finish(image, eff_kimage, wrap_size, add_to_image=False)

        image.shift(original_center)
        image.wcs = original_wcs
        return image

    def _static_spatial_profile(self, wave_eff):
        """Return unit-flux spatial profile at *wave_eff* with static params.

        *wave_eff* must be a concrete Python float (use
        a compile-time wavelength).  Subclasses override this when
        they have a dedicated static spatial object (e.g. ``SimpleChromaticTransformation`` has
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
        # Keep this local to avoid a circular import with jax_galsim.convolve.
        from jax_galsim.convolve import Convolve

        wave_eff = _setup_wavelength_from_bandpass(bandpass)
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

            pixel_kvals = jax.vmap(lambda k: pixel._kValue(PositionD(k[0], k[1])))(
                kcoords
            )

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
            kv = jax.vmap(lambda k: prof._kValue(PositionD(k[0], k[1])))(kcoords)
            return kv * bandpass(wave)

        all_kvals = jax.vmap(kvals_at_wave)(waves)  # (n_waves, n_k)
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

    @implements(_galsim.ChromaticObject.__add__)
    def __add__(self, other):
        return ChromaticSum([self, other])

    def __radd__(self, other):
        return ChromaticSum([other, self])

    @implements(_galsim.ChromaticObject.__mul__)
    def __mul__(self, other):
        if isinstance(other, SED):
            base_obj = getattr(self, "_base_obj", None)
            if base_obj is None:
                raise TypeError(
                    "Only achromatic ChromaticObject wrappers can be multiplied by SED."
                )
            return SimpleChromaticTransformation(base_obj, other)
        return _ScaledChromaticObject(self, other)

    @implements(_galsim.ChromaticObject.__rmul__)
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


@implements(
    _galsim.ChromaticSum,
    lax_description="JAX-GalSim treats ChromaticSum as non-separable, matching GalSim semantics.",
)
class ChromaticSum(ChromaticObject):
    _separable = False

    def __init__(self, *args):
        if len(args) == 0:
            raise TypeError("At least one object must be provided.")
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            self.obj_list = list(args[0])
        else:
            self.obj_list = list(args)
        self._separable = False

    @property
    @implements(_galsim.ChromaticSum.gsparams)
    def gsparams(self):
        return GSParams.combine([obj.gsparams for obj in self.obj_list])

    @implements(_galsim.ChromaticSum.withGSParams)
    def withGSParams(self, gsparams=None, **kwargs):
        gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        return ChromaticSum([obj.withGSParams(gsparams) for obj in self.obj_list])

    @implements(_galsim.ChromaticSum.atRedshift)
    def atRedshift(self, redshift):
        return ChromaticSum([obj.atRedshift(redshift) for obj in self.obj_list])

    @implements(_galsim.ChromaticSum.evaluateAtWavelength)
    def evaluateAtWavelength(self, wave):
        return Sum([o.evaluateAtWavelength(wave) for o in self.obj_list])

    @implements(
        _galsim.ChromaticSum.drawImage,
        lax_description=(
            "JAX-GalSim draws sums component-by-component to preserve non-separable semantics."
        ),
    )
    def drawImage(self, bandpass, n_waves=64, **kwargs):
        # Draw each component and sum
        images = [
            obj.drawImage(bandpass, n_waves=n_waves, **kwargs) for obj in self.obj_list
        ]
        result = images[0]
        for img in images[1:]:
            result._array = result._array + img._array
        return result


# ---------------------------------------------------------------------------
# SimpleChromaticTransformation — separable GSObject × SED
# ---------------------------------------------------------------------------


@implements(
    _galsim.SimpleChromaticTransformation,
    lax_description="""\
JAX-GalSim implements the simple separable case GSObject * SED.  General
chromatic affine transformations are not implemented.
""",
)
@register_pytree_node_class
class SimpleChromaticTransformation(ChromaticObject):
    _separable = True

    def __init__(self, obj, sed):
        self.obj = obj
        self.sed = sed

    @property
    @implements(_galsim.SimpleChromaticTransformation.gsparams)
    def gsparams(self):
        return self.obj.gsparams

    @implements(_galsim.SimpleChromaticTransformation.withGSParams)
    def withGSParams(self, gsparams=None, **kwargs):
        gsparams = GSParams.check(gsparams, self.obj.gsparams, **kwargs)
        return SimpleChromaticTransformation(self.obj.withGSParams(gsparams), self.sed)

    @implements(_galsim.SimpleChromaticTransformation.atRedshift)
    def atRedshift(self, redshift):
        return SimpleChromaticTransformation(self.obj, self.sed.atRedshift(redshift))

    @implements(_galsim.SimpleChromaticTransformation.evaluateAtWavelength)
    def evaluateAtWavelength(self, wave):
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
        return f"galsim.SimpleChromaticTransformation({self.obj!r}, sed={self.sed!r})"


# Backwards-compatible internal alias.  Not exported at top level because upstream
# GalSim exposes SimpleChromaticTransformation, not Chromatic.
Chromatic = SimpleChromaticTransformation


# ---------------------------------------------------------------------------
# ChromaticAtmosphere — seeing PSF with wavelength-dependent FWHM
# ---------------------------------------------------------------------------


@register_pytree_node_class
@implements(
    _galsim.ChromaticAtmosphere,
    lax_description="""\
JAX-GalSim implements a differentiable Gaussian/Moffat seeing PSF with
FWHM(lambda) = fwhm_ref * (lambda / lam_ref)**alpha.  Differential
chromatic refraction and atmospheric coordinate parameters are not
implemented.
""",
)
class ChromaticAtmosphere(ChromaticObject):
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
    @implements(
        None,
        lax_description="JAX-GalSim-specific reference FWHM for ChromaticAtmosphere.",
    )
    def fwhm_ref(self):
        return self._fwhm_ref

    @property
    @implements(
        None,
        lax_description="JAX-GalSim-specific reference wavelength for ChromaticAtmosphere.",
    )
    def lam_ref(self):
        return self._lam_ref

    @property
    @implements(
        None,
        lax_description="JAX-GalSim-specific wavelength power-law index for ChromaticAtmosphere.",
    )
    def alpha(self):
        return self._alpha

    @property
    @implements(_galsim.ChromaticAtmosphere.gsparams)
    def gsparams(self):
        return self._gsparams

    @implements(_galsim.ChromaticAtmosphere.withGSParams)
    def withGSParams(self, gsparams=None, **kwargs):
        gsparams = GSParams.check(gsparams, self._gsparams, **kwargs)
        return ChromaticAtmosphere(
            fwhm_ref=self._fwhm_ref,
            lam_ref=self._lam_ref,
            alpha=self._alpha,
            profile=self._profile,
            moffat_beta=self._moffat_beta,
            gsparams=gsparams,
        )

    @implements(_galsim.ChromaticAtmosphere.atRedshift)
    def atRedshift(self, redshift):
        return self

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    @implements(_galsim.ChromaticAtmosphere.evaluateAtWavelength)
    def evaluateAtWavelength(self, wave):
        fwhm = self._fwhm_ref * (wave / self._lam_ref) ** self._alpha

        if self._profile == "gaussian":
            return Gaussian(fwhm=fwhm, flux=1.0, gsparams=self._gsparams)

        elif self._profile == "moffat":
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


@implements(
    _galsim.ChromaticConvolution,
    lax_description="""\
JAX-GalSim supports FFT drawing of chromatic convolutions.  Real-space
chromatic convolution, photon shooting, and full GalSim chromatic
transformations are not implemented.
""",
)
class ChromaticConvolution(ChromaticObject):
    _separable = False

    def __init__(self, *args, **kwargs):
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
            raise NotImplementedError(
                "Real-space chromatic convolutions are not implemented"
            )

        # Wrap plain GSObjects with a flat SED so they fit the interface
        processed = []
        for obj in obj_list:
            if isinstance(obj, ChromaticConvolution):
                processed.extend(obj.obj_list)
                continue
            if isinstance(obj, GSObject):
                wave_stub = jnp.array([100.0, 2000.0])
                flat_sed = SED(wave_stub, jnp.ones(2))
                processed.append(SimpleChromaticTransformation(obj, flat_sed))
            else:
                processed.append(obj)
        self.obj_list = processed

    @property
    @implements(_galsim.ChromaticConvolution.gsparams)
    def gsparams(self):
        return self._gsparams

    @implements(_galsim.ChromaticConvolution.withGSParams)
    def withGSParams(self, gsparams=None, **kwargs):
        ret = self.__class__.__new__(self.__class__)
        ret.obj_list = self.obj_list
        ret._gsparams = GSParams.check(gsparams, self._gsparams, **kwargs)
        ret._propagate_gsparams = self._propagate_gsparams
        return ret

    @implements(_galsim.ChromaticConvolution.atRedshift)
    def atRedshift(self, redshift):
        return ChromaticConvolution(
            [obj.atRedshift(redshift) for obj in self.obj_list],
            gsparams=self._gsparams,
            propagate_gsparams=self._propagate_gsparams,
        )

    @implements(_galsim.ChromaticConvolution.evaluateAtWavelength)
    def evaluateAtWavelength(self, wave):
        # Keep this local to avoid a circular import with jax_galsim.convolve.
        from jax_galsim.convolve import Convolve

        return Convolve([o.evaluateAtWavelength(wave) for o in self.obj_list])

    # ------------------------------------------------------------------
    # Drawing with the optimised separable split
    # ------------------------------------------------------------------

    @implements(
        _galsim.ChromaticConvolution.drawImage,
        lax_description="JAX-GalSim supports FFT drawing with static wavelength grid size.",
    )
    def drawImage(self, bandpass, n_waves=64, **kwargs):
        # Keep this local to avoid a circular import with jax_galsim.convolve.
        from jax_galsim.convolve import Convolve

        sep_objs = [o for o in self.obj_list if o._separable]
        nonsep_objs = [o for o in self.obj_list if not o._separable]

        wave_eff = _setup_wavelength_from_bandpass(bandpass)
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
                    o._static_spatial_profile(wave_eff)
                    if o._separable
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
            pixel_kvals = jax.vmap(lambda k: pixel._kValue(PositionD(k[0], k[1])))(
                kcoords
            )

            # Match gsobject._adjust_offset: even-sized images need a -0.5 pixel
            # true-center correction before the FFT draw.  Apply it as a k-space
            # phase so chromatic draws align with monochromatic drawImage.
            img_shape = image.array.shape  # (ny, nx); unchanged by setCenter
            dx_corr = -0.5 * pixel_scale * ((img_shape[1] + 1) % 2)
            dy_corr = -0.5 * pixel_scale * ((img_shape[0] + 1) % 2)
            phase_corr = jnp.exp(
                -1j * (kcoords[:, 0] * dx_corr + kcoords[:, 1] * dy_corr)
            )

            if not nonsep_objs:
                # Pre-compute k-values of the full (spatial+pixel) convolution
                base_kvals = (
                    jax.vmap(lambda k: grid_prof_conv._kValue(PositionD(k[0], k[1])))(
                        kcoords
                    )
                    * phase_corr
                )
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
                    kv = kv * jax.vmap(lambda k: prof._kValue(PositionD(k[0], k[1])))(
                        kcoords
                    )
                return kv * sep_weight(wave)

            all_kvals = jax.vmap(kvals_nonsep_at_wave)(waves)  # (n_waves, n_k)
            eff_kvals = jnp.trapezoid(all_kvals, waves, axis=0)  # (n_k,)

            # Multiply by separable spatial k-values and pixel convolution
            kvals = eff_kvals * sep_kvals * pixel_kvals

        karray = kvals.reshape(kshape).astype(kimage.dtype)
        eff_kimage = Image(array=karray, bounds=kbounds, wcs=kwcs, _check_bounds=False)
        grid_prof_conv.drawFFT_finish(image, eff_kimage, wrap_size, add_to_image=False)

        image.shift(original_center)
        image.wcs = original_wcs
        return image

    def __repr__(self):
        inner = ", ".join(repr(o) for o in self.obj_list)
        return f"ChromaticConvolution([{inner}])"


# ---------------------------------------------------------------------------
# Monkey-patch GSObject.__mul__ to return a simple chromatic object for SEDs
# ---------------------------------------------------------------------------


@implements(
    _galsim.GSObject.__mul__,
    lax_description="Also accepts array-backed JAX-GalSim SED objects and returns a SimpleChromaticTransformation.",
)
def _gsobject_mul_sed(self, other):
    if isinstance(other, SED):
        return SimpleChromaticTransformation(self, other)
    # Fall through to original implementation (flux scaling)
    return self.withScaledFlux(other)


@implements(
    _galsim.GSObject.__rmul__,
    lax_description="Also accepts array-backed JAX-GalSim SED objects and returns a SimpleChromaticTransformation.",
)
def _gsobject_rmul_sed(self, other):
    return _gsobject_mul_sed(self, other)


# Apply the patch once at import time
def _patch_gsobject():
    GSObject.__mul__ = _gsobject_mul_sed
    GSObject.__rmul__ = _gsobject_rmul_sed


_patch_gsobject()
