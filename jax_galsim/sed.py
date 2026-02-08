# Copyright (c) 2012-2023 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

from numbers import Real
from pathlib import PosixPath

import galsim as _galsim
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from jax_galsim.core.utils import ensure_hashable, implements
from jax_galsim.errors import (
    GalSimError,
    GalSimIncompatibleValuesError,
    GalSimRangeError,
    GalSimSEDError,
    GalSimValueError,
)
from jax_galsim.table import LookupTable, _LookupTable

# Physical constants in CGS/nm
_h = 6.62607015e-27  # erg s (Planck's constant)
_c = 2.99792458e17  # nm/s (speed of light)


@implements(
    _galsim.SED,
    lax_description="""\
The JAX-GalSim version of SED always uses fast=True semantics (pre-converts
units at initialization). The fast=False path relies on astropy unit
conversions at call-time which is incompatible with JAX tracing.
Only string wave_type ('nm', 'Angstrom') and string flux_type
('flambda', 'fnu', 'fphotons', '1') are supported. Astropy units are
not supported.
""",
)
@register_pytree_node_class
class SED:
    def __init__(
        self,
        spec,
        wave_type,
        flux_type,
        redshift=0.0,
        fast=True,
        interpolant="linear",
        _blue_limit=0.0,
        _red_limit=np.inf,
        _wave_list=None,
    ):
        self._flux_type = flux_type
        self.interpolant = interpolant

        # Parse wave_type
        self.wave_type, self.wave_factor = self._parse_wave_type(wave_type)

        # Parse flux_type
        self.flux_factor = None
        if isinstance(flux_type, str):
            if flux_type.lower() == "flambda":
                self.flux_type = "flambda"
                self.spectral = True
                self.flux_factor = 1.0 / (_h * _c)
            elif flux_type.lower() == "fphotons":
                self.spectral = True
                self.flux_type = "fphotons"
                self.flux_factor = self.wave_factor
            elif flux_type.lower() == "fnu":
                self.spectral = True
                self.flux_type = "fnu"
                self.flux_factor = self.wave_factor / _h
            elif flux_type == "1":
                self.flux_type = "1"
                self.spectral = False
            else:
                raise GalSimValueError(
                    "Unknown flux_type",
                    flux_type,
                    ("flambda", "fnu", "fphotons", "1"),
                )
        else:
            raise GalSimValueError(
                "Only string flux_type values are supported in jax_galsim.SED",
                flux_type,
            )

        self.redshift = redshift
        self.fast = (
            True  # Always fast in JAX-GalSim (fast param accepted for API compat)
        )

        # Convert string/file/number input into a callable/LookupTable
        self._orig_spec = spec
        self._initialize_spec()

        # Setup wave_list, blue_limit, red_limit
        if _wave_list is not None:
            self.wave_list = np.asarray(_wave_list, dtype=float)
            self.blue_limit = float(_blue_limit)
            self.red_limit = float(_red_limit)
        elif isinstance(self._spec, LookupTable):
            self.wave_list = np.asarray(self._spec.getArgs())
            self.wave_list = self.wave_list * (1.0 + self.redshift) / self.wave_factor
            self.blue_limit = float(np.min(self.wave_list))
            self.red_limit = float(np.max(self.wave_list))
        else:
            self.blue_limit = 0.0
            self.red_limit = np.inf
            self.wave_list = np.array([], dtype=float)

        # Build the fast spec (pre-converted units)
        self._build_fast_spec()

    @staticmethod
    def _parse_wave_type(wave_type):
        if isinstance(wave_type, str):
            if wave_type.lower() in ("nm", "nanometer", "nanometers"):
                return "nm", 1.0
            elif wave_type.lower() in ("a", "ang", "angstrom", "angstroms"):
                return "Angstrom", 10.0
            else:
                raise GalSimValueError(
                    "Unknown wave_type", wave_type, ("nm", "Angstrom")
                )
        else:
            raise GalSimValueError(
                "Only string wave_type values are supported in jax_galsim.SED",
                wave_type,
            )

    def _initialize_spec(self):
        self._const = False
        if isinstance(self._orig_spec, (int, float)):
            if not self.dimensionless:
                raise GalSimSEDError(
                    "Attempt to set spectral SED using float or integer.", self
                )
            self._const = True
            self._spec = float(self._orig_spec)
        elif isinstance(self._orig_spec, str):
            # Try as filename first
            import os

            if os.path.isfile(self._orig_spec):
                self._spec = LookupTable.from_file(
                    self._orig_spec, interpolant=self.interpolant
                )
            else:
                # Try as constant string
                try:
                    val = float(self._orig_spec)
                    self._const = True
                    self._spec = val
                    return
                except ValueError:
                    pass
                # Try as expression
                try:
                    func = eval("lambda wave : " + self._orig_spec)
                    test_value = func(700.0)
                except ArithmeticError:
                    test_value = 0
                except Exception as e:
                    raise GalSimValueError(
                        "String spec must either be a valid filename or something that "
                        "can eval to a function of wave.\n"
                        "Caught error: {0}".format(e),
                        self._orig_spec,
                    )
                if not isinstance(test_value, Real):
                    raise GalSimValueError(
                        "The given SED function did not return a valid number "
                        "at test wavelength %s: got %s" % (700.0, test_value),
                        self._orig_spec,
                    )
                self._spec = func
        elif isinstance(self._orig_spec, PosixPath):
            self._spec = LookupTable.from_file(
                str(self._orig_spec), interpolant=self.interpolant
            )
        elif isinstance(self._orig_spec, LookupTable):
            self._spec = self._orig_spec
        elif callable(self._orig_spec):
            self._spec = self._orig_spec
        else:
            self._spec = self._orig_spec

    def _build_fast_spec(self):
        """Build the fast spec that returns photons/nm/cm^2/s (spectral) or
        dimensionless value, given wavelength in nm in the rest frame."""
        if self._const:
            self._fast_spec = float(self._spec)
            return

        if self.wave_factor == 1.0 and self.flux_factor == 1.0:
            self._fast_spec = self._spec
            return

        if len(self.wave_list) > 0:
            x = self.wave_list / (1.0 + self.redshift)
            if self.spectral:
                f = self._rest_nm_to_photons(x)
            else:
                f = self._rest_nm_to_dimensionless(x)
            interp = (
                self._spec.interpolant
                if isinstance(self._spec, LookupTable)
                else "linear"
            )
            self._fast_spec = _LookupTable(x, f, interpolant=interp)
        else:
            # Analytic SED - store conversion as a function
            if self.spectral:
                self._fast_spec = self._rest_nm_to_photons
            else:
                self._fast_spec = self._rest_nm_to_dimensionless

    def _rest_nm_to_photons(self, wave):
        """Convert from rest-frame nm wavelength to photons/nm/cm^2/s."""
        wave_native = np.asarray(wave) * self.wave_factor
        flux_native = self._spec(wave_native) if callable(self._spec) else self._spec
        return self._flux_to_photons(flux_native, wave_native)

    def _rest_nm_to_dimensionless(self, wave):
        """Convert from rest-frame nm wavelength to dimensionless value."""
        wave_native = np.asarray(wave) * self.wave_factor
        return self._spec(wave_native) if callable(self._spec) else self._spec

    def _flux_to_photons(self, flux_native, wave_native):
        if self.flux_type == "fphotons":
            return flux_native * self.flux_factor
        elif self.flux_type == "flambda":
            return flux_native * wave_native * self.flux_factor
        elif self.flux_type == "fnu":
            return flux_native / wave_native * self.flux_factor
        else:
            return flux_native

    @property
    def dimensionless(self):
        return not self.spectral

    def _check_bounds(self, wave):
        if hasattr(wave, "__iter__"):
            wmin = np.min(wave)
            wmax = np.max(wave)
        else:
            wmin = wmax = wave

        extrapolation_slop = 1.0e-6
        if wmin < self.blue_limit - extrapolation_slop:
            raise GalSimRangeError(
                "Requested wavelength is bluer than blue_limit.",
                wave,
                self.blue_limit,
                self.red_limit,
            )
        if wmax > self.red_limit + extrapolation_slop:
            raise GalSimRangeError(
                "Requested wavelength is redder than red_limit.",
                wave,
                self.blue_limit,
                self.red_limit,
            )

    def __call__(self, wave):
        wave = jnp.asarray(wave, dtype=float)
        self._check_bounds(wave)

        if self._const:
            return jnp.full_like(wave, self._fast_spec, dtype=float)

        rest_wave = wave / (1.0 + self.redshift)
        if isinstance(self._fast_spec, LookupTable):
            return self._fast_spec(rest_wave)
        elif callable(self._fast_spec):
            return jnp.asarray(self._fast_spec(rest_wave), dtype=float)
        else:
            return jnp.full_like(wave, float(self._fast_spec), dtype=float)

    def atRedshift(self, redshift):
        if redshift == self.redshift:
            return self
        if redshift <= -1:
            raise GalSimRangeError("Invalid redshift", redshift, -1.0)
        zfactor = (1.0 + redshift) / (1.0 + self.redshift)
        wave_list = self.wave_list * zfactor
        blue_limit = self.blue_limit * zfactor
        red_limit = self.red_limit * zfactor
        return SED(
            self._spec,
            self.wave_type,
            self.flux_type,
            redshift,
            self.fast,
            _wave_list=wave_list,
            _blue_limit=blue_limit,
            _red_limit=red_limit,
        )

    def calculateFlux(self, bandpass):
        if self.dimensionless:
            raise GalSimSEDError("Cannot calculate flux of dimensionless SED.", self)
        if bandpass is None:
            from .bandpass import Bandpass

            bandpass = Bandpass(lambda _: 1.0, "nm", blue_limit=0.0, red_limit=1e30)

        if len(bandpass.wave_list) > 0 or len(self.wave_list) > 0:
            from . import utilities

            slop = 1e-6
            if (
                self.blue_limit > bandpass.blue_limit + slop
                or self.red_limit < bandpass.red_limit - slop
            ):
                raise GalSimRangeError(
                    "Bandpass is not completely within defined wavelength "
                    "range for this SED.",
                    (bandpass.blue_limit, bandpass.red_limit),
                    self.blue_limit,
                    self.red_limit,
                )
            wmin = max(self.blue_limit, bandpass.blue_limit)
            wmax = min(self.red_limit, bandpass.red_limit)

            if isinstance(self._fast_spec, LookupTable):
                wf = 1.0 / (1.0 + self.redshift) / bandpass.wave_factor
                ff = 1.0 / bandpass.wave_factor
                _wmin = wmin * bandpass.wave_factor
                _wmax = wmax * bandpass.wave_factor
                return (
                    self._fast_spec.integrate_product(bandpass._tp, _wmin, _wmax, wf)
                    * ff
                )
            else:
                wave, _, _ = utilities.combine_wave_list(self, bandpass)
                interpolant = (
                    bandpass._tp.interpolant
                    if hasattr(bandpass._tp, "interpolant")
                    else "linear"
                )
                return _LookupTable(
                    wave, bandpass(wave), interpolant
                ).integrate_product(self)
        else:
            from jax_galsim.integ import int1d

            return int1d(
                lambda w: bandpass(w) * self(w),
                bandpass.blue_limit,
                bandpass.red_limit,
            )

    def calculateMagnitude(self, bandpass):
        if self.dimensionless:
            raise GalSimSEDError(
                "Cannot calculate magnitude of dimensionless SED.", self
            )
        if bandpass.zeropoint is None:
            raise GalSimError(
                "Cannot do this calculation for a bandpass without an assigned zeropoint"
            )
        flux = self.calculateFlux(bandpass)
        return -2.5 * np.log10(flux) + bandpass.zeropoint

    def withFlux(self, target_flux, bandpass):
        current_flux = self.calculateFlux(bandpass)
        norm = target_flux / current_flux
        return self * norm

    def withFluxDensity(self, target_flux_density, wavelength):
        if self.dimensionless:
            raise GalSimSEDError("Cannot set flux density of dimensionless SED.", self)
        current_flux_density = self(wavelength)
        factor = target_flux_density / current_flux_density
        return self * factor

    def withMagnitude(self, target_magnitude, bandpass):
        if bandpass.zeropoint is None:
            raise GalSimError(
                "Cannot call SED.withMagnitude on this bandpass, because it does "
                "not have a zeropoint.  See Bandpass.withZeropoint()"
            )
        current_magnitude = self.calculateMagnitude(bandpass)
        norm = 10 ** (-0.4 * (target_magnitude - current_magnitude))
        return self * norm

    def _mul_sed(self, other):
        redshift = self.redshift + other.redshift
        from . import utilities

        wave_list, blue_limit, red_limit = utilities.combine_wave_list(self, other)
        if (
            isinstance(self._fast_spec, LookupTable)
            and not self._fast_spec.x_log
            and not self._fast_spec.f_log
        ):
            x = wave_list / (1.0 + self.redshift)
            x = utilities.merge_sorted([x, np.linspace(x[0], x[-1], 500)])
            zfactor2 = (1.0 + redshift) / (1.0 + other.redshift)
            f = self._fast_spec(x) * other._fast_spec(x * zfactor2)
            spec = _LookupTable(x, f, self._fast_spec.interpolant)
        elif (
            isinstance(other._fast_spec, LookupTable)
            and not other._fast_spec.x_log
            and not other._fast_spec.f_log
        ):
            x = wave_list / (1.0 + other.redshift)
            x = utilities.merge_sorted([x, np.linspace(x[0], x[-1], 500)])
            zfactor1 = (1.0 + redshift) / (1.0 + self.redshift)
            f = self._fast_spec(x * zfactor1) * other._fast_spec(x)
            spec = _LookupTable(x, f, other._fast_spec.interpolant)
        else:
            zfactor1 = (1.0 + redshift) / (1.0 + self.redshift)
            zfactor2 = (1.0 + redshift) / (1.0 + other.redshift)

            def spec(w):
                return self._fast_spec(w * zfactor1) * other._fast_spec(w * zfactor2)

        spectral = self.spectral or other.spectral
        flux_type = "fphotons" if spectral else "1"
        return SED(
            spec,
            "nm",
            flux_type,
            redshift=redshift,
            fast=True,
            _blue_limit=blue_limit,
            _red_limit=red_limit,
            _wave_list=wave_list,
        )

    def _mul_bandpass(self, other):
        from . import utilities

        wave_list, blue_limit, red_limit = utilities.combine_wave_list(self, other)
        zfactor = (1.0 + self.redshift) / other.wave_factor
        if (
            isinstance(self._fast_spec, LookupTable)
            and not self._fast_spec.x_log
            and not self._fast_spec.f_log
        ):
            x = wave_list / (1.0 + self.redshift)
            x = utilities.merge_sorted([x, np.linspace(x[0], x[-1], 500)])
            f = self._fast_spec(x) * other._tp(x * zfactor)
            spec = _LookupTable(x, f, self._fast_spec.interpolant)
        else:
            spec = lambda w: self._fast_spec(w) * other._tp(w * zfactor)  # noqa: E731
        return SED(
            spec,
            "nm",
            "fphotons",
            redshift=self.redshift,
            fast=True,
            _blue_limit=blue_limit,
            _red_limit=red_limit,
            _wave_list=wave_list,
        )

    def _mul_scalar(self, other, spectral):
        if isinstance(self._spec, LookupTable):
            wave_type = self.wave_type
            flux_type = self._flux_type
            x = self._spec.getArgs()
            f = jnp.asarray(self._spec.getVals()) * other
            spec = _LookupTable(
                x,
                f,
                x_log=self._spec.x_log,
                f_log=self._spec.f_log,
                interpolant=self._spec.interpolant,
            )
        elif self._const and not spectral:
            spec = float(self._spec) * other
            wave_type = "nm"
            flux_type = "1"
        else:
            wave_type = "nm"
            flux_type = "fphotons" if spectral else "1"
            spec = lambda w: self._fast_spec(w) * other  # noqa: E731
        return SED(
            spec,
            wave_type,
            flux_type,
            redshift=self.redshift,
            fast=True,
            _blue_limit=self.blue_limit,
            _red_limit=self.red_limit,
            _wave_list=self.wave_list,
        )

    def __mul__(self, other):
        # SED * SED
        if isinstance(other, SED):
            if self.spectral and other.spectral:
                raise GalSimIncompatibleValuesError(
                    "Cannot multiply two spectral densities together.",
                    self_sed=self,
                    other=other,
                )
            if other._const:
                return self._mul_scalar(
                    float(other._spec), self.spectral or other.spectral
                )
            elif self._const:
                return other._mul_scalar(
                    float(self._spec), self.spectral or other.spectral
                )
            else:
                return self._mul_sed(other)

        # SED * GSObject -> ChromaticObject
        from jax_galsim.gsobject import GSObject

        if isinstance(other, GSObject):
            return other * self

        # SED * Bandpass -> SED

        if isinstance(other, Bandpass):
            return self._mul_bandpass(other)

        # SED * callable -> SED
        if hasattr(other, "__call__"):
            spec = lambda w: self._fast_spec(w) * other(w * (1.0 + self.redshift))  # noqa: E731
            flux_type = "fphotons" if self.spectral else "1"
            return SED(
                spec,
                "nm",
                flux_type,
                redshift=self.redshift,
                fast=True,
                _blue_limit=self.blue_limit,
                _red_limit=self.red_limit,
                _wave_list=self.wave_list,
            )

        # SED * scalar (including JAX scalars)
        if isinstance(other, (int, float)):
            return self._mul_scalar(other, self.spectral)

        # Handle JAX/numpy scalar arrays
        try:
            return self._mul_scalar(float(other), self.spectral)
        except (TypeError, ValueError):
            pass

        raise TypeError("Cannot multiply an SED by %s" % (other))

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        if isinstance(other, SED) and other.spectral:
            raise GalSimSEDError("Cannot divide by spectral SED.", other)
        if hasattr(other, "__call__"):

            def spec(w):
                return self(w * (1.0 + self.redshift)) / other(
                    w * (1.0 + self.redshift)
                )

        elif isinstance(self._spec, LookupTable):
            x = self._spec.getArgs()
            f = jnp.asarray(self._spec.getVals()) / other
            spec = _LookupTable(
                x,
                f,
                x_log=self._spec.x_log,
                f_log=self._spec.f_log,
                interpolant=self._spec.interpolant,
            )
        else:
            spec = lambda w: self(w * (1.0 + self.redshift)) / other  # noqa: E731

        return SED(
            spec,
            flux_type=self.flux_type,
            wave_type=self.wave_type,
            redshift=self.redshift,
            fast=True,
            _wave_list=self.wave_list,
            _blue_limit=self.blue_limit,
            _red_limit=self.red_limit,
        )

    __truediv__ = __div__

    def __add__(self, other):
        if self.redshift != other.redshift:
            raise GalSimIncompatibleValuesError(
                "Can only add SEDs with same redshift.",
                self_sed=self,
                other=other,
            )

        if self.dimensionless and other.dimensionless:
            flux_type = "1"
        elif self.spectral and other.spectral:
            flux_type = "fphotons"
        else:
            raise GalSimIncompatibleValuesError(
                "Cannot add SEDs with incompatible dimensions.",
                self_sed=self,
                other=other,
            )

        from . import utilities

        wave_list, blue_limit, red_limit = utilities.combine_wave_list(self, other)

        # If both fast specs are LookupTables with linear interpolation, merge them
        if (
            isinstance(self._fast_spec, LookupTable)
            and isinstance(other._fast_spec, LookupTable)
            and not self._fast_spec.x_log
            and not other._fast_spec.x_log
            and not self._fast_spec.f_log
            and not other._fast_spec.f_log
            and self._fast_spec.interpolant == "linear"
            and other._fast_spec.interpolant == "linear"
        ):
            x = wave_list / (1.0 + self.redshift)
            f = self._fast_spec(x) + other._fast_spec(x)
            spec = _LookupTable(x, f, interpolant="linear")
        else:

            def spec(w):
                return self(w * (1.0 + self.redshift)) + other(
                    w * (1.0 + self.redshift)
                )

        return SED(
            spec,
            wave_type="nm",
            flux_type=flux_type,
            redshift=self.redshift,
            fast=True,
            _wave_list=wave_list,
            _blue_limit=blue_limit,
            _red_limit=red_limit,
        )

    def __sub__(self, other):
        return self.__add__(-1.0 * other)

    def __eq__(self, other):
        return self is other or (
            isinstance(other, SED)
            and self._orig_spec == other._orig_spec
            and self.fast == other.fast
            and self.wave_type == other.wave_type
            and self.flux_type == other.flux_type
            and self.redshift == other.redshift
            and self.red_limit == other.red_limit
            and self.blue_limit == other.blue_limit
            and np.array_equal(self.wave_list, other.wave_list)
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        if not hasattr(self, "_hash"):
            self._hash = hash(
                (
                    "galsim.SED",
                    ensure_hashable(self._orig_spec)
                    if not isinstance(self._orig_spec, LookupTable)
                    else self._orig_spec,
                    self.wave_type,
                    self.flux_type,
                    self.redshift,
                    self.fast,
                    self.blue_limit,
                    self.red_limit,
                    tuple(self.wave_list),
                )
            )
        return self._hash

    def __repr__(self):
        outstr = (
            "galsim.SED(%r, wave_type=%r, flux_type=%r, redshift=%r, fast=%r, "
            "interpolant=%r, _wave_list=%r, _blue_limit=%r, _red_limit=%s)"
        ) % (
            self._orig_spec,
            self.wave_type,
            self._flux_type,
            self.redshift,
            self.fast,
            self.interpolant,
            self.wave_list,
            self.blue_limit,
            "float('inf')" if self.red_limit == np.inf else repr(self.red_limit),
        )
        return outstr

    def __str__(self):
        orig_spec = repr(self._orig_spec)
        if len(orig_spec) > 80:
            orig_spec = str(self._orig_spec)
        return "galsim.SED(%s, redshift=%s)" % (orig_spec, self.redshift)

    def tree_flatten(self):
        # The fast_spec is either a LookupTable (pytree) or a scalar
        if isinstance(self._fast_spec, LookupTable):
            children = (self._fast_spec, jnp.asarray(self.redshift, dtype=float))
        elif isinstance(self._fast_spec, (int, float)):
            children = (
                jnp.asarray(self._fast_spec, dtype=float),
                jnp.asarray(self.redshift, dtype=float),
            )
        else:
            # Callable fast_spec - not traceable, store as aux
            children = (jnp.asarray(self.redshift, dtype=float),)

        aux_data = {
            "wave_type": self.wave_type,
            "flux_type": self.flux_type,
            "spectral": self.spectral,
            "wave_factor": self.wave_factor,
            "flux_factor": self.flux_factor,
            "blue_limit": self.blue_limit,
            "red_limit": self.red_limit,
            "wave_list": self.wave_list,
            "interpolant": self.interpolant,
            "_const": self._const,
            "fast": self.fast,
            "_orig_spec": self._orig_spec,
            "_flux_type": self._flux_type,
            "_spec": self._spec,
            "_fast_spec_is_lt": isinstance(self._fast_spec, LookupTable),
            "_fast_spec_is_scalar": isinstance(self._fast_spec, (int, float)),
            "_fast_spec_callable": (
                self._fast_spec
                if not isinstance(self._fast_spec, (LookupTable, int, float))
                else None
            ),
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.wave_type = aux_data["wave_type"]
        obj.flux_type = aux_data["flux_type"]
        obj._flux_type = aux_data["_flux_type"]
        obj.spectral = aux_data["spectral"]
        obj.wave_factor = aux_data["wave_factor"]
        obj.flux_factor = aux_data["flux_factor"]
        obj.blue_limit = aux_data["blue_limit"]
        obj.red_limit = aux_data["red_limit"]
        obj.wave_list = aux_data["wave_list"]
        obj.interpolant = aux_data["interpolant"]
        obj._const = aux_data["_const"]
        obj.fast = aux_data["fast"]
        obj._orig_spec = aux_data["_orig_spec"]
        obj._spec = aux_data["_spec"]

        if aux_data["_fast_spec_is_lt"]:
            obj._fast_spec = children[0]
            obj.redshift = children[1]
        elif aux_data["_fast_spec_is_scalar"]:
            obj._fast_spec = float(children[0])
            obj.redshift = float(children[1])
        else:
            obj._fast_spec = aux_data["_fast_spec_callable"]
            obj.redshift = float(children[0])

        return obj


# Put this at the bottom to avoid circular import errors.
from .bandpass import Bandpass  # noqa: E402, F401
