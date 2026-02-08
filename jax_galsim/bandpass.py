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
    GalSimIncompatibleValuesError,
    GalSimValueError,
)
from jax_galsim.table import LookupTable, _LookupTable


@implements(
    _galsim.Bandpass,
    lax_description="""\
The JAX-GalSim version of Bandpass. Only string wave_type ('nm', 'Angstrom')
is supported. Astropy units are not supported.
The withZeropoint method only supports float values and 'AB'/'ST' strings.
'Vega' is not supported.
""",
)
@register_pytree_node_class
class Bandpass:
    def __init__(
        self,
        throughput,
        wave_type,
        blue_limit=None,
        red_limit=None,
        zeropoint=None,
        interpolant="linear",
        _wave_list=None,
        _tp=None,
    ):
        self._orig_tp = throughput
        self._tp = _tp
        self.zeropoint = zeropoint
        self.interpolant = interpolant

        # Parse wave_type (reuse SED's parser for consistency)
        from .sed import SED

        self.wave_type, self.wave_factor = SED._parse_wave_type(wave_type)

        self.blue_limit = blue_limit
        self.red_limit = red_limit

        # Convert string/file input into a function
        self._initialize_tp()

        if _wave_list is not None:
            self.wave_list = np.asarray(_wave_list, dtype=float)
            return

        # Account for wave_factor in wavelength limits
        if self.wave_factor != 1.0:
            if self.blue_limit is not None:
                self.blue_limit /= self.wave_factor
            if self.red_limit is not None:
                self.red_limit /= self.wave_factor

        # Assign blue and red limits from LookupTable
        if isinstance(self._tp, LookupTable):
            if self.blue_limit is None:
                self.blue_limit = float(self._tp.x_min) / self.wave_factor
            if self.red_limit is None:
                self.red_limit = float(self._tp.x_max) / self.wave_factor
        else:
            if self.blue_limit is None or self.red_limit is None:
                raise GalSimIncompatibleValuesError(
                    "red_limit and blue_limit are required if throughput is not a LookupTable.",
                    blue_limit=blue_limit,
                    red_limit=red_limit,
                    throughput=throughput,
                )

        # Set up wave_list
        if isinstance(self._tp, LookupTable):
            self.wave_list = np.asarray(self._tp.getArgs()) / self.wave_factor
            # Remove values outside limits
            self.wave_list = self.wave_list[
                (self.wave_list >= self.blue_limit) & (self.wave_list <= self.red_limit)
            ]
        else:
            self.wave_list = np.array([], dtype=float)

    def _initialize_tp(self):
        if self._tp is not None:
            return
        if isinstance(self._orig_tp, (str, PosixPath)):
            import os

            fname = str(self._orig_tp)
            if os.path.isfile(fname):
                self._tp = LookupTable.from_file(fname, interpolant=self.interpolant)
            else:
                if self.blue_limit is None or self.red_limit is None:
                    raise GalSimIncompatibleValuesError(
                        "red_limit and blue_limit are required if throughput is not a LookupTable.",
                        blue_limit=None,
                        red_limit=None,
                        throughput=self._orig_tp,
                    )
                try:
                    self._tp = eval("lambda wave : " + str(self._orig_tp))
                    test_value = self._tp(
                        self.blue_limit if self.blue_limit is not None else 500.0
                    )
                except Exception as e:
                    raise GalSimValueError(
                        "String throughput must either be a valid filename or something that "
                        "can eval to a function of wave.\n Caught error: %s." % (e),
                        self._orig_tp,
                    )
                if not isinstance(test_value, Real):
                    raise GalSimValueError(
                        "The given throughput function did not return a valid "
                        "number at test wavelength: got %s." % (test_value),
                        self._orig_tp,
                    )
        else:
            self._tp = self._orig_tp

    def __call__(self, wave):
        wave = jnp.asarray(wave, dtype=float)
        in_range = (wave >= self.blue_limit) & (wave <= self.red_limit)
        wave_native = wave * self.wave_factor

        if isinstance(self._tp, LookupTable):
            # Clip to avoid out-of-range errors; out-of-range values
            # are zeroed out by the jnp.where below.
            clipped = jnp.clip(wave_native, self._tp.x_min, self._tp.x_max)
            vals = self._tp(clipped)
        elif callable(self._tp):
            vals = jnp.asarray(self._tp(wave_native), dtype=float)
        else:
            vals = jnp.full_like(wave, float(self._tp))

        return jnp.where(in_range, vals, 0.0)

    @property
    def effective_wavelength(self):
        return self.calculateEffectiveWavelength()

    def calculateEffectiveWavelength(self):
        if not hasattr(self, "_effective_wavelength"):
            if len(self.wave_list) > 0:
                num = self._tp.integrate_product(
                    lambda w: w,
                    self.blue_limit,
                    self.red_limit,
                    x_factor=self.wave_factor,
                )
                denom = (
                    self._tp.integrate(
                        self.blue_limit * self.wave_factor,
                        self.red_limit * self.wave_factor,
                    )
                    / self.wave_factor
                )
            else:
                from jax_galsim.integ import int1d

                def _func_wave(w):
                    return self(w) * w

                num = int1d(_func_wave, self.blue_limit, self.red_limit)
                denom = int1d(self.__call__, self.blue_limit, self.red_limit)

            self._effective_wavelength = float(num / denom)

        return self._effective_wavelength

    def withZeropoint(self, zeropoint):
        from .sed import SED

        if isinstance(zeropoint, str):
            if zeropoint.upper() == "AB":
                AB_source = 3631e-23  # 3631 Jy in units of erg/s/Hz/cm^2
                sed = SED(lambda wave: AB_source, wave_type="nm", flux_type="fnu")
            elif zeropoint.upper() == "ST":
                ST_flambda = 3.63e-8  # erg/s/cm^2/nm
                sed = SED(lambda wave: ST_flambda, wave_type="nm", flux_type="flambda")
            else:
                raise GalSimValueError(
                    "Unrecognized Zeropoint string.",
                    zeropoint,
                    ("AB", "ST"),
                )
            zeropoint = sed

        if isinstance(zeropoint, SED):
            flux = zeropoint.calculateFlux(self)
            zeropoint = 2.5 * np.log10(flux)

        if not isinstance(zeropoint, (float, int)):
            raise TypeError(
                "Don't know how to handle zeropoint of type: {0}".format(
                    type(zeropoint)
                )
            )

        return Bandpass(
            self._orig_tp,
            self.wave_type,
            self.blue_limit,
            self.red_limit,
            zeropoint=zeropoint,
            interpolant=self.interpolant,
            _wave_list=self.wave_list,
            _tp=self._tp,
        )

    def truncate(self, blue_limit=None, red_limit=None, relative_throughput=None):
        if blue_limit is None:
            blue_limit = self.blue_limit
        if red_limit is None:
            red_limit = self.red_limit

        wave_list = self.wave_list
        if len(self.wave_list) > 0:
            if relative_throughput is not None:
                wave = np.asarray(self.wave_list)
                tp = np.asarray(self(wave))
                w = (tp >= tp.max() * relative_throughput).nonzero()
                blue_limit = max(np.min(wave[w]), blue_limit)
                red_limit = min(np.max(wave[w]), red_limit)
            wave_list = wave_list[(wave_list >= blue_limit) & (wave_list <= red_limit)]

        return Bandpass(
            self._orig_tp,
            self.wave_type,
            blue_limit,
            red_limit,
            interpolant=self.interpolant,
            _wave_list=wave_list,
            _tp=self._tp,
        )

    def __mul__(self, other):
        from .sed import SED

        if isinstance(other, SED):
            return other.__mul__(self)

        from . import utilities

        if isinstance(other, Bandpass):
            wave_list, blue_limit, red_limit = utilities.combine_wave_list(
                [self, other]
            )
            tp = lambda w: self(w) * other(w)  # noqa: E731
            return Bandpass(
                tp,
                "nm",
                blue_limit=blue_limit,
                red_limit=red_limit,
                zeropoint=None,
                _wave_list=wave_list,
            )

        # Scalar or callable
        wave_type = "nm"
        if hasattr(other, "__call__"):
            tp = lambda w: self(w) * other(w)  # noqa: E731
        elif isinstance(self._tp, LookupTable):
            wave_type = self.wave_type
            x = self._tp.getArgs()
            f = jnp.asarray(self._tp.getVals()) * other
            tp = _LookupTable(
                x,
                f,
                interpolant=self._tp.interpolant,
            )
        else:
            tp = lambda w: self(w) * other  # noqa: E731

        return Bandpass(
            tp,
            wave_type,
            self.blue_limit,
            self.red_limit,
            _wave_list=self.wave_list,
        )

    def __rmul__(self, other):
        return self * other

    def __eq__(self, other):
        return self is other or (
            isinstance(other, Bandpass)
            and self._orig_tp == other._orig_tp
            and self.blue_limit == other.blue_limit
            and self.red_limit == other.red_limit
            and self.wave_factor == other.wave_factor
            and self.zeropoint == other.zeropoint
            and np.array_equal(self.wave_list, other.wave_list)
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        if not hasattr(self, "_hash"):
            self._hash = hash(
                (
                    "galsim.Bandpass",
                    ensure_hashable(self._orig_tp)
                    if not isinstance(self._orig_tp, LookupTable)
                    else self._orig_tp,
                    self.blue_limit,
                    self.red_limit,
                    self.wave_factor,
                    self.zeropoint,
                    tuple(self.wave_list),
                )
            )
        return self._hash

    def __repr__(self):
        return (
            "galsim.Bandpass(%r, wave_type=%r, blue_limit=%r, red_limit=%r, zeropoint=%r, "
            "interpolant=%r, _wave_list=array(%r))"
        ) % (
            self._orig_tp,
            self.wave_type,
            self.blue_limit,
            self.red_limit,
            self.zeropoint,
            self.interpolant,
            self.wave_list.tolist(),
        )

    def __str__(self):
        return "galsim.Bandpass(%s)" % self._orig_tp

    def tree_flatten(self):
        if isinstance(self._tp, LookupTable):
            children = (self._tp,)
        else:
            children = ()
        aux_data = {
            "wave_type": self.wave_type,
            "wave_factor": self.wave_factor,
            "blue_limit": self.blue_limit,
            "red_limit": self.red_limit,
            "zeropoint": self.zeropoint,
            "interpolant": self.interpolant,
            "wave_list": self.wave_list,
            "_orig_tp": self._orig_tp,
            "_tp_is_lt": isinstance(self._tp, LookupTable),
            "_tp_callable": self._tp if not isinstance(self._tp, LookupTable) else None,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.wave_type = aux_data["wave_type"]
        obj.wave_factor = aux_data["wave_factor"]
        obj.blue_limit = aux_data["blue_limit"]
        obj.red_limit = aux_data["red_limit"]
        obj.zeropoint = aux_data["zeropoint"]
        obj.interpolant = aux_data["interpolant"]
        obj.wave_list = aux_data["wave_list"]
        obj._orig_tp = aux_data["_orig_tp"]
        if aux_data["_tp_is_lt"]:
            obj._tp = children[0]
        else:
            obj._tp = aux_data["_tp_callable"]
        return obj
