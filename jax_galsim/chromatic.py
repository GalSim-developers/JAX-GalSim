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

import galsim as _galsim
import numpy as np
from jax.tree_util import register_pytree_node_class

from jax_galsim.core.utils import implements
from jax_galsim.errors import (
    GalSimNotImplementedError,
    GalSimSEDError,
)
from jax_galsim.gsparams import GSParams


@implements(
    _galsim.ChromaticObject,
    lax_description="""\
The JAX-GalSim version of ChromaticObject only supports the separable
drawing path (GSObject * SED). Non-separable profiles, chromatic PSFs,
ChromaticConvolution, ChromaticSum, InterpolatedChromaticObject, and
photon shooting are not implemented.
""",
)
class ChromaticObject:
    """Base class for wavelength-dependent objects."""

    def __init__(self, obj):
        from jax_galsim.gsobject import GSObject

        if not isinstance(obj, GSObject) and not isinstance(obj, ChromaticObject):
            raise TypeError(
                "Can only directly instantiate ChromaticObject with a GSObject "
                "or ChromaticObject argument."
            )
        self._obj = obj
        self.separable = True
        self.interpolated = False
        self.deinterpolated = self

    @property
    def sed(self):
        return self._obj.sed

    @property
    def wave_list(self):
        return self.sed.wave_list

    @property
    def gsparams(self):
        return self._obj.gsparams

    @property
    def redshift(self):
        return self.sed.redshift

    @property
    def spectral(self):
        return self.sed.spectral

    @property
    def dimensionless(self):
        return self.sed.dimensionless

    def evaluateAtWavelength(self, wave):
        raise GalSimNotImplementedError(
            "evaluateAtWavelength not implemented for base ChromaticObject"
        )

    def calculateFlux(self, bandpass):
        return self.sed.calculateFlux(bandpass)

    @staticmethod
    def _get_multiplier(sed, bandpass, wave_list):
        wave_list = np.array(wave_list)
        from jax_galsim.table import _LookupTable

        if len(wave_list) > 0:
            bp = _LookupTable(wave_list, bandpass(wave_list), "linear")
            multiplier = bp.integrate_product(sed)
        else:
            from jax_galsim.integ import int1d

            multiplier = int1d(
                lambda w: sed(w) * bandpass(w),
                bandpass.blue_limit,
                bandpass.red_limit,
            )
        return multiplier

    def drawImage(self, bandpass, image=None, integrator="quadratic", **kwargs):
        if self.sed.dimensionless:
            raise GalSimSEDError(
                "Can only draw ChromaticObjects with spectral SEDs.", self.sed
            )

        if not self.separable:
            raise GalSimNotImplementedError(
                "Non-separable ChromaticObject drawing is not implemented in jax_galsim."
            )

        from . import utilities

        # Determine combined wave_list
        wave_list, _, _ = utilities.combine_wave_list(self, bandpass)

        # For separable profiles: get fiducial profile and scale by integrated flux
        wave0 = bandpass.effective_wavelength
        prof0 = self.evaluateAtWavelength(wave0)

        # Compute the integrated flux multiplier
        multiplier = self._get_multiplier(self.sed, bandpass, tuple(wave_list))

        # Scale the fiducial profile
        sed_at_wave0 = self.sed(wave0)
        prof0 = prof0 * (multiplier / sed_at_wave0)

        # Draw the scaled profile
        image = prof0.drawImage(image=image, **kwargs)
        return image

    def __eq__(self, other):
        return self is other or (
            isinstance(other, ChromaticObject)
            and hasattr(other, "_obj")
            and self._obj == other._obj
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(("galsim.ChromaticObject", self._obj))

    def __repr__(self):
        return "galsim.ChromaticObject(%r)" % self._obj

    def __str__(self):
        return "galsim.ChromaticObject(%s)" % self._obj


@register_pytree_node_class
class SimpleChromaticTransformation(ChromaticObject):
    """A GSObject times an SED -- the simplest kind of chromatic object."""

    def __init__(self, obj, sed=None, gsparams=None, propagate_gsparams=True):
        from jax_galsim.sed import SED

        if sed is None:
            sed = SED(1.0, "nm", "1")

        self.separable = True
        self.interpolated = False
        self.deinterpolated = self

        self._gsparams = GSParams.check(gsparams, obj.gsparams)
        self._propagate_gsparams = propagate_gsparams

        self._original = obj
        self._flux_ratio = sed

        if self._propagate_gsparams:
            self._original = self._original.withGSParams(self._gsparams)

    @property
    def original(self):
        return self._original

    @property
    def sed(self):
        return self._flux_ratio * self._original.flux

    @property
    def wave_list(self):
        return self.sed.wave_list

    @property
    def gsparams(self):
        return self._gsparams

    @property
    def spectral(self):
        return self.sed.spectral

    @property
    def dimensionless(self):
        return self.sed.dimensionless

    def evaluateAtWavelength(self, wave):
        return self._original.withFlux(self.sed(wave))

    def drawImage(self, bandpass, image=None, integrator="quadratic", **kwargs):
        if self.sed.dimensionless:
            raise GalSimSEDError(
                "Can only draw ChromaticObjects with spectral SEDs.", self.sed
            )
        return ChromaticObject.drawImage(self, bandpass, image, integrator, **kwargs)

    def __eq__(self, other):
        return self is other or (
            isinstance(other, SimpleChromaticTransformation)
            and self._original == other._original
            and self._flux_ratio == other._flux_ratio
            and self._gsparams == other._gsparams
            and self._propagate_gsparams == other._propagate_gsparams
        )

    def __hash__(self):
        if not hasattr(self, "_hash"):
            self._hash = hash(
                (
                    "galsim.SimpleChromaticTransformation",
                    self._original,
                    self._flux_ratio,
                    self._gsparams,
                    self._propagate_gsparams,
                )
            )
        return self._hash

    def __repr__(self):
        return (
            "galsim.SimpleChromaticTransformation(%r, sed=%r, "
            "gsparams=%r, propagate_gsparams=%r)"
        ) % (
            self._original,
            self._flux_ratio,
            self._gsparams,
            self._propagate_gsparams,
        )

    def __str__(self):
        return str(self._original) + " * " + str(self.sed)

    def tree_flatten(self):
        children = (self._original, self._flux_ratio)
        aux_data = {
            "gsparams": self._gsparams,
            "propagate_gsparams": self._propagate_gsparams,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj._original = children[0]
        obj._flux_ratio = children[1]
        obj._gsparams = aux_data["gsparams"]
        obj._propagate_gsparams = aux_data["propagate_gsparams"]
        obj.separable = True
        obj.interpolated = False
        obj.deinterpolated = obj
        return obj
