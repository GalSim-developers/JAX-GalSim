from dataclasses import dataclass

import galsim as _galsim

from jax_galsim.core.utils import implements


@implements(_galsim.GSParams)
@dataclass(frozen=True, repr=False)
class GSParams:
    minimum_fft_size: int = 128
    maximum_fft_size: int = 8192
    folding_threshold: float = 5.0e-3
    stepk_minimum_hlr: float = 5.0
    maxk_threshold: float = 1.0e-3
    kvalue_accuracy: float = 1.0e-5
    xvalue_accuracy: float = 1.0e-5
    table_spacing: int = 1
    realspace_relerr: float = 1.0e-4
    realspace_abserr: float = 1.0e-6
    integration_relerr: float = 1.0e-6
    integration_abserr: float = 1.0e-8
    shoot_accuracy: float = 1.0e-5

    @classmethod
    def from_galsim(cls, gsparams):
        """Create a jax_galsim `GSParams` from a `galsim.GSParams` object."""
        if not isinstance(gsparams, _galsim.GSParams):
            raise TypeError(
                "gsparams must be a %s instance, got %s"
                % (_galsim.GSParams.__name__, gsparams)
            )
        return cls(
            gsparams._minimum_fft_size,
            gsparams.maximum_fft_size,
            gsparams.folding_threshold,
            gsparams.stepk_minimum_hlr,
            gsparams.maxk_threshold,
            gsparams.kvalue_accuracy,
            gsparams.xvalue_accuracy,
            gsparams.table_spacing,
            gsparams.realspace_relerr,
            gsparams.realspace_abserr,
            gsparams.integration_relerr,
            gsparams.integration_abserr,
            gsparams.shoot_accuracy,
        )

    def to_galsim(self):
        return _galsim.GSParams(
            minimum_fft_size=self.minimum_fft_size,
            maximum_fft_size=self.maximum_fft_size,
            folding_threshold=self.folding_threshold,
            stepk_minimum_hlr=self.stepk_minimum_hlr,
            maxk_threshold=self.maxk_threshold,
            kvalue_accuracy=self.kvalue_accuracy,
            xvalue_accuracy=self.xvalue_accuracy,
            table_spacing=self.table_spacing,
            realspace_relerr=self.realspace_relerr,
            realspace_abserr=self.realspace_abserr,
            integration_relerr=self.integration_relerr,
            integration_abserr=self.integration_abserr,
            shoot_accuracy=self.shoot_accuracy,
        )

    @staticmethod
    @implements(_galsim.GSParams.check)
    def check(gsparams, default=None, **kwargs):
        if gsparams is None:
            if default is not None:
                if isinstance(default, GSParams):
                    gsparams = default
                else:
                    raise TypeError("Invalid default GSParams: %s" % default)
            else:
                gsparams = GSParams.default
        elif not isinstance(gsparams, GSParams):
            raise TypeError("Invalid GSParams: %s" % gsparams)
        return gsparams.withParams(**kwargs)

    @implements(_galsim.GSParams.withParams)
    def withParams(self, **kwargs):
        if len(kwargs) == 0:
            return self
        else:
            d = self.__dict__.copy()
            for k in kwargs:
                if k not in d:
                    raise TypeError("parameter %s is invalid" % k)
                d[k] = kwargs[k]
            return GSParams(**d)

    @staticmethod
    @implements(_galsim.GSParams.combine)
    def combine(gsp_list):
        if len(gsp_list) == 1:
            return gsp_list[0]
        elif all(g == gsp_list[0] for g in gsp_list[1:]):
            return gsp_list[0]
        else:
            return GSParams(
                max([g.minimum_fft_size for g in gsp_list if g is not None]),
                max([g.maximum_fft_size for g in gsp_list if g is not None]),
                min([g.folding_threshold for g in gsp_list if g is not None]),
                max([g.stepk_minimum_hlr for g in gsp_list if g is not None]),
                min([g.maxk_threshold for g in gsp_list if g is not None]),
                min([g.kvalue_accuracy for g in gsp_list if g is not None]),
                min([g.xvalue_accuracy for g in gsp_list if g is not None]),
                min([g.table_spacing for g in gsp_list if g is not None]),
                min([g.realspace_relerr for g in gsp_list if g is not None]),
                min([g.realspace_abserr for g in gsp_list if g is not None]),
                min([g.integration_relerr for g in gsp_list if g is not None]),
                min([g.integration_abserr for g in gsp_list if g is not None]),
                min([g.shoot_accuracy for g in gsp_list if g is not None]),
            )

    # Define once the order of args in __init__, since we use it a few times.
    def _getinitargs(self):
        return (
            self.minimum_fft_size,
            self.maximum_fft_size,
            self.folding_threshold,
            self.stepk_minimum_hlr,
            self.maxk_threshold,
            self.kvalue_accuracy,
            self.xvalue_accuracy,
            self.table_spacing,
            self.realspace_relerr,
            self.realspace_abserr,
            self.integration_relerr,
            self.integration_abserr,
            self.shoot_accuracy,
        )

    def __getstate__(self):
        return self._getinitargs()

    def __setstate__(self, state):
        self.__init__(*state)

    def __repr__(self):
        return (
            "galsim.GSParams(%d,%d,%r,%r,%r,%r,%r,%d,%r,%r,%r,%r,%r)"
            % self._getinitargs()
        )

    def __eq__(self, other):
        return self is other or (
            isinstance(other, GSParams) and self._getinitargs() == other._getinitargs()
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(repr(self))


# We use the default a lot, so make it a class attribute.
GSParams.default = GSParams()
