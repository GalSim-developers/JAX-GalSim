from contextlib import contextmanager

import galsim as _galsim
import jax
import jax.numpy as jnp
import jax.random as jrng
from jax._src.numpy.util import _wraps
from jax.tree_util import register_pytree_node_class

from jax_galsim.core.utils import cast_to_python_int
from jax_galsim.errors import (
    GalSimIncompatibleValuesError,
    GalSimRangeError,
    GalSimUndefinedBoundsError,
    GalSimValueError,
)
from jax_galsim.random import BaseDeviate, UniformDeviate

from ._pyfits import pyfits

_JAX_GALSIM_PHOTON_ARRAY_SIZE = None


@contextmanager
def fixed_photon_array_size(size):
    """Context manager to temporarily set a fixed size for photon arrays."""
    global _JAX_GALSIM_PHOTON_ARRAY_SIZE
    old_size = _JAX_GALSIM_PHOTON_ARRAY_SIZE
    _JAX_GALSIM_PHOTON_ARRAY_SIZE = size
    try:
        yield
    finally:
        _JAX_GALSIM_PHOTON_ARRAY_SIZE = old_size


@_wraps(
    _galsim.PhotonArray,
    lax_description="""\
JAX-GalSim PhotonArrays have significant differences from the original GalSim.

    - They always copy input data and operations on them always copy.
    - They (usually) do not do any type/size checking on input data.
    - They do not support indexed assignement directly on the attributes.
    - The additional properties `dxdz`, `dydz`, `wavelength`, `pupil_u`, `pupil_v`,
      and `time` are set to arrays of NaNs by default. They are thus always allocated.
      However, the methods like `hasAllocatedAngles` etc. return false if the arrays
      are all NaNs.

Further, a context manager `fixed_photon_array_size` is provided to temporarily
set a fixed size for photon arrays.

  - This functionality is useful when apply JIT to operations that vary the
    number of photons drawn using Poisson statistics.
  - When using this context manager, the attribute `_nokeep` stores a boolean mask
    indicating which photons are to be kept.
  - The attribute `_num_keep` stores the number of photons to be kept. If you set
    this attribute, the `_nokeep` mask is updated by sorting _nokeep so that things
    to be kept are at the start, the first `_num_keep` photons are marked to be kept,
    and finally the array is sorted back to its original order.
  - You may get an error if you ask for more photons than the fixed size, but not always,
    especially in JITed code.
  - Operations on photon arrays with fixed sizes but different `_num_keep` values are not
    defined and will not raise an error.
  - The `.flux` property scales `._flux` by the ratio of the fixed size to the number of kept photons
    and sets non-kept photons to zero flux. Setting `.flux` to `._flux` will break things badly.
  - Profiles should always draw the full number of photons given by `.size()` or `len()`
    so that they use fixed array sizes and things are JIT compatible.

**The `_nokeep`, `_num_keep`, and associated methods are private and should not be set by hand
unless you know what you are doing!**
""",
)
@register_pytree_node_class
class PhotonArray:
    def __init__(
        self,
        N,
        x=None,
        y=None,
        flux=None,
        dxdz=None,
        dydz=None,
        wavelength=None,
        pupil_u=None,
        pupil_v=None,
        time=None,
        _nokeep=None,
    ):
        self._Ntot = _JAX_GALSIM_PHOTON_ARRAY_SIZE or N
        if _JAX_GALSIM_PHOTON_ARRAY_SIZE is not None:
            try:
                # this will raise a boolean conversion error in JAX
                # which we swallow
                err_cond = (N > _JAX_GALSIM_PHOTON_ARRAY_SIZE) or False
            except Exception:
                err_cond = False

            if err_cond:
                raise GalSimValueError(
                    f"The given photon array size {N} is larger than "
                    f"the allowed total size {_JAX_GALSIM_PHOTON_ARRAY_SIZE}."
                )
        if _nokeep is not None:
            self._nokeep = _nokeep
        else:
            self._nokeep = jnp.arange(self._Ntot) >= N

        # Only x, y, flux are built by default, since these are always required.
        # The others we leave as None unless/until they are needed.
        self._x = jnp.zeros(self._Ntot, dtype=float)
        self._y = jnp.zeros(self._Ntot, dtype=float)
        self._flux = jnp.zeros(self._Ntot, dtype=float)
        self._dxdz = jnp.full(self._Ntot, jnp.nan, dtype=float)
        self._dydz = jnp.full(self._Ntot, jnp.nan, dtype=float)
        self._wave = jnp.full(self._Ntot, jnp.nan, dtype=float)
        self._pupil_u = jnp.full(self._Ntot, jnp.nan, dtype=float)
        self._pupil_v = jnp.full(self._Ntot, jnp.nan, dtype=float)
        self._time = jnp.full(self._Ntot, jnp.nan, dtype=float)
        self._is_corr = jnp.array(False)

        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if flux is not None:
            self.flux = flux
        if dxdz is not None:
            self.dxdz = dxdz
        if dydz is not None:
            self.dydz = dydz
        if wavelength is not None:
            self.wavelength = wavelength
        if pupil_u is not None:
            self.pupil_u = pupil_u
        if pupil_v is not None:
            self.pupil_v = pupil_v
        if time is not None:
            self.time = time

    @classmethod
    @_wraps(
        _galsim.PhotonArray.fromArrays,
        lax_description="JAX-GalSim does not do input type/size checking.",
    )
    def fromArrays(
        cls,
        x,
        y,
        flux,
        dxdz=None,
        dydz=None,
        wavelength=None,
        pupil_u=None,
        pupil_v=None,
        time=None,
        is_corr=False,
    ):
        return cls._fromArrays(
            x, y, flux, dxdz, dydz, wavelength, pupil_u, pupil_v, time, is_corr
        )

    @classmethod
    @_wraps(_galsim.PhotonArray._fromArrays)
    def _fromArrays(
        cls,
        x,
        y,
        flux,
        dxdz=None,
        dydz=None,
        wavelength=None,
        pupil_u=None,
        pupil_v=None,
        time=None,
        is_corr=False,
    ):
        if (
            _JAX_GALSIM_PHOTON_ARRAY_SIZE is not None
            and x.shape[0] != _JAX_GALSIM_PHOTON_ARRAY_SIZE
        ):
            raise GalSimValueError(
                "The given arrays do not match the expected total size",
                x.shape[0],
                _JAX_GALSIM_PHOTON_ARRAY_SIZE,
            )

        ret = cls.__new__(cls)
        ret._Ntot = _JAX_GALSIM_PHOTON_ARRAY_SIZE or x.shape[0]
        ret._x = x.copy()
        ret._y = y.copy()
        ret._flux = flux.copy()
        ret._nokeep = jnp.arange(ret._Ntot) >= x.shape[0]
        ret._dxdz = (
            dxdz.copy()
            if dxdz is not None
            else jnp.full(ret._Ntot, jnp.nan, dtype=float)
        )
        ret._dydz = (
            dydz.copy()
            if dydz is not None
            else jnp.full(ret._Ntot, jnp.nan, dtype=float)
        )
        ret._wave = (
            wavelength.copy()
            if wavelength is not None
            else jnp.full(ret._Ntot, jnp.nan, dtype=float)
        )
        ret._pupil_u = (
            pupil_u.copy()
            if pupil_u is not None
            else jnp.full(ret._Ntot, jnp.nan, dtype=float)
        )
        ret._pupil_v = (
            pupil_v.copy()
            if pupil_v is not None
            else jnp.full(ret._Ntot, jnp.nan, dtype=float)
        )
        ret._time = (
            time.copy()
            if time is not None
            else jnp.full(ret._Ntot, jnp.nan, dtype=float)
        )
        ret.setCorrelated(is_corr)
        return ret

    def tree_flatten(self):
        children = (
            (self._x, self._y, self._flux, self._nokeep),
            {
                "dxdz": self._dxdz,
                "dydz": self._dydz,
                "wavelength": self._wave,
                "pupil_u": self._pupil_u,
                "pupil_v": self._pupil_v,
                "time": self._time,
                "is_corr": self._is_corr,
            },
        )
        aux_data = (self._Ntot,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        ret = cls.__new__(cls)
        ret._Ntot = aux_data[0]
        ret._nokeep = children[0][3]
        ret._x = children[0][0]
        ret._y = children[0][1]
        ret._flux = children[0][2]
        ret._dxdz = children[1]["dxdz"]
        ret._dydz = children[1]["dydz"]
        ret._wave = children[1]["wavelength"]
        ret._pupil_u = children[1]["pupil_u"]
        ret._pupil_v = children[1]["pupil_v"]
        ret._time = children[1]["time"]
        ret._is_corr = children[1]["is_corr"]
        return ret

    def size(self):
        """Return the size of the photon array.  Equivalent to ``len(self)``."""
        return self._Ntot

    def __len__(self):
        return self._Ntot

    @property
    def _num_keep(self):
        """The number of actual photons in the array."""
        return jnp.sum(~self._nokeep).astype(int)

    @_num_keep.setter
    def _num_keep(self, num_keep):
        """Set the number of actual photons in the array."""
        sinds = jnp.argsort(self._nokeep)
        self._sort_by_nokeep(sinds=sinds)
        self._nokeep = jnp.arange(self._Ntot) >= num_keep
        self._set_self_at_inds(sinds)

    @property
    def x(self):
        """The incidence x position in image coordinates (pixels), typically at the top of
        the detector.
        """
        return self._x

    @x.setter
    def x(self, value):
        self._x = self._x.at[:].set(value)

    @property
    def y(self):
        """The incidence y position in image coordinates (pixels), typically at the top of
        the detector.
        """
        return self._y

    @y.setter
    def y(self, value):
        self._y = self._y.at[:].set(value)

    @property
    def flux(self):
        """The flux of the photons."""
        # we use jax.lax.cond to save some multiplications when
        # there are no masked photos.
        return jax.lax.cond(
            self._Ntot == self._num_keep,
            lambda flux, ratio: flux,
            lambda flux, ratio: flux * ratio,
            jnp.where(self._nokeep, 0.0, self._flux),
            self._Ntot / self._num_keep,
        )

    @flux.setter
    def flux(self, value):
        self._flux = self._flux.at[:].set(
            value
            # scale it down to account for scaling in flux getter above
            # this factor has to be computed after _nokeep is set above
            # so that _num_keep is the right value
            / (self._Ntot / self._num_keep)
        )

    @property
    def dxdz(self):
        """The tangent of the inclination angles in the x direction: dx/dz."""
        return self._dxdz

    @dxdz.setter
    def dxdz(self, value):
        self._dxdz = self._dxdz.at[:].set(value)
        self._dydz = jax.lax.cond(
            jnp.any(jnp.isfinite(self._dxdz)) & jnp.all(~jnp.isfinite(self._dydz)),
            lambda dydz: jnp.zeros_like(dydz),
            lambda dydz: dydz,
            self._dydz,
        )

    @property
    def dydz(self):
        """The tangent of the inclination angles in the y direction: dy/dz."""
        return self._dydz

    @dydz.setter
    def dydz(self, value):
        self._dydz = self._dydz.at[:].set(value)
        self._dxdz = jax.lax.cond(
            jnp.any(jnp.isfinite(self._dydz)) & jnp.all(~jnp.isfinite(self._dxdz)),
            lambda dxdz: jnp.zeros_like(dxdz),
            lambda dxdz: dxdz,
            self._dxdz,
        )

    @property
    def wavelength(self):
        """The wavelength of the photons (in nm)."""
        return self._wave

    @wavelength.setter
    def wavelength(self, value):
        self._wave = self._wave.at[:].set(value)

    @property
    def pupil_u(self):
        """Horizontal location of photon as it intersected the entrance pupil plane."""
        return self._pupil_u

    @pupil_u.setter
    def pupil_u(self, value):
        self._pupil_u = self._pupil_u.at[:].set(value)
        self._pupil_v = jax.lax.cond(
            jnp.any(jnp.isfinite(self._pupil_u))
            & jnp.all(~jnp.isfinite(self._pupil_v)),
            lambda pupil_v: jnp.zeros_like(pupil_v),
            lambda pupil_v: pupil_v,
            self._pupil_v,
        )

    @property
    def pupil_v(self):
        """Vertical location of photon as it intersected the entrance pupil plane."""
        return self._pupil_v

    @pupil_v.setter
    def pupil_v(self, value):
        self._pupil_v = self._pupil_v.at[:].set(value)
        self._pupil_u = jax.lax.cond(
            jnp.any(jnp.isfinite(self._pupil_v))
            & jnp.all(~jnp.isfinite(self._pupil_u)),
            lambda pupil_u: jnp.zeros_like(pupil_u),
            lambda pupil_u: pupil_u,
            self._pupil_u,
        )

    @property
    def time(self):
        """Time stamp of when photon encounters the pupil plane."""
        return self._time

    @time.setter
    def time(self, value):
        self._time = self._time.at[:].set(value)

    def hasAllocatedAngles(self):
        """Returns whether the arrays for the incidence angles `dxdz` and `dydz` have been
        allocated.
        """
        return jnp.any(jnp.isfinite(self.dxdz) | jnp.isfinite(self.dydz))

    def allocateAngles(self):
        """Allocate memory for the incidence angles, `dxdz` and `dydz`."""
        pass

    def hasAllocatedWavelengths(self):
        """Returns whether the `wavelength` array has been allocated."""
        return jnp.any(jnp.isfinite(self.wavelength))

    def allocateWavelengths(self):
        """Allocate the memory for the `wavelength` array."""
        pass

    def hasAllocatedPupil(self):
        """Returns whether the arrays for the pupil coordinates `pupil_u` and `pupil_v` have been
        allocated.
        """
        return jnp.any(jnp.isfinite(self.pupil_u) | jnp.isfinite(self.pupil_v))

    def allocatePupil(self):
        """Allocate the memory for the pupil coordinates, `pupil_u` and `pupil_v`."""
        pass

    def hasAllocatedTimes(self):
        """Returns whether the array for the time stamps `time` has been allocated."""
        return jnp.any(jnp.isfinite(self.time))

    def allocateTimes(self):
        """Allocate the memory for the time stamps, `time`."""
        return True

    def isCorrelated(self):
        """Returns whether the photons are correlated"""
        from .deprecated import depr

        depr(
            "isCorrelated",
            2.5,
            "",
            "We don't think this is necessary anymore.  If you have a use case that "
            "requires it, please open an issue.",
        )
        return self._is_corr

    def setCorrelated(self, is_corr=True):
        """Set whether the photons are correlated"""
        from .deprecated import depr

        depr(
            "setCorrelated",
            2.5,
            "",
            "We don't think this is necessary anymore.  If you have a use case that "
            "requires it, please open an issue.",
        )
        self._is_corr = jnp.array(is_corr, dtype=bool)

    def getTotalFlux(self):
        """Return the total flux of all the photons."""
        return self.flux.sum()

    def setTotalFlux(self, flux):
        """Rescale the photon fluxes to achieve the given total flux.

        Parameter:
            flux:       The target flux
        """
        self.scaleFlux(flux / self.getTotalFlux())

        return self

    def scaleFlux(self, scale):
        """Rescale the photon fluxes by the given factor.

        Parameter:
            scale:      The factor by which to scale the fluxes.
        """
        self._flux *= scale

        return self

    def scaleXY(self, scale):
        """Scale the photon positions (`x` and `y`) by the given factor.

        Parameter:
            scale:      The factor by which to scale the positions.
        """
        self._x *= scale
        self._y *= scale

        return self

    def _sort_by_nokeep(self, sinds=None):
        # now sort things to keep to the left
        if sinds is None:
            sinds = jnp.argsort(self._nokeep)
        self._x = self._x.at[sinds].get()
        self._y = self._y.at[sinds].get()
        self._flux = self._flux.at[sinds].get()
        self._nokeep = self._nokeep.at[sinds].get()
        self._dxdz = self._dxdz.at[sinds].get()
        self._dydz = self._dydz.at[sinds].get()
        self._wave = self._wave.at[sinds].get()
        self._pupil_u = self._pupil_u.at[sinds].get()
        self._pupil_v = self._pupil_v.at[sinds].get()
        self._time = self._time.at[sinds].get()

        return self

    def _set_self_at_inds(self, sinds):
        self._x = self._x.at[sinds].set(self._x)
        self._y = self._y.at[sinds].set(self._y)
        self._flux = self._flux.at[sinds].set(self._flux)
        self._nokeep = self._nokeep.at[sinds].set(self._nokeep)
        self._dxdz = self._dxdz.at[sinds].set(self._dxdz)
        self._dydz = self._dydz.at[sinds].set(self._dydz)
        self._wave = self._wave.at[sinds].set(self._wave)
        self._pupil_u = self._pupil_u.at[sinds].set(self._pupil_u)
        self._pupil_v = self._pupil_v.at[sinds].set(self._pupil_v)
        self._time = self._time.at[sinds].set(self._time)

        return self

    @_wraps(_galsim.PhotonArray.assignAt)
    def assignAt(self, istart, rhs):
        from .deprecated import depr

        depr(
            "PhotonArray.assignAt",
            2.5,
            "copyFrom(rhs, slice(istart, istart+rhs.size()))",
        )
        if istart + rhs.size() > self.size():
            raise GalSimValueError(
                "The given rhs does not fit into this array starting at %d" % istart,
                rhs,
            )
        s = slice(istart, istart + rhs.size())
        return self._copyFrom(rhs, s, slice(None))

    @_wraps(
        _galsim.PhotonArray.copyFrom,
        lax_description="The JAX version of PhotonArray.copyFrom does not raise for out of bounds indices.",
    )
    def copyFrom(
        self,
        rhs,
        target_indices=slice(None),
        source_indices=slice(None),
        do_xy=True,
        do_flux=True,
        do_other=True,
    ):
        return self._copyFrom(
            rhs, target_indices, source_indices, do_xy, do_flux, do_other
        )

    def _copyFrom(
        self,
        rhs,
        target_indices,
        source_indices,
        do_xy=True,
        do_flux=True,
        do_other=True,
    ):
        """Equivalent to self.copyFrom(rhs, target_indices, source_indices), but without any
        checks that the indices are valid.
        """
        # Aliases for notational convenience.
        s1 = target_indices
        s2 = source_indices

        @jax.jit
        def _cond_set_indices(arr1, arr2, cond_val):
            return jax.lax.cond(
                cond_val,
                lambda arr1, arr2: arr1.at[s1].set(arr2.at[s2].get()),
                lambda arr1, arr2: arr1,
                arr1,
                arr2,
            )

        old_flux_ratio = self._Ntot / self._num_keep

        if do_xy or do_flux or do_other:
            self._nokeep = self._nokeep.at[s1].set(rhs._nokeep.at[s2].get())

        new_flux_ratio = self._Ntot / self._num_keep

        if do_xy:
            self._x = self._x.at[s1].set(rhs.x.at[s2].get())
            self._y = self._y.at[s1].set(rhs.y.at[s2].get())

        if do_flux:
            # we first scale the existing fluxes to account for the change in num_keep
            self._flux = (
                self._flux
                # this factor gets us back to true flux
                * old_flux_ratio
                # this factor gets us back to the internal units
                / new_flux_ratio
            )

            # next we assign the RHS fluxes accounting for the change in num_keep from the
            # RHS to the new flux_ratio
            self._flux = self._flux.at[s1].set(
                rhs._flux.at[s2].get()
                # these factors conserve the flux of the assigned photons
                # gets us to the true flux of the photon
                * (rhs._Ntot / rhs._num_keep)
                # scale it back down to account for scaling later
                # this factor has to be computed after _nokeep is set above
                # so that _num_keep is the right value
                / new_flux_ratio
            )

        if do_other:
            self._dxdz = _cond_set_indices(
                self._dxdz, rhs.dxdz, rhs.hasAllocatedAngles()
            )
            self._dydz = _cond_set_indices(
                self._dydz, rhs.dydz, rhs.hasAllocatedAngles()
            )
            self._wave = _cond_set_indices(
                self._wave, rhs.wavelength, rhs.hasAllocatedWavelengths()
            )
            self._pupil_u = _cond_set_indices(
                self._pupil_u, rhs.pupil_u, rhs.hasAllocatedPupil()
            )
            self._pupil_v = _cond_set_indices(
                self._pupil_v, rhs.pupil_v, rhs.hasAllocatedPupil()
            )
            self._time = _cond_set_indices(
                self._time, rhs.time, rhs.hasAllocatedTimes()
            )

        return self

    def _assign_from_categorical_index(self, cat_inds, cat_ind_to_assign, rhs):
        """Assign the contents of another `PhotonArray` to this one at locations
        where cat_ind == cat_ind_to_assign.
        """
        msk = cat_ind_to_assign == cat_inds
        old_flux_ratio = self._Ntot / self._num_keep
        self._nokeep = jnp.where(msk, rhs._nokeep, self._nokeep)
        new_flux_ratio = self._Ntot / self._num_keep

        rhs_flux_ratio = rhs._Ntot / rhs._num_keep

        self._x = jnp.where(msk, rhs._x, self._x)
        self._y = jnp.where(msk, rhs._y, self._y)
        self._flux = jnp.where(
            msk,
            rhs._flux * rhs_flux_ratio / new_flux_ratio,
            self._flux * old_flux_ratio / new_flux_ratio,
        )

        self._dxdz = jnp.where(msk, rhs._dxdz, self._dxdz)
        self._dydz = jnp.where(msk, rhs._dydz, self._dydz)
        self._wave = jnp.where(msk, rhs._wave, self._wave)
        self._pupil_u = jnp.where(msk, rhs._pupil_u, self._pupil_u)
        self._pupil_v = jnp.where(msk, rhs._pupil_v, self._pupil_v)
        self._time = jnp.where(msk, rhs._time, self._time)

        return self

    def convolve(self, rhs, rng=None):
        """Convolve this `PhotonArray` with another.

        ..note::

            If both self and rhs have wavelengths, angles, pupil coordinates or times assigned,
            then the values from the first array (i.e. self) take precedence.
        """
        if rhs.size() != self.size():
            raise GalSimIncompatibleValuesError(
                "PhotonArray.convolve with unequal size arrays", self_pa=self, rhs=rhs
            )

        # We need to make sure that the arrays are sorted by _nokeep before convolving
        # we sort them back to their original order after convolving
        self_sinds = jnp.argsort(self._nokeep)
        rhs_sinds = jnp.argsort(rhs._nokeep)
        self._sort_by_nokeep(sinds=self_sinds)
        rhs._sort_by_nokeep(sinds=rhs_sinds)

        # When two photon arrays are convolved, you basically perturb the positions of one
        # by adding the positions of the other. For example, if you have a delta function
        # and want to convolve with a Gaussian, then the photon arrays are an array of zeros
        # for the delta function and an array of Gaussian draws for the Gaussian. The convolution
        # is then implemented by adding the positions of the two arrays.

        # The edge case here is if the photons in anb array are correlated. for example, if
        # you draw photons from a sum of two profiles, you could have the photons from one
        # of the components only at the start of the array and the photons from the other
        # component only at the end of the array like this
        #
        #  [A, A, A, ..., A, B, B, B. ..., B]
        #
        # where A and B represent which component the photon came from. If you convolve two
        # photon arrays where both arrays have intenral correlations in the ordering of the
        # photons, then you need to randomly sort one of the arrays before the convolution.
        # Otherwise you won't properly be adding a random draew from one profile to the other.

        # the indexing and PRNG code snippets below handle this case of convolving two internally
        # correlated photon arrays.

        # these are indicies that randomly sort the RHS's photons.
        rng = BaseDeviate(rng)
        rsinds = jrng.choice(
            rng._state.split_one(),
            self._Ntot,
            shape=(self.size(),),
            replace=False,
        )
        # these indices do not randomly sort the RHS's photons
        nrsinds = jnp.arange(self.size())

        # now we randomly sort if both arrays are internally correlated
        # however there is a catch. The RHS may not be keeping all of its photons
        # (i.e., rhs._nokeep is True for some photons). In this case, we additionally
        # sort the random indices by the value of rhs._nokeep so that the photons to be
        # kept are still at the front of the array but are in a new random order.
        sinds = jax.lax.cond(
            self._is_corr & rhs._is_corr,
            lambda nrsinds, rsinds: rsinds.at[
                jnp.argsort(rhs._nokeep.at[rsinds].get())
            ].get(),
            lambda nrsinds, rsinds: nrsinds,
            nrsinds,
            rsinds,
        )

        self.dxdz, self.dydz = jax.lax.cond(
            rhs.hasAllocatedAngles() & (~self.hasAllocatedAngles()),
            lambda self_dxdz, rhs_dxdz, self_dydz, rhs_dydz, sinds: (
                rhs_dxdz.at[sinds].get(),
                rhs_dydz.at[sinds].get(),
            ),
            lambda self_dxdz, rhs_dxdz, self_dydz, rhs_dydz, sinds: (
                self_dxdz,
                self_dydz,
            ),
            self.dxdz,
            rhs.dxdz,
            self.dydz,
            rhs.dydz,
            sinds,
        )

        self.wavelength = jax.lax.cond(
            rhs.hasAllocatedWavelengths() & (~self.hasAllocatedWavelengths()),
            lambda self_wave, rhs_wave, sinds: rhs_wave.at[sinds].get(),
            lambda self_wave, rhs_wave, sinds: self_wave,
            self.wavelength,
            rhs.wavelength,
            sinds,
        )

        self.pupil_u, self.pupil_v = jax.lax.cond(
            rhs.hasAllocatedPupil() & (~self.hasAllocatedPupil()),
            lambda self_pupil_u, rhs_pupil_u, self_pupil_v, rhs_pupil_v, sinds: (
                rhs_pupil_u.at[sinds].get(),
                rhs_pupil_v.at[sinds].get(),
            ),
            lambda self_pupil_u, rhs_pupil_u, self_pupil_v, rhs_pupil_v, sinds: (
                self_pupil_u,
                self_pupil_v,
            ),
            self.pupil_u,
            rhs.pupil_u,
            self.pupil_v,
            rhs.pupil_v,
            sinds,
        )

        self.time = jax.lax.cond(
            rhs.hasAllocatedTimes() & (~self.hasAllocatedTimes()),
            lambda self_time, rhs_time, sinds: rhs_time.at[sinds].get(),
            lambda self_time, rhs_time, sinds: self_time,
            self.time,
            rhs.time,
            sinds,
        )

        self._is_corr = self._is_corr | rhs._is_corr

        self._x = self._x + rhs._x.at[sinds].get()
        self._y = self._y + rhs._y.at[sinds].get()
        self._flux = self._flux * rhs._flux.at[sinds].get() * self.size()

        # sort the arrays back to their original order
        self._set_self_at_inds(self_sinds)
        rhs._set_self_at_inds(rhs_sinds)

        return self

    def __repr__(self):
        import numpy as np

        s = "galsim.PhotonArray(%r, x=array(%r), y=array(%r), flux=array(%r)" % (
            cast_to_python_int(self.size()),
            np.array(self.x).tolist(),
            np.array(self.y).tolist(),
            np.array(self.flux).tolist(),
        )
        if self.hasAllocatedAngles():
            s += ", dxdz=array(%r), dydz=array(%r)" % (
                np.array(self.dxdz).tolist(),
                np.array(self.dydz).tolist(),
            )
        if self.hasAllocatedWavelengths():
            s += ", wavelength=array(%r)" % (np.array(self.wavelength).tolist())
        if self.hasAllocatedPupil():
            s += ", pupil_u=array(%r), pupil_v=array(%r)" % (
                np.array(self.pupil_u).tolist(),
                np.array(self.pupil_v).tolist(),
            )
        if self.hasAllocatedTimes():
            s += ", time=array(%r)" % np.array(self.time).tolist()
        s += ", _nokeep=array(%r)" % np.array(self._nokeep).tolist()
        s += ")"
        return s

    def __str__(self):
        return "galsim.PhotonArray(%r)" % cast_to_python_int(self.size())

    __hash__ = None

    def __eq__(self, other):
        return self is other or (
            isinstance(other, PhotonArray)
            and jnp.array_equal(self.x, other.x)
            and jnp.array_equal(self.y, other.y)
            and jnp.array_equal(self.flux, other.flux)
            and jnp.array_equal(self._nokeep, other._nokeep)
            and jnp.array_equal(self.dxdz, other.dxdz, equal_nan=True)
            and jnp.array_equal(self.dydz, other.dydz, equal_nan=True)
            and jnp.array_equal(self.wavelength, other.wavelength, equal_nan=True)
            and jnp.array_equal(self.pupil_u, other.pupil_u, equal_nan=True)
            and jnp.array_equal(self.pupil_v, other.pupil_v, equal_nan=True)
            and jnp.array_equal(self.time, other.time, equal_nan=True)
        )

    def __ne__(self, other):
        return not self == other

    @_wraps(
        _galsim.PhotonArray.addTo,
        lax_description="The JAX equivalent of galsim.PhotonArray.addTo may not raise for undefined bounds.",
    )
    def addTo(self, image):
        if not image.bounds.isDefined():
            raise GalSimUndefinedBoundsError(
                "Attempting to PhotonArray::addTo an Image with undefined Bounds"
            )

        _arr, _flux_sum = _add_photons_to_image(
            self._x,
            self._y,
            # this computation is the same as self.flux, but we've left it duplicated here
            # so that we don't change this line to self._flux only by accident in the future
            jnp.where(self._nokeep, 0.0, self._flux) * self._Ntot / self._num_keep,
            image.bounds.xmin,
            image.bounds.ymin,
            image._array,
        )
        image._array = _arr

        return _flux_sum

    @classmethod
    def makeFromImage(cls, image, max_flux=1.0, rng=None):
        """Turn an existing `Image` into a `PhotonArray` that would accumulate into this image.

        The flux in each non-zero pixel will be turned into 1 or more photons with random positions
        within the pixel bounds.  The ``max_flux`` parameter (which defaults to 1) sets an upper
        limit for the absolute value of the flux of any photon.  Pixels with abs values > maxFlux
        will spawn multiple photons.

        Parameters:
            image:      The image to turn into a `PhotonArray`
            max_flux:   The maximum flux value to use for any output photon [default: 1]
            rng:        A `BaseDeviate` to use for the random number generation [default: None]

        Returns:
            a `PhotonArray`
        """

        if max_flux <= 0:
            raise GalSimRangeError("max_flux must be positive", max_flux, 0.0)

        n_per = jnp.clip(jnp.ceil(jnp.abs(image.array) / max_flux), 1).astype(int)
        flux_per = (image.array / n_per).ravel()
        n_per = n_per.ravel()
        inds = jnp.arange(image.array.size)
        inds = jnp.repeat(inds, n_per)
        yinds, xinds = jnp.unravel_index(inds, image.array.shape)

        xedges = jnp.arange(image.bounds.xmin, image.bounds.xmax + 2) - 0.5
        yedges = jnp.arange(image.bounds.ymin, image.bounds.ymax + 2) - 0.5

        # now we draw the position within the pixel
        ud = UniformDeviate(rng)
        photons = cls(n_per.sum())
        photons.x = ud.generate(photons.x) + xedges[xinds]
        photons.y = ud.generate(photons.y) + yedges[yinds]
        photons.flux = flux_per[inds]

        if image.scale is not None:
            photons.scaleXY(image.scale)

        return photons

    def write(self, file_name):
        """Write a `PhotonArray` to a FITS file.

        The output file will be a FITS binary table with a row for each photon in the `PhotonArray`.
        Columns will include 'id' (sequential from 1 to nphotons), 'x', 'y', and 'flux'.
        Additionally, the columns 'dxdz', 'dydz', and 'wavelength' will be included if they are
        set for this `PhotonArray` object.

        The file can be read back in with the classmethod `PhotonArray.read`::

            >>> photons.write('photons.fits')
            >>> photons2 = galsim.PhotonArray.read('photons.fits')

        Parameters:
            file_name:  The file name of the output FITS file.
        """
        import numpy as np

        from jax_galsim import fits

        cols = []
        cols.append(pyfits.Column(name="id", format="J", array=range(self.size())))
        cols.append(pyfits.Column(name="x", format="D", array=np.array(self.x)))
        cols.append(pyfits.Column(name="y", format="D", array=np.array(self.y)))
        cols.append(pyfits.Column(name="flux", format="D", array=np.array(self.flux)))
        cols.append(
            pyfits.Column(name="_nokeep", format="L", array=np.array(self._nokeep))
        )

        if self.hasAllocatedAngles():
            cols.append(
                pyfits.Column(name="dxdz", format="D", array=np.array(self.dxdz))
            )
            cols.append(
                pyfits.Column(name="dydz", format="D", array=np.array(self.dydz))
            )

        if self.hasAllocatedWavelengths():
            cols.append(
                pyfits.Column(
                    name="wavelength", format="D", array=np.array(self.wavelength)
                )
            )

        if self.hasAllocatedPupil():
            cols.append(
                pyfits.Column(name="pupil_u", format="D", array=np.array(self.pupil_u))
            )
            cols.append(
                pyfits.Column(name="pupil_v", format="D", array=np.array(self.pupil_v))
            )

        if self.hasAllocatedTimes():
            cols.append(
                pyfits.Column(name="time", format="D", array=np.array(self.time))
            )

        cols = pyfits.ColDefs(cols)
        table = pyfits.BinTableHDU.from_columns(cols)
        fits.writeFile(file_name, table)

    @classmethod
    def read(cls, file_name):
        """Create a `PhotonArray`, reading the photon data from a FITS file.

        The file being read in is not arbitrary.  It is expected to be a file that was written
        out with the `PhotonArray.write` method.::

            >>> photons.write('photons.fits')
            >>> photons2 = galsim.PhotonArray.read('photons.fits')

        Parameters:
            file_name:  The file name of the input FITS file.
        """
        with pyfits.open(file_name) as fits:
            data = fits[1].data
        N = len(data)
        names = data.columns.names

        photons = cls(
            N,
            x=jnp.array(data["x"]),
            y=jnp.array(data["y"]),
            flux=jnp.array(data["flux"]),
        )
        photons._nokeep = jnp.array(data["_nokeep"])
        if "dxdz" in names:
            photons.dxdz = jnp.array(data["dxdz"])
            photons.dydz = jnp.array(data["dydz"])
        if "wavelength" in names:
            photons.wavelength = jnp.array(data["wavelength"])
        if "pupil_u" in names:
            photons.pupil_u = jnp.array(data["pupil_u"])
            photons.pupil_v = jnp.array(data["pupil_v"])
        if "time" in names:
            photons.time = jnp.array(data["time"])
        return photons


@jax.jit
def _add_photons_to_image(x, y, flux, xmin, ymin, arr):
    xinds = jnp.floor(x - xmin + 0.5).astype(int)
    yinds = jnp.floor(y - ymin + 0.5).astype(int)
    # the jax documentation says that they drop out of bounds indices,
    # but the galsim unit tests reveal that without the check below,
    # the indices are not dropped.
    # I think maybe it is only indices beyond the end of the array that are
    # dropped and negative indices wrap around
    good = (xinds >= 0) & (xinds < arr.shape[1]) & (yinds >= 0) & (yinds < arr.shape[0])
    _flux = jnp.where(good, flux, 0.0)
    _arr = arr.at[yinds, xinds].add(_flux.astype(arr.dtype))

    return _arr, _flux.sum()
