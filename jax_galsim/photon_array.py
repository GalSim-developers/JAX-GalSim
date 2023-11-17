import galsim as _galsim
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


@_wraps(
    _galsim.PhotonArray,
    lax_description="""\
JAX-GalSim PhotonArrays have significant differences from the original GalSim.

    - They always copy input data and operations on them always copy.
    - They (usually) do not do any type/size checking on input data.
    - They do not support indexed assignement directly on the attributes.
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
    ):
        self._N = N

        # Only x, y, flux are built by default, since these are always required.
        # The others we leave as None unless/until they are needed.
        self._x = jnp.zeros(self._N, dtype=float)
        self._y = jnp.zeros(self._N, dtype=float)
        self._flux = jnp.zeros(self._N, dtype=float)
        self._dxdz = None
        self._dydz = None
        self._wave = None
        self._pupil_u = None
        self._pupil_v = None
        self._time = None
        self._is_corr = False

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

    @_wraps(
        _galsim.PhotonArray.fromArrays,
        lax_description="JAX-GalSim does not do input type/size checking.",
    )
    @classmethod
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

    @_wraps(_galsim.PhotonArray._fromArrays)
    @classmethod
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
        ret = cls.__new__(cls)
        ret._N = x.shape[0]
        ret._x = x
        ret._y = y
        ret._flux = flux
        ret._dxdz = dxdz
        ret._dydz = dydz
        ret._wave = wavelength
        ret._pupil_u = pupil_u
        ret._pupil_v = pupil_v
        ret._time = time
        ret._is_corr = is_corr
        return ret

    def tree_flatten(self):
        children = (
            (self.x, self.y, self.flux),
            {
                "dxdz": self.dxdz,
                "dydz": self.dydz,
                "wavelength": self.wavelength,
                "pupil_u": self.pupil_u,
                "pupil_v": self.pupil_v,
                "time": self.time,
                "is_corr": self.isCorrelated(),
            },
        )
        aux_data = (self._N,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        ret = cls.__new__(cls)
        ret._N = aux_data[0]
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
        return self._N

    def __len__(self):
        return self._N

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
        return self._flux

    @flux.setter
    def flux(self, value):
        self._flux = self._flux.at[:].set(value)

    @property
    def dxdz(self):
        """The tangent of the inclination angles in the x direction: dx/dz."""
        return self._dxdz

    @dxdz.setter
    def dxdz(self, value):
        self.allocateAngles()
        self._dxdz = self._dxdz.at[:].set(value)

    @property
    def dydz(self):
        """The tangent of the inclination angles in the y direction: dy/dz."""
        return self._dydz

    @dydz.setter
    def dydz(self, value):
        self.allocateAngles()
        self._dydz = self._dydz.at[:].set(value)

    @property
    def wavelength(self):
        """The wavelength of the photons (in nm)."""
        return self._wave

    @wavelength.setter
    def wavelength(self, value):
        self.allocateWavelengths()
        self._wave = self._wave.at[:].set(value)

    @property
    def pupil_u(self):
        """Horizontal location of photon as it intersected the entrance pupil plane."""
        return self._pupil_u

    @pupil_u.setter
    def pupil_u(self, value):
        self.allocatePupil()
        self._pupil_u = self._pupil_u.at[:].set(value)

    @property
    def pupil_v(self):
        """Vertical location of photon as it intersected the entrance pupil plane."""
        return self._pupil_v

    @pupil_v.setter
    def pupil_v(self, value):
        self.allocatePupil()
        self._pupil_v = self._pupil_v.at[:].set(value)

    @property
    def time(self):
        """Time stamp of when photon encounters the pupil plane."""
        return self._time

    @time.setter
    def time(self, value):
        self.allocateTimes()
        self._time = self._time.at[:].set(value)

    def hasAllocatedAngles(self):
        """Returns whether the arrays for the incidence angles `dxdz` and `dydz` have been
        allocated.
        """
        return self._dxdz is not None and self._dydz is not None

    def allocateAngles(self):
        """Allocate memory for the incidence angles, `dxdz` and `dydz`."""
        if not self.hasAllocatedAngles():
            self._dxdz = jnp.zeros(self._N, dtype=float)
            self._dydz = jnp.zeros(self._N, dtype=float)

    def hasAllocatedWavelengths(self):
        """Returns whether the `wavelength` array has been allocated."""
        return self._wave is not None

    def allocateWavelengths(self):
        """Allocate the memory for the `wavelength` array."""
        if not self.hasAllocatedWavelengths():
            self._wave = jnp.zeros(self._N, dtype=float)

    def hasAllocatedPupil(self):
        """Returns whether the arrays for the pupil coordinates `pupil_u` and `pupil_v` have been
        allocated.
        """
        return self._pupil_u is not None and self._pupil_v is not None

    def allocatePupil(self):
        """Allocate the memory for the pupil coordinates, `pupil_u` and `pupil_v`."""
        if not self.hasAllocatedPupil():
            self._pupil_u = jnp.zeros(self._N, dtype=float)
            self._pupil_v = jnp.zeros(self._N, dtype=float)

    def hasAllocatedTimes(self):
        """Returns whether the array for the time stamps `time` has been allocated."""
        return self._time is not None

    def allocateTimes(self):
        """Allocate the memory for the time stamps, `time`."""
        if not self.hasAllocatedTimes():
            self._time = jnp.zeros(self._N, dtype=float)

    def isCorrelated(self):
        """Returns whether the photons are correlated"""
        return self._is_corr

    def setCorrelated(self, is_corr=True):
        """Set whether the photons are correlated"""
        self._is_corr = is_corr

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

    def assignAt(self, istart, rhs):
        """Assign the contents of another `PhotonArray` to this one starting at istart."""
        if istart + rhs.size() > self.size():
            raise GalSimValueError(
                "The given rhs does not fit into this array starting at %d" % istart,
                rhs,
            )
        self._x = self._x.at[istart : istart + rhs.size()].set(rhs.x)
        self._y = self._y.at[istart : istart + rhs.size()].set(rhs.y)
        self._flux = self._flux.at[istart : istart + rhs.size()].set(rhs.flux)
        if rhs.hasAllocatedAngles():
            self.allocateAngles()
            self._dxdz = self._dxdz.at[istart : istart + rhs.size()].set(rhs.dxdz)
            self._dydz = self._dydz.at[istart : istart + rhs.size()].set(rhs.dydz)
        if rhs.hasAllocatedWavelengths():
            self.allocateWavelengths()
            self._wave = self._wave.at[istart : istart + rhs.size()].set(rhs.wavelength)
        if rhs.hasAllocatedPupil():
            self.allocatePupil()
            self._pupil_u = self._pupil_u.at[istart : istart + rhs.size()].set(
                rhs.pupil_u
            )
            self._pupil_v = self._pupil_v.at[istart : istart + rhs.size()].set(
                rhs.pupil_v
            )
        if rhs.hasAllocatedTimes():
            self.allocateTimes()
            self._time = self._time.at[istart : istart + rhs.size()].set(rhs.time)

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

        if rhs.isCorrelated() and self.isCorrelated():
            rng = BaseDeviate(rng)
            subkey = rng._state.split_one()
            sinds = jrng.choice(
                subkey,
                self.size(),
                shape=(self.size(),),
                replace=False,
            )
        else:
            sinds = jnp.arange(self.size())

        if rhs.hasAllocatedAngles() and not self.hasAllocatedAngles():
            self.dxdz = rhs.dxdz[sinds]
            self.dydz = rhs.dydz[sinds]

        if rhs.hasAllocatedWavelengths() and not self.hasAllocatedWavelengths():
            self.wavelength = rhs.wavelength

        if rhs.hasAllocatedPupil() and not self.hasAllocatedPupil():
            self.pupil_u = rhs.pupil_u[sinds]
            self.pupil_v = rhs.pupil_v[sinds]

        if rhs.hasAllocatedTimes() and not self.hasAllocatedTimes():
            self.time = rhs.time[sinds]

        if rhs.isCorrelated():
            self.setCorrelated()

        self._x = self._x + rhs._x[sinds]
        self._y = self._y + rhs._y[sinds]
        self._flux = self._flux * rhs._flux[sinds] * self.size()

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
            and self.hasAllocatedAngles() == other.hasAllocatedAngles()
            and self.hasAllocatedWavelengths() == other.hasAllocatedWavelengths()
            and self.hasAllocatedPupil() == other.hasAllocatedPupil()
            and self.hasAllocatedTimes() == other.hasAllocatedTimes()
            and (
                jnp.array_equal(self.dxdz, other.dxdz)
                if self.hasAllocatedAngles()
                else True
            )
            and (
                jnp.array_equal(self.dydz, other.dydz)
                if self.hasAllocatedAngles()
                else True
            )
            and (
                jnp.array_equal(self.wavelength, other.wavelength)
                if self.hasAllocatedWavelengths()
                else True
            )
            and (
                jnp.array_equal(self.pupil_u, other.pupil_u)
                if self.hasAllocatedPupil()
                else True
            )
            and (
                jnp.array_equal(self.pupil_v, other.pupil_v)
                if self.hasAllocatedPupil()
                else True
            )
            and (
                jnp.array_equal(self.time, other.time)
                if self.hasAllocatedTimes()
                else True
            )
        )

    def __ne__(self, other):
        return not self == other

    def addTo(self, image):
        """Add flux of photons to an image by binning into pixels.

        Photons in this `PhotonArray` are binned into the pixels of the input
        `Image` and their flux summed into the pixels.  The `Image` is assumed to represent
        surface brightness, so photons' fluxes are divided by image pixel area.
        Photons past the edges of the image are discarded.

        Parameters:
            image:      The `Image` to which the photons' flux will be added.

        Returns:
            the total flux of photons the landed inside the image bounds.
        """
        if not image.bounds.isDefined():
            raise GalSimUndefinedBoundsError(
                "Attempting to PhotonArray::addTo an Image with undefined Bounds"
            )
        xinds = jnp.floor(self._x - image.bounds.xmin).astype(int)
        yinds = jnp.floor(self._y - image.bounds.ymin).astype(int)
        image._array = image._array.at[yinds, xinds].add(self._flux)

        return self._flux.sum()

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
