import copy
import os

import galsim as _galsim
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from jax_galsim import fits
from jax_galsim.angle import AngleUnit, arcsec, degrees, radians
from jax_galsim.celestial import CelestialCoord
from jax_galsim.core.utils import (
    cast_to_float,
    cast_to_python_float,
    ensure_hashable,
    implements,
)
from jax_galsim.errors import (
    GalSimError,
    GalSimIncompatibleValuesError,
    GalSimNotImplementedError,
    GalSimValueError,
    galsim_warn,
)
from jax_galsim.position import PositionD
from jax_galsim.utilities import horner2d
from jax_galsim.wcs import (
    AffineTransform,
    CelestialWCS,
    JacobianWCS,
    OffsetWCS,
    PixelScale,
)

#########################################################################################
#
# We have the following WCS classes that know how to read the WCS from a FITS file:
#
#     GSFitsWCS
#
# As for all CelestialWCS classes, they must define the following:
#
#     _radec            function returning (ra, dec) in _radians_ at position (x,y)
#     _xy               function returning (x, y) given (ra, dec) in _radians_.
#     _writeHeader      function that writes the WCS to a fits header.
#     _readHeader       static function that reads the WCS from a fits header.
#     copy              return a copy
#     __eq__            check if this equals another WCS
#
#########################################################################################


@implements(
    _galsim.fitswcs.GSFitsWCS,
    lax_description=(
        "The JAX-GalSim version of this class does not raise errors if inverting the WCS to "
        "map ra,dec to (x,y) fails. Instead it returns NaNs."
    ),
)
@register_pytree_node_class
class GSFitsWCS(CelestialWCS):
    _req_params = {"file_name": str}
    _opt_params = {"dir": str, "hdu": int, "origin": PositionD, "compression": str}

    def __init__(
        self,
        file_name=None,
        dir=None,
        hdu=None,
        header=None,
        compression="auto",
        origin=None,
        _data=None,
    ):
        # Note: _data is not intended for end-user use.  It enables the equivalent of a
        #       private constructor of GSFitsWCS by the function TanWCS.  The details of its
        #       use are intentionally not documented above.

        self._color = None
        self._tag = None  # Write something useful here (see below). This is just used for the str.

        # If _data is given, copy the data and we're done.
        if _data is not None:
            self.wcs_type = _data[0]
            self.crpix = _data[1]
            self.cd = _data[2]
            self.center = _data[3]
            self.pv = _data[4]
            self.ab = _data[5]
            self.abp = _data[6]
            if self.wcs_type in ("TAN", "TPV", "TNX", "TAN-SIP"):
                self.projection = "gnomonic"
            elif self.wcs_type in ("STG", "STG-SIP"):
                self.projection = "stereographic"
            elif self.wcs_type in ("ZEA", "ZEA-SIP"):
                self.projection = "lambert"
            elif self.wcs_type in ("ARC", "ARC-SIP"):
                self.projection = "postel"
            else:
                raise ValueError("Invalid wcs_type in _data")

            # set cdinv and convert to jax
            self.cd = jnp.array(self.cd)
            self.crpix = jnp.array(self.crpix)
            if self.pv is not None:
                self.pv = jnp.array(self.pv)
            if self.ab is not None:
                self.ab = jnp.array(self.ab)
            if self.abp is not None:
                self.abp = jnp.array(self.abp)
            self.cdinv = jnp.linalg.inv(self.cd)
            return

        # Read the file if given.
        if file_name is not None:
            if dir is not None:
                self._tag = repr(os.path.join(dir, file_name))
            else:
                self._tag = repr(file_name)
            if hdu is not None:
                self._tag += ", hdu=%r" % hdu
            if compression != "auto":
                self._tag += ", compression=%r" % compression
            if header is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot provide both file_name and pyfits header",
                    file_name=file_name,
                    header=header,
                )
            hdu, hdu_list, fin = fits.readFile(file_name, dir, hdu, compression)

        try:
            if file_name is not None:
                header = hdu.header

            if header is None:
                raise GalSimIncompatibleValuesError(
                    "Must provide either file_name or header",
                    file_name=file_name,
                    header=header,
                )

            # Read the wcs information from the header.
            self._read_header(header)

        finally:
            if file_name is not None:
                fits.closeHDUList(hdu_list, fin)

        if origin is not None:
            self.crpix += jnp.array([origin.x, origin.y])

    def tree_flatten(self):
        """This function flattens the WCS into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        children = (
            self.crpix,
            self.cd,
            self.center,
            self.pv,
            self.ab,
            self.abp,
        )
        # Define auxiliary static data that doesn’t need to be traced
        aux_data = (self.wcs_type,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        return cls(_data=aux_data + children)

    # The origin is a required attribute/property, since it is used by some functions like
    # shiftOrigin to get the current origin value.  We don't use it in this class, though, so
    # just make origin a dummy property that returns 0,0.
    @property
    def origin(self):
        """The origin in image coordinates of the WCS function."""
        return PositionD(0.0, 0.0)

    def _read_header(self, header):
        # Start by reading the basic WCS stuff that most types have.
        ctype1 = header.get("CTYPE1", "")
        ctype2 = header.get("CTYPE2", "")
        if ctype1.startswith("DEC--") and ctype2.startswith("RA---"):
            flip = True
        elif ctype1.startswith("RA---") and ctype2.startswith("DEC--"):
            flip = False
        else:
            raise GalSimError(
                "GSFitsWCS only supports celestial coordinate systems."
                "Expecting CTYPE1,2 to start with RA--- and DEC--.  Got %s, %s"
                % (ctype1, ctype2)
            )
        if ctype1[5:] != ctype2[5:]:  # pragma: no cover
            raise OSError("ctype1, ctype2 do not seem to agree on the WCS type")
        self.wcs_type = ctype1[5:]
        if self.wcs_type in ("TAN", "TPV", "TNX", "TAN-SIP"):
            self.projection = "gnomonic"
        elif self.wcs_type in ("STG", "STG-SIP"):
            self.projection = "stereographic"
        elif self.wcs_type in ("ZEA", "ZEA-SIP"):
            self.projection = "lambert"
        elif self.wcs_type in ("ARC", "ARC-SIP"):
            self.projection = "postel"
        else:
            raise GalSimValueError(
                "GSFitsWCS cannot read files using given wcs_type.",
                self.wcs_type,
                (
                    "TAN",
                    "TPV",
                    "TNX",
                    "TAN-SIP",
                    "STG",
                    "STG-SIP",
                    "ZEA",
                    "ZEA-SIP",
                    "ARC",
                    "ARC-SIP",
                ),
            )
        crval1 = float(header["CRVAL1"])
        crval2 = float(header["CRVAL2"])
        crpix1 = float(header["CRPIX1"])
        crpix2 = float(header["CRPIX2"])
        if "CD1_1" in header:
            cd11 = float(header["CD1_1"])
            cd12 = float(header["CD1_2"])
            cd21 = float(header["CD2_1"])
            cd22 = float(header["CD2_2"])
        elif "CDELT1" in header:
            if "PC1_1" in header:
                cd11 = float(header["PC1_1"]) * float(header["CDELT1"])
                cd12 = float(header["PC1_2"]) * float(header["CDELT1"])
                cd21 = float(header["PC2_1"]) * float(header["CDELT2"])
                cd22 = float(header["PC2_2"]) * float(header["CDELT2"])
            else:
                cd11 = float(header["CDELT1"])
                cd12 = 0.0
                cd21 = 0.0
                cd22 = float(header["CDELT2"])
        else:  # pragma: no cover  (all our test files have either CD or CDELT)
            cd11 = 1.0
            cd12 = 0.0
            cd21 = 0.0
            cd22 = 1.0

        # Usually the units are degrees, but make sure
        if "CUNIT1" in header:
            cunit1 = header["CUNIT1"]
            cunit2 = header["CUNIT2"]
            ra_units = AngleUnit.from_name(cunit1)
            dec_units = AngleUnit.from_name(cunit2)
        else:
            ra_units = degrees
            dec_units = degrees

        if flip:
            crval1, crval2 = crval2, crval1
            ra_units, dec_units = dec_units, ra_units
            cd11, cd21 = cd21, cd11
            cd12, cd22 = cd22, cd12

        self.crpix = np.array([crpix1, crpix2])
        self.cd = np.array([[cd11, cd12], [cd21, cd22]])

        self.center = CelestialCoord(crval1 * ra_units, crval2 * dec_units)

        # There was an older proposed standard that used TAN with PV values, which is used by
        # SCamp, so we want to support it if possible.  The standard is now called TPV, so
        # use that for our wcs_type if we see the PV values with TAN.
        if self.wcs_type == "TAN" and "PV1_1" in header:
            self.wcs_type = "TPV"

        self.pv = None
        self.ab = None
        self.abp = None
        if self.wcs_type == "TPV":
            self._read_tpv(header)
        elif self.wcs_type == "TNX":
            self._read_tnx(header)
        elif self.wcs_type in ("TAN-SIP", "STG-SIP", "ZEA-SIP", "ARC-SIP"):
            self._read_sip(header)

        # I think the CUNIT specification applies to the CD matrix as well, but I couldn't actually
        # find good documentation for this.  Plus all the examples I saw used degrees anyway, so
        # it's hard to tell.  Hopefully this will never matter, but if CUNIT is not deg, this
        # next bit might be wrong.
        # I did see documentation that the PV matrices always use degrees, so at least we shouldn't
        # have to worry about that.
        if ra_units != degrees:  # pragma: no cover
            self.cd[0, :] *= 1.0 * ra_units / degrees
        if dec_units != degrees:  # pragma: no cover
            self.cd[1, :] *= 1.0 * dec_units / degrees

        # convert to JAX after reading
        self.cd = jnp.array(self.cd)
        self.crpix = jnp.array(self.crpix)
        self.cdinv = jnp.linalg.inv(self.cd)

    def _read_tpv(self, header):
        # See http://fits.gsfc.nasa.gov/registry/tpvwcs/tpv.html for details about how
        # the TPV standard is defined.

        # The standard includes an option to have odd powers of r, which kind of screws
        # up the numbering of these coefficients.  We don't implement these terms, so
        # before going further, check to make sure none are present.
        odd_indices = [3, 11, 23, 39]
        if any(
            (
                header.get("PV%s_%s" % (i, j), 0.0) != 0.0
                for i in [1, 2]
                for j in odd_indices
            )
        ):
            raise GalSimNotImplementedError("TPV not implemented for odd powers of r")

        pv1 = [
            float(header.get("PV1_%s" % k, 0.0))
            for k in range(40)
            if k not in odd_indices
        ]
        pv2 = [
            float(header.get("PV2_%s" % k, 0.0))
            for k in range(40)
            if k not in odd_indices
        ]

        maxk = max(np.nonzero(pv1)[0][-1], np.nonzero(pv2)[0][-1])
        # maxk = (order+1) * (order+2) / 2 - 1
        order = int(np.floor(np.sqrt(2 * (maxk + 1)))) - 1
        self.pv = np.zeros((2, order + 1, order + 1))

        # Another strange thing is that the two matrices are defined in the opposite order
        # with respect to their element ordering.  But at least now, without the odd terms,
        # we can just proceed in order in the k indices.  So what we call k=3..9 here were
        # originally PVi_4..10.
        # For reference, here is what it would look like for order = 3:
        # self.pv = np.array( [ [ [ pv1[0], pv1[2], pv1[5], pv1[9] ],
        #                         [ pv1[1], pv1[4], pv1[8],   0.   ],
        #                         [ pv1[3], pv1[7],   0.  ,   0.   ],
        #                         [ pv1[6],   0.  ,   0.  ,   0.   ] ],
        #                       [ [ pv2[0], pv2[1], pv2[3], pv2[6] ],
        #                         [ pv2[2], pv2[4], pv2[7],   0.   ],
        #                         [ pv2[5], pv2[8],   0.  ,   0.   ],
        #                         [ pv2[9],   0.  ,   0.  ,   0.   ] ] ] )
        k = 0
        for N in range(order + 1):
            for j in range(N + 1):
                i = N - j
                self.pv[0, i, j] = pv1[k]
                self.pv[1, j, i] = pv2[k]
                k = k + 1

        # convert to JAX after reading
        self.pv = jnp.array(self.pv)

    def _read_sip(self, header):
        a_order = int(header["A_ORDER"])
        b_order = int(header["B_ORDER"])
        order = max(a_order, b_order)  # Use the same order for both
        a = [
            float(header.get("A_" + str(i) + "_" + str(j), 0.0))
            for i in range(order + 1)
            for j in range(order + 1)
        ]
        a = np.array(a).reshape((order + 1, order + 1))
        b = [
            float(header.get("B_" + str(i) + "_" + str(j), 0.0))
            for i in range(order + 1)
            for j in range(order + 1)
        ]
        b = np.array(b).reshape((order + 1, order + 1))
        a[1, 0] += (
            1  # Standard A,B are a differential calculation.  It's more convenient to
        )
        b[0, 1] += 1  # keep this as an absolute calculation like PV does.
        self.ab = np.array([a, b])

        # The reverse transformation is not required to be there.
        if "AP_ORDER" in header:
            ap_order = int(header["AP_ORDER"])
            bp_order = int(header["BP_ORDER"])
            order = max(ap_order, bp_order)  # Use the same order for both
            ap = [
                float(header.get("AP_" + str(i) + "_" + str(j), 0.0))
                for i in range(order + 1)
                for j in range(order + 1)
            ]
            ap = np.array(ap).reshape((order + 1, order + 1))
            bp = [
                float(header.get("BP_" + str(i) + "_" + str(j), 0.0))
                for i in range(order + 1)
                for j in range(order + 1)
            ]
            bp = np.array(bp).reshape((order + 1, order + 1))
            ap[1, 0] += 1
            bp[0, 1] += 1
            self.abp = np.array([ap, bp])

            # convert to JAX after reading
            self.abp = jnp.array(self.abp)

        # convert to JAX after reading
        self.ab = jnp.array(self.ab)

    def _read_tnx(self, header):
        # TNX has a few different options.  Rather than keep things in the native format,
        # we actually convert to the equivalent of TPV to make the actual operations faster.
        # See http://iraf.noao.edu/projects/ccdmosaic/tnx.html for details.

        # First, parse the input values, which are stored in WAT keywords:
        k = 1
        wat1 = ""
        key = "WAT1_%03d" % k
        while key in header:
            wat1 += header[key]
            k = k + 1
            key = "WAT1_%03d" % k
        wat1 = wat1.split()

        k = 1
        wat2 = ""
        key = "WAT2_%03d" % k
        while key in header:
            wat2 += header[key]
            k = k + 1
            key = "WAT2_%03d" % k
        wat2 = wat2.split()

        if (
            len(wat1) < 12
            or wat1[0] != "wtype=tnx"
            or wat1[1] != "axtype=ra"
            or wat1[2] != "lngcor"
            or wat1[3] != "="
            or not wat1[4].startswith('"')
            or not wat1[-1].endswith('"')
        ):  # pragma: no cover
            raise GalSimError("TNX WAT1 was not as expected")
        if (
            len(wat2) < 12
            or wat2[0] != "wtype=tnx"
            or wat2[1] != "axtype=dec"
            or wat2[2] != "latcor"
            or wat2[3] != "="
            or not wat2[4].startswith('"')
            or not wat2[-1].endswith('"')
        ):  # pragma: no cover
            raise GalSimError("TNX WAT2 was not as expected")

        # Break the next bit out into another function, since it is the same for x and y.
        pv1 = self._parse_tnx_data(wat1[4:])
        pv2 = self._parse_tnx_data(wat2[4:])

        # Those just give the adjustments to the position, not the matrix that gives the final
        # position.  i.e. the TNX standard uses u = u + [1 u u^2 u^3] PV [1 v v^2 v^3]T.
        # So we need to add 1 to the correct term in each matrix to get what we really want.
        pv1[1, 0] += 1.0
        pv2[0, 1] += 1.0

        # Finally, store these as our pv 3-d array.
        self.pv = np.array([pv1, pv2])

        # We've now converted this to TPV, so call it that when we output to a fits header.
        self.wcs_type = "TPV"

        # convert to JAX after reading
        self.pv = jnp.array(self.pv)

    def _parse_tnx_data(self, data):
        # I'm not sure if there is any requirement on there being a space before the final " and
        # not before the initial ".  But both the example in the description of the standard and
        # the one we have in our test directory are this way.  Here, if the " is by itself, I
        # remove the item, and if it is part of a longer string, I just strip it off.  Seems the
        # most sensible thing to do.
        if data[0] == '"':  # pragma: no cover
            data = data[1:]
        else:
            data[0] = data[0][1:]
        if data[-1] == '"':
            data = data[:-1]
        else:  # pragma: no cover
            data[-1] = data[-1][:-1]

        code = int(
            data[0].strip(".")
        )  # Weirdly, these integers are given with decimal points.
        xorder = int(data[1].strip("."))
        yorder = int(data[2].strip("."))
        cross = int(data[3].strip("."))
        if cross != 2:  # pragma: no cover
            raise GalSimNotImplementedError(
                "TNX only implemented for half-cross option."
            )
        if xorder != 4 or yorder != 4:  # pragma: no cover
            raise GalSimNotImplementedError("TNX only implemented for order = 4")
        # Note: order = 4 really means cubic.  order is how large the pv matrix is, i.e. 4x4.

        xmin = float(data[4])
        xmax = float(data[5])
        ymin = float(data[6])
        ymax = float(data[7])

        pv1 = [float(x) for x in data[8:]]
        if len(pv1) != 10:  # pragma: no cover
            raise GalSimError("Wrong number of items found in WAT data")

        # Put these into our matrix formulation.
        pv = np.array(
            [
                [pv1[0], pv1[4], pv1[7], pv1[9]],
                [pv1[1], pv1[5], pv1[8], 0.0],
                [pv1[2], pv1[6], 0.0, 0.0],
                [pv1[3], 0.0, 0.0, 0.0],
            ]
        )

        # Convert from Legendre or Chebyshev polynomials into regular polynomials.
        if code < 3:  # pragma: no branch (The only test file I can find has code = 1)
            # Instead of 1, x, x^2, x^3, Chebyshev uses: 1, x', 2x'^2 - 1, 4x'^3 - 3x
            # where x' = (2x - xmin - xmax) / (xmax-xmin).
            # Similarly, with y' = (2y - ymin - ymin) / (ymax-ymin)
            # We'd like to convert the pv matrix from being in terms of x' and y' to being
            # in terms of just x, y.  To see how this works, look at what pv[1,1] means:
            #
            # First, let's say we can write x as (a + bx), and we can write y' as (c + dy).
            # Then the term for pv[1,1] is:
            #
            # term = x' * pv[1,1] * y'
            #      = (a + bx) * pv[1,1] * (d + ey)
            #      =       a * pv[1,1] * c  +      a * pv[1,1] * d * y
            #        + x * b * pv[1,1] * c  +  x * b * pv[1,1] * d * y
            #
            # So the single term initially will contribute to 4 different terms in the final
            # matrix.  And the contributions will just be pv[1,1] times the outer product
            # [a b]T [d e].  So if we can determine the matrix that converts from
            # [1, x, x^2, x^3] to the Chebyshev vector, the the matrix we want is simply
            # xmT pv ym.
            a = -(xmax + xmin) / (xmax - xmin)
            b = 2.0 / (xmax - xmin)
            c = -(ymax + ymin) / (ymax - ymin)
            d = 2.0 / (ymax - ymin)
            xm = np.zeros((4, 4))
            ym = np.zeros((4, 4))
            xm[0, 0] = 1.0
            xm[1, 0] = a
            xm[1, 1] = b
            ym[0, 0] = 1.0
            ym[1, 0] = c
            ym[1, 1] = d
            if code == 1:
                for m in range(2, 4):
                    # The recursion rule is Pm = 2 x' Pm-1 - Pm-2
                    # Pm = 2 a Pm-1 - Pm-2 + x * 2 b Pm-1
                    xm[m] = 2.0 * a * xm[m - 1] - xm[m - 2]
                    xm[m, 1:] += 2.0 * b * xm[m - 1, :-1]
                    ym[m] = 2.0 * c * ym[m - 1] - ym[m - 2]
                    ym[m, 1:] += 2.0 * d * ym[m - 1, :-1]
            else:  # pragma: no cover
                # code == 2 means Legendre.  The same argument applies, but we have a
                # different recursion rule.
                # WARNING: This branch has not been tested!  I don't have any TNX files
                # with Legendre functions to test it on.  I think it's right, but beware!
                for m in range(2, 4):
                    # The recursion rule is Pm = ((2m-1) x' Pm-1 - (m-1) Pm-2) / m
                    # Pm = ((2m-1) a Pm-1 - (m-1) Pm-2) / m
                    #      + x * ((2m-1) b Pm-1) / m
                    xm[m] = (
                        (2.0 * m - 1.0) * a * xm[m - 1] - (m - 1.0) * xm[m - 2]
                    ) / m
                    xm[m, 1:] += ((2.0 * m - 1.0) * b * xm[m - 1, :-1]) / m
                    ym[m] = (
                        (2.0 * m - 1.0) * c * ym[m - 1] - (m - 1.0) * ym[m - 2]
                    ) / m
                    ym[m, 1:] += ((2.0 * m - 1.0) * d * ym[m - 1, :-1]) / m

            pv2 = np.dot(xm.T, np.dot(pv, ym))
            return pv2

    def _apply_ab(self, x, y, ab):
        # Note: this is used for both pv and ab, since the action is the same.
        # They just occur at two different places in the calculation.
        x1 = horner2d(x, y, ab[0], triangle=True)
        y1 = horner2d(x, y, ab[1], triangle=True)
        return x1, y1

    def _uv(self, x, y):
        # Most of the work for _radec.  But stop at (u,v).

        # Start with (u,v) = the image position
        x = cast_to_float(x)
        y = cast_to_float(y)

        x -= self.crpix[0]
        y -= self.crpix[1]

        if self.ab is not None:
            x, y = self._apply_ab(x, y, self.ab)

        # This converts to (u,v) in the tangent plane
        # Expanding this out is a bit faster than using np.dot for 2x2 matrix.
        # This is a bit faster than using np.dot for 2x2 matrix.
        u = self.cd[0, 0] * x + self.cd[0, 1] * y
        v = self.cd[1, 0] * x + self.cd[1, 1] * y

        if self.pv is not None:
            u, v = self._apply_ab(u, v, self.pv)

        # Convert (u,v) from degrees to radians
        # Also, the FITS standard defines u,v backwards relative to our standard.
        # They have +u increasing to the east, not west.  Hence the - for u.
        factor = 1.0 * degrees / radians
        u *= -factor
        v *= factor
        return u, v

    def _radec(self, x, y, color=None):
        # Get the position in the tangent plane
        u, v = self._uv(x, y)
        # Then convert from (u,v) to (ra, dec) using the appropriate projection.
        ra, dec = self.center.deproject_rad(u, v, projection=self.projection)

        return ra, dec

    def _xy(self, ra, dec, color=None):
        u, v = self.center.project_rad(ra, dec, projection=self.projection)

        # Again, FITS has +u increasing to the east, not west.  Hence the - for u.
        factor = radians / degrees
        u *= -factor
        v *= factor

        if self.pv is not None:
            u, v = _invert_ab_noraise(u, v, self.pv)

        # This is a bit faster than using np.dot for 2x2 matrix.
        x = self.cdinv[0, 0] * u + self.cdinv[0, 1] * v
        y = self.cdinv[1, 0] * u + self.cdinv[1, 1] * v

        if self.ab is not None:
            x, y = _invert_ab_noraise(x, y, self.ab, abp=self.abp)

        x += self.crpix[0]
        y += self.crpix[1]

        return x, y

    # Override the version in CelestialWCS, since we can do this more efficiently.
    def _local(self, image_pos, color=None):
        if image_pos is None:
            raise TypeError("origin must be a PositionD or PositionI argument")

        # The key lemma here is that chain rule for jacobians is just matrix multiplication.
        # i.e. if s = s(u,v), t = t(u,v) and u = u(x,y), v = v(x,y), then
        # ( dsdx  dsdy ) = ( dsdu dudx + dsdv dvdx   dsdu dudy + dsdv dvdy )
        # ( dtdx  dtdy ) = ( dtdu dudx + dtdv dvdx   dtdu dudy + dtdv dvdy )
        #                = ( dsdu  dsdv )  ( dudx  dudy )
        #                  ( dtdu  dtdv )  ( dvdx  dvdy )
        #
        # So if we can find the jacobian for each step of the process, we just multiply the
        # jacobians.
        #
        # We also need to keep track of the position along the way, so we have to repeat many
        # of the steps in _radec.

        p1 = jnp.array([image_pos.x, image_pos.y], dtype=float)

        # Start with unit jacobian
        jac = jnp.eye(2)

        # No effect on the jacobian from this step.
        p1 -= self.crpix

        if self.ab is not None:
            x = p1[0]
            y = p1[1]
            order = len(self.ab[0]) - 1
            xpow = x ** jnp.arange(order + 1)
            ypow = y ** jnp.arange(order + 1)
            p1 = jnp.dot(jnp.dot(self.ab, ypow), xpow)

            dxpow = jnp.zeros(order + 1)
            dypow = jnp.zeros(order + 1)
            dxpow = dxpow.at[1:].set((jnp.arange(order) + 1.0) * xpow[:-1])
            dypow = dypow.at[1:].set((jnp.arange(order) + 1.0) * ypow[:-1])
            j1 = jnp.transpose(
                jnp.array(
                    [
                        jnp.dot(jnp.dot(self.ab, ypow), dxpow),
                        jnp.dot(jnp.dot(self.ab, dypow), xpow),
                    ]
                )
            )
            jac = jnp.dot(j1, jac)

        # The jacobian here is just the cd matrix.
        p2 = jnp.dot(self.cd, p1)
        jac = jnp.dot(self.cd, jac)

        if self.pv is not None:
            # Now we apply the distortion terms
            u = p2[0]
            v = p2[1]
            order = len(self.pv[0]) - 1

            upow = u ** jnp.arange(order + 1)
            vpow = v ** jnp.arange(order + 1)

            p2 = jnp.dot(jnp.dot(self.pv, vpow), upow)

            # The columns of the jacobian for this step are the same function with dupow
            # or dvpow.
            dupow = jnp.zeros(order + 1)
            dvpow = jnp.zeros(order + 1)
            dupow = dupow.at[1:].set((jnp.arange(order) + 1.0) * upow[:-1])
            dvpow = dvpow.at[1:].set((jnp.arange(order) + 1.0) * vpow[:-1])
            j1 = jnp.transpose(
                jnp.array(
                    [
                        jnp.dot(jnp.dot(self.pv, vpow), dupow),
                        jnp.dot(jnp.dot(self.pv, dvpow), upow),
                    ]
                )
            )
            jac = jnp.dot(j1, jac)

        unit_convert = jnp.array([-1 * degrees / radians, 1 * degrees / radians])
        p2 *= unit_convert
        # Subtle point: Don't use jac *= ..., because jac might currently be self.cd, and
        #               that would change self.cd!
        jac = jac * jnp.transpose(jnp.array([unit_convert]))

        # Finally convert from (u,v) to (ra, dec).  We have a special function that computes
        # the jacobian of this step in the CelestialCoord class.
        j2 = self.center.jac_deproject_rad(p2[0], p2[1], projection=self.projection)
        jac = jnp.dot(j2, jac)

        # This now has units of radians/pixel.  We want instead arcsec/pixel.
        jac *= radians / arcsec

        return JacobianWCS(jac[0, 0], jac[0, 1], jac[1, 0], jac[1, 1])

    def _newOrigin(self, origin):
        ret = self.copy()
        ret.crpix = ret.crpix + jnp.array([origin.x, origin.y])
        return ret

    def _writeHeader(self, header, bounds):
        header["GS_WCS"] = ("GSFitsWCS", "GalSim WCS name")
        header["CTYPE1"] = "RA---" + self.wcs_type
        header["CTYPE2"] = "DEC--" + self.wcs_type
        header["CRPIX1"] = cast_to_python_float(self.crpix[0])
        header["CRPIX2"] = cast_to_python_float(self.crpix[1])
        header["CD1_1"] = cast_to_python_float(self.cd[0][0])
        header["CD1_2"] = cast_to_python_float(self.cd[0][1])
        header["CD2_1"] = cast_to_python_float(self.cd[1][0])
        header["CD2_2"] = cast_to_python_float(self.cd[1][1])
        header["CUNIT1"] = "deg"
        header["CUNIT2"] = "deg"
        header["CRVAL1"] = cast_to_python_float(self.center.ra / degrees)
        header["CRVAL2"] = cast_to_python_float(self.center.dec / degrees)
        if self.pv is not None:
            order = len(self.pv[0]) - 1
            k = 0
            odd_indices = [3, 11, 23, 39]
            for n in range(order + 1):
                for j in range(n + 1):
                    i = n - j
                    header["PV1_" + str(k)] = cast_to_python_float(self.pv[0, i, j])
                    header["PV2_" + str(k)] = cast_to_python_float(self.pv[1, j, i])
                    k = k + 1
                    if k in odd_indices:
                        k = k + 1
        if self.ab is not None:
            order = len(self.ab[0]) - 1
            header["A_ORDER"] = order
            for i in range(order + 1):
                for j in range(order + 1):
                    aij = self.ab[0, i, j]
                    if i == 1 and j == 0:
                        aij -= 1  # Turn back into standard form.
                    if aij != 0.0:
                        header["A_" + str(i) + "_" + str(j)] = cast_to_python_float(aij)
            header["B_ORDER"] = order
            for i in range(order + 1):
                for j in range(order + 1):
                    bij = self.ab[1, i, j]
                    if i == 0 and j == 1:
                        bij -= 1
                    if bij != 0.0:
                        header["B_" + str(i) + "_" + str(j)] = cast_to_python_float(bij)
        if self.abp is not None:
            order = len(self.abp[0]) - 1
            header["AP_ORDER"] = order
            for i in range(order + 1):
                for j in range(order + 1):
                    apij = self.abp[0, i, j]
                    if i == 1 and j == 0:
                        apij -= 1
                    if apij != 0.0:
                        header["AP_" + str(i) + "_" + str(j)] = cast_to_python_float(
                            apij
                        )
            header["BP_ORDER"] = order
            for i in range(order + 1):
                for j in range(order + 1):
                    bpij = self.abp[1, i, j]
                    if i == 0 and j == 1:
                        bpij -= 1
                    if bpij != 0.0:
                        header["BP_" + str(i) + "_" + str(j)] = cast_to_python_float(
                            bpij
                        )
        return header

    @staticmethod
    def _readHeader(header):
        return GSFitsWCS(header=header)

    def copy(self):
        # The copy module version of copying the dict works fine here.
        return copy.copy(self)

    def __eq__(self, other):
        return self is other or (
            isinstance(other, GSFitsWCS)
            and self.wcs_type == other.wcs_type
            and jnp.array_equal(self.crpix, other.crpix)
            and jnp.array_equal(self.cd, other.cd)
            and self.center == other.center
            and (
                jnp.array_equal(self.pv, other.pv)
                or (self.pv is None and other.pv is None)
            )
            and (
                jnp.array_equal(self.ab, other.ab)
                or (self.ab is None and other.ab is None)
            )
            and (
                jnp.array_equal(self.abp, other.abp)
                or (self.abp is None and other.abp is None)
            )
        )

    def __repr__(self):
        pv_repr = repr(ensure_hashable(self.pv))
        ab_repr = repr(ensure_hashable(self.ab))
        abp_repr = repr(ensure_hashable(self.abp))
        return "galsim.GSFitsWCS(_data = [%r, %r, %r, %r, %s, %s, %s])" % (
            self.wcs_type,
            ensure_hashable(self.crpix),
            ensure_hashable(self.cd),
            self.center,
            pv_repr,
            ab_repr,
            abp_repr,
        )

    def __str__(self):
        if self._tag is None:
            return self.__repr__()
        else:
            return "galsim.GSFitsWCS(%s)" % (self._tag)

    def __hash__(self):
        return hash(repr(self))


@implements(_galsim.fitswcs.TanWCS)
def TanWCS(affine, world_origin, units=arcsec):
    # These will raise the appropriate errors if affine is not the right type.
    dudx = affine.dudx * units / degrees
    dudy = affine.dudy * units / degrees
    dvdx = affine.dvdx * units / degrees
    dvdy = affine.dvdy * units / degrees
    origin = affine.origin
    # The - signs are because the Fits standard is in terms of +u going east, rather than west
    # as we have defined.  So just switch the sign in the CD matrix.
    cd = jnp.array([[-dudx, -dudy], [dvdx, dvdy]], dtype=float)
    crpix = jnp.array([origin.x, origin.y], dtype=float)

    # We also need to absorb the affine world_origin back into crpix, since GSFits is expecting
    # crpix to be the location of the tangent point in image coordinates. i.e. where (u,v) = (0,0)
    # (u,v) = CD * (x-x0,y-y0) + (u0,v0)
    # (0,0) = CD * (x0',y0') - CD * (x0,y0) + (u0,v0)
    # CD (x0',y0') = CD (x0,y0) - (u0,v0)
    # (x0',y0') = (x0,y0) - CD^-1 (u0,v0)
    uv = jnp.array(
        [
            affine.world_origin.x * units / degrees,
            affine.world_origin.y * units / degrees,
        ]
    )
    crpix -= jnp.dot(jnp.linalg.inv(cd), uv)

    # Invoke the private constructor of GSFits using the _data kwarg.
    data = ("TAN", crpix, cd, world_origin, None, None, None)
    return GSFitsWCS(_data=data)


# This is a list of all the WCS types that can potentially read a WCS from a FITS file.
# The function FitsWCS will try each of these in order and return the first one that
# succeeds.  AffineTransform should be last, since it will always succeed.
# The list is defined here at global scope so that external modules can add extra
# WCS types to the list if desired.

fits_wcs_types = [
    GSFitsWCS,  # This doesn't work for very many WCS types, but it works for the very common
    # TAN projection, and also TPV, which is used by SCamp.  If it does work, it
    # is a good choice, since it is easily the fastest of any of these.
]


@implements(
    _galsim.fitswcs.FitsWCS,
    lax_description="JAX-GalSim only supports the GSFitsWCS class for celestial WCS types.",
)
def FitsWCS(
    file_name=None,
    dir=None,
    hdu=None,
    header=None,
    compression="auto",
    text_file=False,
    suppress_warning=False,
):
    if file_name is not None:
        if header is not None:
            raise GalSimIncompatibleValuesError(
                "Cannot provide both file_name and pyfits header",
                file_name=file_name,
                header=header,
            )
        header = fits.FitsHeader(
            file_name=file_name,
            dir=dir,
            hdu=hdu,
            compression=compression,
            text_file=text_file,
        )
    else:
        file_name = "header"  # For sensible error messages below.
    if header is None:
        raise GalSimIncompatibleValuesError(
            "Must provide either file_name or header",
            file_name=file_name,
            header=header,
        )
    if not isinstance(header, fits.FitsHeader):
        header = fits.FitsHeader(header)

    if "CTYPE1" not in header and "CDELT1" not in header:
        if not suppress_warning:
            galsim_warn(
                "No WCS information found in %r. Defaulting to PixelScale(1.0)"
                % (file_name)
            )
        return PixelScale(1.0)

    # For linear WCS specifications, AffineTransformation should work.
    # Note: Most files will have CTYPE1,2, but old style with only CDELT1,2 sometimes omits it.
    if header.get("CTYPE1", "LINEAR") == "LINEAR":
        wcs = AffineTransform._readHeader(header)
        # Convert to PixelScale if possible.
        # TODO: Should we do this check in JAX-GalSim or maybe always return AffineTransform?
        if wcs.dudx == wcs.dvdy and wcs.dudy == wcs.dvdx == 0:
            if wcs.x0 == wcs.y0 == wcs.u0 == wcs.v0 == 0:
                wcs = PixelScale(wcs.dudx)
            else:
                wcs = OffsetWCS(wcs.dudx, wcs.origin, wcs.world_origin)
        return wcs

    # Otherwise (and typically), try the various wcs types that can read celestial coordinates.
    for wcs_type in fits_wcs_types:
        try:
            wcs = wcs_type._readHeader(header)
            # Give it a better tag for the repr if appropriate.
            if hasattr(wcs, "_tag") and file_name != "header":
                if dir is not None:
                    wcs._tag = repr(os.path.join(dir, file_name))
                else:
                    wcs._tag = repr(file_name)
                if hdu is not None:
                    wcs._tag += ", hdu=%r" % hdu
                if compression != "auto":
                    wcs._tag += ", compression=%r" % compression
            return wcs
        except Exception:
            pass
    else:
        # Finally, this one is really the last resort, since it only reads in the linear part of the
        # WCS.  It defaults to the equivalent of a pixel scale of 1.0 if even these are not present.
        if not suppress_warning:
            galsim_warn(
                "All the fits WCS types failed to read %r. Using AffineTransform "
                "instead, which will not really be correct." % (file_name)
            )
        return AffineTransform._readHeader(header)


# Let this function work like a class in config.
FitsWCS._req_params = {"file_name": str}
FitsWCS._opt_params = {"dir": str, "hdu": int, "compression": str, "text_file": bool}


@jax.jit
def _invert_ab_noraise(u, v, ab, abp=None):
    # get guess from abp if we have it
    if abp is None:
        x = u.copy()
        y = v.copy()
    else:
        x = horner2d(u, v, abp[0])
        y = horner2d(u, v, abp[1])

    # Code below from galsim C++ layer and written by Josh Meyers
    # Matt Becker translated into jax

    # Assemble horner2d coefs for derivatives
    nab = ab.shape[1]
    dudxcoef = (jnp.arange(nab)[:, None] * ab[0])[1:, :-1]
    dudycoef = (jnp.arange(nab) * ab[0])[:-1, 1:]
    dvdxcoef = (jnp.arange(nab)[:, None] * ab[1])[1:, :-1]
    dvdycoef = (jnp.arange(nab) * ab[1])[:-1, 1:]

    def _step(i, args):
        x, y, _, _, u, v, ab, dudxcoef, dudycoef, dvdxcoef, dvdycoef = args

        # Want Jac^-1 . du
        # du
        du = horner2d(x, y, ab[0], triangle=True) - u
        dv = horner2d(x, y, ab[1], triangle=True) - v
        # J
        dudx = horner2d(x, y, dudxcoef, triangle=True)
        dudy = horner2d(x, y, dudycoef, triangle=True)
        dvdx = horner2d(x, y, dvdxcoef, triangle=True)
        dvdy = horner2d(x, y, dvdycoef, triangle=True)
        # J^-1 . du
        det = dudx * dvdy - dudy * dvdx
        duu = -(du * dvdy - dv * dudy) / det
        dvv = -(-du * dvdx + dv * dudx) / det

        x += duu
        y += dvv

        return x, y, duu, dvv, u, v, ab, dudxcoef, dudycoef, dvdxcoef, dvdycoef

    x, y, dx, dy = jax.lax.fori_loop(
        0,
        10,
        _step,
        (
            x,
            y,
            jnp.zeros_like(x),
            jnp.zeros_like(y),
            u,
            v,
            ab,
            dudxcoef,
            dudycoef,
            dvdxcoef,
            dvdycoef,
        ),
    )[0:4]

    x, y = jax.lax.cond(
        jnp.maximum(jnp.max(jnp.abs(dx)), jnp.max(jnp.abs(dy))) > 2e-12,
        lambda x, y: (x * jnp.nan, y * jnp.nan),
        lambda x, y: (x, y),
        x,
        y,
    )

    return x, y
