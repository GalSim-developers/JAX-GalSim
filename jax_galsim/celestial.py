# original source license:
#
# Copyright (c) 2013-2017 LSST Dark Energy Science Collaboration (DESC)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import warnings
from functools import partial

import coord as _coord
import galsim as _galsim
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from jax_galsim.angle import Angle, _Angle, arcsec, degrees, radians
from jax_galsim.core.utils import ensure_hashable, implements


# we have to copy this one since JAX sends in `t` as a traced array
# and the coord.Angle classes don't know how to handle that
@implements(_coord.util.ecliptic_obliquity)
def _ecliptic_obliquity(epoch):
    # We need to figure out the time in Julian centuries from J2000 for this epoch.
    t = (epoch - 2000.0) / 100.0
    # Then we use the last (most recent) formula listed under
    # http://en.wikipedia.org/wiki/Ecliptic#Obliquity_of_the_ecliptic, from
    # JPL's 2010 calculations.
    ep = Angle.from_dms("23:26:21.406")
    ep -= Angle.from_dms("00:00:46.836769") * t
    ep -= Angle.from_dms("00:00:0.0001831") * (t**2)
    ep += Angle.from_dms("00:00:0.0020034") * (t**3)
    # There are even higher order terms, but they are probably not important for any reasonable
    # calculation someone would do with this package.
    return ep


def _sun_position_ecliptic(date):
    return _Angle(_coord.util.sun_position_ecliptic(date).rad)


@implements(
    _galsim.celestial.CelestialCoord,
    lax_description=(
        "The JAX version of this object does not check that the declination is between -90 and 90."
    ),
)
@register_pytree_node_class
class CelestialCoord(object):
    def __init__(self, ra, dec=None):
        if isinstance(ra, CelestialCoord) and dec is None:
            # Copy constructor
            self._ra = ra._ra
            self._dec = ra._dec
        elif ra is None or dec is None:
            raise TypeError("ra and dec are both required")
        elif not isinstance(ra, Angle):
            raise TypeError("ra must be a galsim.Angle")
        elif not isinstance(dec, Angle):
            raise TypeError("dec must be a galsim.Angle")
        else:
            # Normal case
            self._ra = ra
            self._dec = dec

    @property
    @implements(_galsim.celestial.CelestialCoord.ra)
    def ra(self):
        return self._ra

    @property
    @implements(_galsim.celestial.CelestialCoord.dec)
    def dec(self):
        return self._dec

    @property
    @implements(_galsim.celestial.CelestialCoord.rad)
    def rad(self):
        return (self._ra.rad, self._dec.rad)

    @jax.jit
    def _get_aux(self):
        _sindec, _cosdec = self._dec.sincos()
        _sinra, _cosra = self._ra.sincos()
        _x = _cosdec * _cosra
        _y = _cosdec * _sinra
        _z = _sindec
        return _cosra, _sinra, _cosdec, _sindec, _x, _y, _z

    # DO NOT ACUTALLY USE THIS, HERE FOR TESTING PURPOSES ONLY
    def _set_aux(self):
        aux = self._get_aux()
        (
            self._cosra,
            self._sinra,
            self._cosdec,
            self._sindec,
            self._x,
            self._y,
            self._z,
        ) = aux

    @implements(_galsim.celestial.CelestialCoord.get_xyz)
    def get_xyz(self):
        return self._get_aux()[4:]

    @staticmethod
    @jax.jit
    @implements(
        _galsim.celestial.CelestialCoord.from_xyz,
        lax_description=(
            "The JAX version of this static method does not check that the norm of the input "
            "vector is non-zero."
        ),
    )
    def from_xyz(x, y, z):
        norm = jnp.sqrt(x * x + y * y + z * z)
        ret = CelestialCoord.__new__(CelestialCoord)
        ret._x = x / norm
        ret._y = y / norm
        ret._z = z / norm
        ret._sindec = ret._z
        ret._cosdec = jnp.sqrt(ret._x * ret._x + ret._y * ret._y)
        ret._sinra = jnp.where(
            ret._cosdec == 0,
            0,
            ret._y / ret._cosdec,
        )
        ret._cosra = jnp.where(
            ret._cosdec == 0,
            1.0,
            ret._x / ret._cosdec,
        )
        ret._ra = (jnp.arctan2(ret._sinra, ret._cosra) * radians).wrap(_Angle(jnp.pi))
        ret._dec = jnp.arctan2(ret._sindec, ret._cosdec) * radians
        return ret

    @staticmethod
    @jax.jit
    @implements(_galsim.celestial.CelestialCoord.radec_to_xyz)
    def radec_to_xyz(ra, dec, r=1.0):
        cosdec = jnp.cos(dec)
        x = cosdec * jnp.cos(ra) * r
        y = cosdec * jnp.sin(ra) * r
        z = jnp.sin(dec) * r
        return x, y, z

    @staticmethod
    @partial(jax.jit, static_argnames=("return_r",))
    @implements(_galsim.celestial.CelestialCoord.xyz_to_radec)
    def xyz_to_radec(x, y, z, return_r=False):
        xy2 = x**2 + y**2
        ra = jnp.arctan2(y, x)
        # Note: We don't need arctan2, since always quadrant 1 or 4.
        #       Using plain arctan is slightly faster.  About 10% for the whole function.
        #       However, if any points have x=y=0, then this will raise a numpy warning.
        #       It still gives the right answer, but we catch and ignore the warning here.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            dec = jnp.arctan(z / jnp.sqrt(xy2))
        if return_r:
            return ra, dec, jnp.sqrt(xy2 + z**2)
        else:
            return ra, dec

    @implements(_galsim.celestial.CelestialCoord.normal)
    def normal(self):
        return _CelestialCoord(self.ra.wrap(_Angle(jnp.pi)), self.dec)

    @staticmethod
    @jax.jit
    def _raw_dsq(auxc1, auxc2):
        # Compute the raw dsq between two coordinates.
        c1_x, c1_y, c1_z = auxc1[4:]
        c2_x, c2_y, c2_z = auxc2[4:]
        return (c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2 + (c1_z - c2_z) ** 2

    @staticmethod
    @jax.jit
    def _raw_cross(auxc1, auxc2):
        # Compute the raw cross product between two coordinates.
        c1_x, c1_y, c1_z = auxc1[4:]
        c2_x, c2_y, c2_z = auxc2[4:]
        return (
            c1_y * c2_z - c2_y * c1_z,
            c1_z * c2_x - c2_z * c1_x,
            c1_x * c2_y - c2_x * c1_y,
        )

    @implements(_galsim.celestial.CelestialCoord.distanceTo)
    @jax.jit
    def distanceTo(self, coord2):
        # The easiest way to do this in a way that is stable for small separations
        # is to calculate the (x,y,z) position on the unit sphere corresponding to each
        # coordinate position.
        #
        # x = cos(dec) cos(ra)
        # y = cos(dec) sin(ra)
        # z = sin(dec)

        aux = self._get_aux()
        auxc = coord2._get_aux()

        # The direct distance between the two points is
        #
        # d^2 = (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2
        dsq = self._raw_dsq(aux, auxc)

        theta = jnp.where(
            dsq < 3.99,
            # (The usual case.  This formula is perfectly stable here.)
            # This direct distance can then be converted to a great circle distance via
            #
            # sin(theta/2) = d/2
            2.0 * jnp.arcsin(0.5 * jnp.sqrt(dsq)),
            # Points are nearly antipodes where the accuracy of this formula starts to break down.
            # But in this case, the cross product provides an accurate distance.
            jnp.pi
            - jnp.arcsin(jnp.sqrt(jnp.sum(jnp.array(self._raw_cross(aux, auxc)) ** 2))),
        )

        return _Angle(theta)

    @implements(
        _galsim.celestial.CelestialCoord.greatCirclePoint,
        lax_description=(
            "The JAX version of this method does not check that coord2 defines a unique great "
            "circle with the current coord at angle theta."
        ),
    )
    @jax.jit
    def greatCirclePoint(self, coord2, theta):
        aux = self._get_aux()
        auxc = coord2._get_aux()

        # Define u = self
        #        v = coord2
        #        w = (u x v) x u
        # The great circle through u and v is then
        #
        #   R(t) = u cos(t) + w sin(t)
        #
        # Rather than directly calculate (u x v) x u, let's do some simplification first.
        # u x v = ( uy vz - uz vy )
        #         ( uz vx - ux vz )
        #         ( ux vy - uy vx )
        # wx = (u x v)_y uz - (u x v)_z uy
        #    = (uz vx - ux vz) uz - (ux vy - uy vx) uy
        #    = vx uz^2 - vz ux uz - vy ux uy + vx uy^2
        #    = vx (1 - ux^2) - ux (uz vz + uy vy)
        #    = vx - ux (u . v)
        #    = vx - ux (1 - d^2/2)
        #    = vx - ux + ux d^2/2
        # wy = vy - uy + uy d^2/2
        # wz = vz - uz + uz d^2/2

        dsq = self._raw_dsq(aux, auxc)

        # These are unnormalized yet.
        _x, _y, _z = aux[4:]
        c_x, c_y, c_z = auxc[4:]
        wx = c_x - _x + _x * dsq / 2.0
        wy = c_y - _y + _y * dsq / 2.0
        wz = c_z - _z + _z * dsq / 2.0

        # Normalize
        wr = (wx**2 + wy**2 + wz**2) ** 0.5
        # if wr == 0.:
        #     raise ValueError("coord2 does not define a unique great circle with self.")
        wx /= wr
        wy /= wr
        wz /= wr

        # R(theta)
        s, c = theta.sincos()
        rx = _x * c + wx * s
        ry = _y * c + wy * s
        rz = _z * c + wz * s
        return CelestialCoord.from_xyz(rx, ry, rz)

    @jax.jit
    def _triple(self, aux, auxc2, auxc3):
        """Compute the scalar triple product of the three vectors:

                (A x C). B = sina sinb sinC

        where C = self, A = coord2, B = coord3.  This is used by both angleBetween and area.
        (Although note that the triple product is invariant to the ordering modulo a sign.)
        """
        _x, _y, _z = aux[4:]
        c2_x, c2_y, c2_z = auxc2[4:]
        c3_x, c3_y, c3_z = auxc3[4:]

        # Note, the scalar triple product, (AxC).B, is the determinant of the 3x3 matrix
        #     [ xA yA zA ]
        #     [ xC yC zC ]
        #     [ xB yB zB ]
        # Furthermore, it is more stable to calculate it that way than computing the cross
        # product by hand and then dotting it to the other vector.

        # JAX has separate code path for 3x3 determinants that doesn't match the LU path in
        # galsim/ numpy. The slogdet function uses the LU decomp by default, so we use that.
        sign, logdet = jnp.linalg.slogdet(
            jnp.array(
                [[c2_x, c2_y, c2_z], [_x, _y, _z], [c3_x, c3_y, c3_z]],
                dtype=float,
            )
        )
        return sign * jnp.exp(logdet)

    @jax.jit
    def _alt_triple(self, aux, auxc2, auxc3):
        """Compute a different triple product of the three vectors:

                (A x C). (B x C) = sina sinb cosC

        where C = self, A = coord2, B = coord3.  This is used by both angleBetween and area.
        """
        # We can simplify (AxC).(BxC) as follows:
        #     (A x C) . (B x C)
        #     = (C x (BxC)) . A         Rotation of triple product with (BxC) one of the vectors
        #     = ((C.C)B - (C.B)C) . A   Vector triple product identity
        #     = A.B - (A.C) (B.C)       C.C = 1
        # Dot products for nearby coordinates are not very accurate.  Better to use the distances
        # between the points: A.B = 1 - d_AB^2/2
        #     = 1 - d_AB^2/2 - (1-d_AC^2/2) (1-d_BC^2/2)
        #     = d_AC^2 / 2 + d_BC^2 / 2 - d_AB^2 / 2 - d_AC^2 d_BC^2 / 4
        dsq_AC = self._raw_dsq(aux, auxc2)
        dsq_BC = self._raw_dsq(aux, auxc3)
        dsq_AB = self._raw_dsq(auxc2, auxc3)
        return 0.5 * (dsq_AC + dsq_BC - dsq_AB - 0.5 * dsq_AC * dsq_BC)

    @implements(_galsim.celestial.CelestialCoord.angleBetween)
    @jax.jit
    def angleBetween(self, coord2, coord3):
        # Call A = coord2, B = coord3, C = self
        # Then we are looking for the angle ACB.
        # If we treat each coord as a (x,y,z) vector, then we can use the following spherical
        # trig identities:
        #
        # (A x C) . B = sina sinb sinC
        # (A x C) . (B x C) = sina sinb cosC
        #
        # Then we can just use atan2 to find C, and atan2 automatically gets the sign right.
        # And we only need 1 trig call, assuming that x,y,z are already set up, which is often
        # the case.

        aux = self._get_aux()
        auxc2 = coord2._get_aux()
        auxc3 = coord3._get_aux()

        sinC = self._triple(aux, auxc2, auxc3)
        cosC = self._alt_triple(aux, auxc2, auxc3)

        C = jnp.arctan2(sinC, cosC)
        return _Angle(C)

    @implements(_galsim.celestial.CelestialCoord.area)
    @jax.jit
    def area(self, coord2, coord3):
        # The area of a spherical triangle is defined by the "spherical excess", E.
        # There are several formulae for E:
        #    (cf. http://en.wikipedia.org/wiki/Spherical_trigonometry#Area_and_spherical_excess)
        #
        # E = A + B + C - pi
        # tan(E/4) = sqrt(tan(s/2) tan((s-a)/2) tan((s-b)/2) tan((s-c)/2)
        # tan(E/2) = tan(a/2) tan(b/2) sin(C) / (1 + tan(a/2) tan(b/2) cos(C))
        #
        # We use the last formula, which is stable both for small triangles and ones that are
        # nearly degenerate (which the middle formula may have trouble with).
        #
        # Furthermore, we can use some of the math for angleBetween and distanceTo to simplify
        # this further:
        #
        # In angleBetween, we have formulae for sina sinb sinC and sina sinb cosC.
        # In distanceTo, we have formulae for sin(a/2) and sin(b/2).
        #
        # Define: F = sina sinb sinC
        #         G = sina sinb cosC
        #         da = 2 sin(a/2)
        #         db = 2 sin(b/2)
        #
        # tan(E/2) = sin(a/2) sin(b/2) sin(C) / (cos(a/2) cos(b/2) + sin(a/2) sin(b/2) cos(C))
        #          = sin(a) sin(b) sin(C) / (4 cos(a/2)^2 cos(b/2)^2 + sin(a) sin(b) cos(C))
        #          = F / (4 (1-sin(a/2)^2) (1-sin(b/2)^2) + G)
        #          = F / (4-da^2) (4-db^2)/4 + G)

        aux = self._get_aux()
        auxc2 = coord2._get_aux()
        auxc3 = coord3._get_aux()

        F = self._triple(aux, auxc2, auxc3)
        G = self._alt_triple(aux, auxc2, auxc3)
        dasq = self._raw_dsq(aux, auxc2)
        dbsq = self._raw_dsq(aux, auxc3)

        tanEo2 = F / (0.25 * (4.0 - dasq) * (4.0 - dbsq) + G)
        E = 2.0 * jnp.arctan(jnp.abs(tanEo2))
        return E

    _valid_projections = [None, "gnomonic", "stereographic", "lambert", "postel"]

    @implements(_galsim.celestial.CelestialCoord.project)
    def project(self, coord2, projection=None):
        if projection not in CelestialCoord._valid_projections:
            raise ValueError("Unknown projection: %s" % projection)

        # The core calculation is done in a helper function:
        u, v = self._project(coord2._get_aux(), projection)

        return u * radians, v * radians

    @implements(_galsim.celestial.CelestialCoord.project_rad)
    def project_rad(self, ra, dec, projection=None):
        if projection not in CelestialCoord._valid_projections:
            raise ValueError("Unknown projection: %s" % projection)

        cosra = jnp.cos(ra)
        sinra = jnp.sin(ra)
        cosdec = jnp.cos(dec)
        sindec = jnp.sin(dec)

        return self._project((cosra, sinra, cosdec, sindec), projection)

    @partial(jax.jit, static_argnums=(2,))
    def _project(self, auxc, projection):
        cosra, sinra, cosdec, sindec = auxc[:4]
        _cosra, _sinra, _cosdec, _sindec = self._get_aux()[:4]
        # The equations are given at the above mathworld websites.  They are the same except
        # for the definition of k:
        #
        # x = k cos(dec) sin(ra-ra0)
        # y = k ( cos(dec0) sin(dec) - sin(dec0) cos(dec) cos(ra-ra0) )
        #
        # Lambert:
        #   k = sqrt( 2  / ( 1 + cos(c) ) )
        # Stereographic:
        #   k = 2 / ( 1 + cos(c) )
        # Gnomonic:
        #   k = 1 / cos(c)
        # Postel:
        #   k = c / sin(c)
        # where cos(c) = sin(dec0) sin(dec) + cos(dec0) cos(dec) cos(ra-ra0)

        # cos(dra) = cos(ra-ra0) = cos(ra0) cos(ra) + sin(ra0) sin(ra)
        cosdra = _cosra * cosra
        cosdra += _sinra * sinra

        # sin(dra) = -sin(ra - ra0)
        # Note: - sign here is to make +x correspond to -ra,
        #       so x increases for decreasing ra.
        #       East is to the left on the sky!
        # sin(dra) = -cos(ra0) sin(ra) + sin(ra0) cos(ra)
        sindra = _sinra * cosra
        sindra -= _cosra * sinra

        # Calculate k according to which projection we are using
        cosc = cosdec * cosdra
        cosc *= _cosdec
        cosc += _sindec * sindec
        if projection is None or projection[0] == "g":
            k = 1.0 / cosc
        elif projection[0] == "s":
            k = 2.0 / (1.0 + cosc)
        elif projection[0] == "l":
            k = jnp.sqrt(2.0 / (1.0 + cosc))
        else:
            c = jnp.arccos(cosc)
            # k = c / np.sin(c)
            # np.sinc is defined as sin(pi x) / (pi x)
            # So need to divide by pi first.
            k = 1.0 / jnp.sinc(c / jnp.pi)

        # u = k * cosdec * sindra
        # v = k * ( self._cosdec * sindec - self._sindec * cosdec * cosdra )
        u = cosdec * sindra
        v = cosdec * cosdra
        v *= -_sindec
        v += _cosdec * sindec
        u *= k
        v *= k

        return u, v

    @implements(_galsim.celestial.CelestialCoord.deproject)
    def deproject(self, u, v, projection=None):
        if projection not in CelestialCoord._valid_projections:
            raise ValueError("Unknown projection: %s" % projection)

        # Again, do the core calculations in a helper function
        ra, dec = self._deproject(u / radians, v / radians, projection)

        return CelestialCoord(_Angle(ra), _Angle(dec))

    @implements(_galsim.celestial.CelestialCoord.deproject_rad)
    def deproject_rad(self, u, v, projection=None):
        if projection not in CelestialCoord._valid_projections:
            raise ValueError("Unknown projection: %s" % projection)

        return self._deproject(u, v, projection)

    @partial(jax.jit, static_argnums=(3,))
    def _deproject(self, u, v, projection):
        # The inverse equations are also given at the same web sites:
        #
        # sin(dec) = cos(c) sin(dec0) + v sin(c) cos(dec0) / r
        # tan(ra-ra0) = u sin(c) / (r cos(dec0) cos(c) - v sin(dec0) sin(c))
        #
        # where
        #
        # r = sqrt(u^2+v^2)
        # c = tan^(-1)(r)     for gnomonic
        # c = 2 tan^(-1)(r/2) for stereographic
        # c = 2 sin^(-1)(r/2) for lambert
        # c = r               for postel

        # Note that we can rewrite the formulae as:
        #
        # sin(dec) = cos(c) sin(dec0) + v (sin(c)/r) cos(dec0)
        # tan(ra-ra0) = u (sin(c)/r) / (cos(dec0) cos(c) - v sin(dec0) (sin(c)/r))
        #
        # which means we only need cos(c) and sin(c)/r.  For most of the projections,
        # this saves us from having to take sqrt(rsq).

        rsq = u * u
        rsq += v * v
        if projection is None or projection[0] == "g":
            # c = arctan(r)
            # cos(c) = 1 / sqrt(1+r^2)
            # sin(c) = r / sqrt(1+r^2)
            cosc = sinc_over_r = 1.0 / jnp.sqrt(1.0 + rsq)
        elif projection[0] == "s":
            # c = 2 * arctan(r/2)
            # Some trig manipulations reveal:
            # cos(c) = (4-r^2) / (4+r^2)
            # sin(c) = 4r / (4+r^2)
            cosc = (4.0 - rsq) / (4.0 + rsq)
            sinc_over_r = 4.0 / (4.0 + rsq)
        elif projection[0] == "l":
            # c = 2 * arcsin(r/2)
            # Some trig manipulations reveal:
            # cos(c) = 1 - r^2/2
            # sin(c) = r sqrt(4-r^2) / 2
            cosc = 1.0 - rsq / 2.0
            sinc_over_r = jnp.sqrt(4.0 - rsq) / 2.0
        else:
            r = jnp.sqrt(rsq)
            cosc = jnp.cos(r)
            sinc_over_r = jnp.sinc(r / jnp.pi)

        # Compute sindec, tandra
        # Note: more efficient to use numpy op= as much as possible to avoid temporary arrays.
        _cosra, _sinra, _cosdec, _sindec = self._get_aux()[:4]
        # sindec = cosc * self._sindec + v * sinc_over_r * self._cosdec
        sindec = v * sinc_over_r
        sindec *= _cosdec
        sindec += cosc * _sindec
        # Remember the - sign so +dra is -u.  East is left.
        tandra_num = u * sinc_over_r
        tandra_num *= -1.0
        # tandra_denom = cosc * self._cosdec - v * sinc_over_r * self._sindec
        tandra_denom = v * sinc_over_r
        tandra_denom *= -_sindec
        tandra_denom += cosc * _cosdec

        dec = jnp.arcsin(sindec)
        ra = self.ra.rad + jnp.arctan2(tandra_num, tandra_denom)

        return ra, dec

    @implements(_galsim.celestial.CelestialCoord.jac_deproject)
    def jac_deproject(self, u, v, projection=None):
        if projection not in CelestialCoord._valid_projections:
            raise ValueError("Unknown projection: %s" % projection)

        return self._jac_deproject(u.rad, v.rad, projection)

    @implements(_galsim.celestial.CelestialCoord.jac_deproject_rad)
    def jac_deproject_rad(self, u, v, projection=None):
        if projection not in CelestialCoord._valid_projections:
            raise ValueError("Unknown projection: %s" % projection)

        return self._jac_deproject(u, v, projection)

    @partial(jax.jit, static_argnums=(3,))
    def _jac_deproject(self, u, v, projection):
        # sin(dec) = cos(c) sin(dec0) + v sin(c)/r cos(dec0)
        # tan(ra-ra0) = u sin(c)/r / (cos(dec0) cos(c) - v sin(dec0) sin(c)/r)
        #
        # d(sin(dec)) = cos(dec) ddec = s0 dc + (v ds + s dv) c0
        # dtan(ra-ra0) = sec^2(ra-ra0) dra
        #              = ( (u ds + s du) A - u s (dc c0 - (v ds + s dv) s0 ) )/A^2
        # where s = sin(c) / r
        #       c = cos(c)
        #       s0 = sin(dec0)
        #       c0 = cos(dec0)
        #       A = c c0 - v s s0

        rsq = u * u + v * v
        # rsq1 = (u + 1.e-4)**2 + v**2
        # rsq2 = u**2 + (v + 1.e-4)**2
        if projection is None or projection[0] == "g":
            c = s = 1.0 / jnp.sqrt(1.0 + rsq)
            s3 = s * s * s
            dcdu = dsdu = -u * s3
            dcdv = dsdv = -v * s3
        elif projection[0] == "s":
            s = 4.0 / (4.0 + rsq)
            c = 2.0 * s - 1.0
            ssq = s * s
            dcdu = -u * ssq
            dcdv = -v * ssq
            dsdu = 0.5 * dcdu
            dsdv = 0.5 * dcdv
        elif projection[0] == "l":
            c = 1.0 - rsq / 2.0
            s = jnp.sqrt(4.0 - rsq) / 2.0
            dcdu = -u
            dcdv = -v
            dsdu = -u / (4.0 * s)
            dsdv = -v / (4.0 * s)
        else:
            r = jnp.sqrt(rsq)

            # original code for reference
            # if r == 0.:
            #     c = s = 1
            #     dcdu = -u
            #     dcdv = -v
            #     dsdu = dsdv = 0
            # else:
            #     c = np.cos(r)
            #     s = np.sin(r)/r
            #     dcdu = -s*u
            #     dcdv = -s*v
            #     dsdu = (c-s)*u/rsq
            #     dsdv = (c-s)*v/rsq

            c = jnp.where(
                r == 0.0,
                1.0,
                jnp.cos(r),
            )
            s = jnp.where(
                r == 0.0,
                1.0,
                jnp.sin(r) / r,
            )
            dcdu = jnp.where(
                r == 0.0,
                -u,
                -s * u,
            )
            dcdv = jnp.where(
                r == 0.0,
                -v,
                -s * v,
            )
            dsdu = jnp.where(
                r == 0.0,
                0.0,
                (c - s) * u / rsq,
            )
            dsdv = jnp.where(
                r == 0.0,
                0.0,
                (c - s) * v / rsq,
            )

        _cosra, _sinra, _cosdec, _sindec = self._get_aux()[:4]
        s0 = _sindec
        c0 = _cosdec
        sindec = c * s0 + v * s * c0
        cosdec = jnp.sqrt(1.0 - sindec * sindec)
        dddu = (s0 * dcdu + v * dsdu * c0) / cosdec
        dddv = (s0 * dcdv + (v * dsdv + s) * c0) / cosdec

        tandra_num = u * s
        tandra_denom = c * c0 - v * s * s0
        # Note: A^2 sec^2(dra) = denom^2 (1 + tan^2(dra) = denom^2 + num^2
        A2sec2dra = tandra_denom**2 + tandra_num**2
        drdu = (
            (u * dsdu + s) * tandra_denom - u * s * (dcdu * c0 - v * dsdu * s0)
        ) / A2sec2dra
        drdv = (
            u * dsdv * tandra_denom - u * s * (dcdv * c0 - (v * dsdv + s) * s0)
        ) / A2sec2dra

        drdu *= cosdec
        drdv *= cosdec
        return jnp.array([[drdu, drdv], [dddu, dddv]])

    @implements(_galsim.celestial.CelestialCoord.precess)
    def precess(self, from_epoch, to_epoch):
        return CelestialCoord._precess(
            from_epoch, to_epoch, self._ra.rad, self._dec.rad
        )

    @implements(_galsim.celestial.CelestialCoord.galactic)
    def galactic(self, epoch=2000.0):
        # cf. Lang, Astrophysical Formulae, page 13
        # cos(b) cos(el-33) = cos(dec) cos(ra-282.25)
        # cos(b) sin(el-33) = sin(dec) sin(62.6) + cos(dec) sin(ra-282.25) cos(62.6)
        #            sin(b) = sin(dec) cos(62.6) - cos(dec) sin(ra-282.25) sin(62.6)
        #
        # Those formulae were for the 1950 epoch.  The corresponding numbers for J2000 are:
        # (cf. https://arxiv.org/pdf/1010.3773.pdf)
        el0 = 32.93191857 * degrees
        r0 = 282.859481208 * degrees
        d0 = 62.8717488056 * degrees
        sind0, cosd0 = d0.sincos()

        sind, cosd = self.dec.sincos()
        sinr, cosr = (self.ra - r0).sincos()

        cbcl = cosd * cosr
        cbsl = sind * sind0 + cosd * sinr * cosd0
        sb = sind * cosd0 - cosd * sinr * sind0

        b = _Angle(jnp.arcsin(sb))
        el = (_Angle(jnp.arctan2(cbsl, cbcl)) + el0).wrap(_Angle(jnp.pi))

        return (el, b)

    @staticmethod
    @implements(_galsim.celestial.CelestialCoord.from_galactic)
    def from_galactic(el, b, epoch=2000.0):
        el0 = 32.93191857 * degrees
        r0 = 282.859481208 * degrees
        d0 = 62.8717488056 * degrees
        sind0, cosd0 = d0.sincos()

        sinb, cosb = b.sincos()
        sinl, cosl = (el - el0).sincos()
        x1 = cosb * cosl
        y1 = cosb * sinl
        z1 = sinb

        x2 = x1
        y2 = y1 * cosd0 - z1 * sind0
        z2 = y1 * sind0 + z1 * cosd0

        temp = CelestialCoord.from_xyz(x2, y2, z2)
        return CelestialCoord(temp.ra + r0, temp.dec).normal()

    @partial(jax.jit, static_argnames=("date",))
    @implements(_galsim.celestial.CelestialCoord.ecliptic)
    def ecliptic(self, epoch=2000.0, date=None):
        # We are going to work in terms of the (x, y, z) projections.
        _x, _y, _z = self._get_aux()[4:]

        # Get the obliquity of the ecliptic.
        if date is not None:
            epoch = date.year
        ep = _ecliptic_obliquity(epoch)
        sin_ep, cos_ep = ep.sincos()

        # Coordinate transformation here, from celestial to ecliptic:
        x_ecl = _x
        y_ecl = cos_ep * _y + sin_ep * _z
        z_ecl = -sin_ep * _y + cos_ep * _z

        beta = _Angle(jnp.arcsin(z_ecl))
        lam = _Angle(jnp.arctan2(y_ecl, x_ecl))

        if date is not None:
            # Find the sun position in ecliptic coordinates on this date.  We have to convert to
            # Julian day in order to use our helper routine to find the Sun position in ecliptic
            # coordinates.
            lam_sun = _sun_position_ecliptic(date)
            # Subtract it off, to get ecliptic coordinates relative to the sun.
            lam -= lam_sun

        return (lam.wrap(), beta)

    @staticmethod
    @partial(jax.jit, static_argnames=("date",))
    @implements(_galsim.celestial.CelestialCoord.from_ecliptic)
    def from_ecliptic(lam, beta, epoch=2000.0, date=None):
        if date is not None:
            lam += _sun_position_ecliptic(date)

        # Get the (x, y, z)_ecliptic from (lam, beta).
        sinbeta, cosbeta = beta.sincos()
        sinlam, coslam = lam.sincos()
        x_ecl = cosbeta * coslam
        y_ecl = cosbeta * sinlam
        z_ecl = sinbeta

        # Get the obliquity of the ecliptic.
        if date is not None:
            epoch = date.year
        ep = _ecliptic_obliquity(epoch)

        # Transform to (x, y, z)_equatorial.
        sin_ep, cos_ep = ep.sincos()
        x_eq = x_ecl
        y_eq = cos_ep * y_ecl - sin_ep * z_ecl
        z_eq = sin_ep * y_ecl + cos_ep * z_ecl

        return CelestialCoord.from_xyz(x_eq, y_eq, z_eq)

    def __repr__(self):
        return "galsim.CelestialCoord(%r, %r)" % (
            ensure_hashable(self._ra),
            ensure_hashable(self._dec),
        )

    def __str__(self):
        return "galsim.CelestialCoord(%s, %s)" % (
            ensure_hashable(self._ra),
            ensure_hashable(self._dec),
        )

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return (
            isinstance(other, CelestialCoord)
            and jnp.array_equal(self._ra.rad, other._ra.rad)
            and jnp.array_equal(self._dec.rad, other._dec.rad)
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def tree_flatten(self):
        """This function flattens the CelestialCoord into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        children = (self._ra, self._dec)
        # Define auxiliary static data that doesn’t need to be traced
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        return _CelestialCoord(*children)

    @jax.jit
    def _precess_kern(from_epoch, to_epoch, _x, _y, _z, _ra, _dec):
        # t0, t below correspond to Lieske's big T and little T
        t0 = (from_epoch - 2000.0) / 100.0
        t = (to_epoch - from_epoch) / 100.0
        t02 = t0 * t0
        t2 = t * t
        t3 = t2 * t

        # a,b,c below correspond to Lieske's zeta_A, z_A and theta_A
        a = (
            (2306.2181 + 1.39656 * t0 - 0.000139 * t02) * t
            + (0.30188 - 0.000344 * t0) * t2
            + 0.017998 * t3
        ) * arcsec
        b = (
            (2306.2181 + 1.39656 * t0 - 0.000139 * t02) * t
            + (1.09468 + 0.000066 * t0) * t2
            + 0.018203 * t3
        ) * arcsec
        c = (
            (2004.3109 - 0.85330 * t0 - 0.000217 * t02) * t
            + (-0.42665 - 0.000217 * t0) * t2
            - 0.041833 * t3
        ) * arcsec
        sina, cosa = a.sincos()
        sinb, cosb = b.sincos()
        sinc, cosc = c.sincos()

        # This is the precession rotation matrix:
        xx = cosa * cosc * cosb - sina * sinb
        yx = -sina * cosc * cosb - cosa * sinb
        zx = -sinc * cosb
        xy = cosa * cosc * sinb + sina * cosb
        yy = -sina * cosc * sinb + cosa * cosb
        zy = -sinc * sinb
        xz = cosa * sinc
        yz = -sina * sinc
        zz = cosc

        # Perform the rotation:
        x2 = xx * _x + yx * _y + zx * _z
        y2 = xy * _x + yy * _y + zy * _z
        z2 = xz * _x + yz * _y + zz * _z

        return CelestialCoord.from_xyz(x2, y2, z2).normal()

    @jax.jit
    def _precess(from_epoch, to_epoch, _ra, _dec):
        _sindec, _cosdec = jnp.sin(_dec), jnp.cos(_dec)
        _sinra, _cosra = jnp.sin(_ra), jnp.cos(_ra)
        _x = _cosdec * _cosra
        _y = _cosdec * _sinra
        _z = _sindec

        return jax.lax.cond(
            jnp.array_equal(from_epoch, to_epoch),
            lambda *args: _CelestialCoord(_Angle(args[-2]), _Angle(args[-1])),
            CelestialCoord._precess_kern,
            from_epoch,
            to_epoch,
            _x,
            _y,
            _z,
            _ra,
            _dec,
        )

    @staticmethod
    def from_galsim(gcoord):
        """Create a jax_galsim `CelestialCoord` from a `galsim.CelestialCoord` object."""
        return _CelestialCoord(_Angle(gcoord.ra.rad), _Angle(gcoord.dec.rad))

    def to_galsim(self):
        """Create a galsim `CelestialCoord` from a `jax_galsim.CelestialCoord` object."""
        return _galsim.celestial.CelestialCoord(
            self.ra.to_galsim(), self.dec.to_galsim()
        )


@implements(_coord._CelestialCoord)
def _CelestialCoord(ra, dec):
    ret = CelestialCoord.__new__(CelestialCoord)
    ret._ra = ra
    ret._dec = dec
    return ret
