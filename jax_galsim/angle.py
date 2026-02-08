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
import galsim as _galsim
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from jax_galsim.core.utils import cast_to_float, ensure_hashable, implements


@implements(_galsim.AngleUnit)
@register_pytree_node_class
class AngleUnit(object):
    valid_names = ["rad", "deg", "hr", "hour", "arcmin", "arcsec"]

    def __init__(self, value):
        if isinstance(value, AngleUnit):
            raise TypeError("Cannot construct AngleUnit from another AngleUnit")
        self._value = cast_to_float(value)

    @property
    @implements(_galsim.AngleUnit.value)
    def value(self):
        return self._value

    def __rmul__(self, theta):
        """float * AngleUnit returns an Angle"""
        return Angle(theta, self)

    def __div__(self, unit):
        """AngleUnit / AngleUnit returns a float giving the relative scaling.

        Note: At least to within machine precision, it is the case that

            (x * angle_unit1) / angle_unit2 == x * (angle_unit1 / angle_unit2)
        """
        if not isinstance(unit, AngleUnit):
            raise TypeError("Cannot divide AngleUnit by %s" % unit)
        return self.value / unit.value

    __truediv__ = __div__

    @staticmethod
    @implements(_galsim.AngleUnit.from_name)
    def from_name(unit):
        unit = unit.strip().lower()
        if unit.startswith("rad"):
            return radians
        elif unit.startswith("deg"):
            return degrees
        elif unit.startswith("hour"):
            return hours
        elif unit.startswith("hr"):
            return hours
        elif unit.startswith("arcmin"):
            return arcmin
        elif unit.startswith("arcsec"):
            return arcsec
        else:
            raise ValueError("Unknown Angle unit: %s" % unit)

    def __repr__(self):
        if self == radians:
            return "galsim.radians"
        elif self == degrees:
            return "galsim.degrees"
        elif self == hours:
            return "galsim.hours"
        elif self == arcmin:
            return "galsim.arcmin"
        elif self == arcsec:
            return "galsim.arcsec"
        else:
            return "galsim.AngleUnit(%r)" % ensure_hashable(self.value)

    def __eq__(self, other):
        return isinstance(other, AngleUnit) and jnp.array_equal(self.value, other.value)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(("galsim.AngleUnit", ensure_hashable(self.value)))

    def tree_flatten(self):
        """This function flattens the AngleUnit into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        children = (self._value,)
        # Define auxiliary static data that doesn’t need to be traced
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        return cls(children[0])


# Convenient pre-set built-in units
# (These are typically the only ones we will use.)

radians = AngleUnit(1.0)
hours = AngleUnit(jnp.pi / 12.0)
degrees = AngleUnit(jnp.pi / 180.0)
arcmin = AngleUnit(jnp.pi / 10800.0)
arcsec = AngleUnit(jnp.pi / 648000.0)


@implements(_galsim.Angle)
@register_pytree_node_class
class Angle(object):
    def __init__(self, theta, unit=None):
        # We also want to allow angle1 = Angle(angle2) as a copy, so check for that.
        if isinstance(theta, Angle):
            if unit is not None:
                raise TypeError(
                    "Cannot provide unit if theta is already an Angle instance"
                )
            self._rad = theta._rad
        elif unit is None:
            raise TypeError("Must provide unit for Angle.__init__")
        elif not isinstance(unit, AngleUnit):
            raise TypeError("Invalid unit %s of type %s" % (unit, type(unit)))
        else:
            # Normal case
            self._rad = cast_to_float(theta) * unit.value

    @property
    @implements(_galsim.Angle.rad)
    def rad(self):
        return self._rad

    @property
    @implements(_galsim.Angle.deg)
    def deg(self):
        return self / degrees

    def __neg__(self):
        return _Angle(-self._rad)

    def __pos__(self):
        return self

    def __abs__(self):
        return _Angle(jnp.abs(self._rad))

    def __add__(self, other):
        if not isinstance(other, Angle):
            raise TypeError(
                "Cannot add %s of type %s to an Angle" % (other, type(other))
            )
        return _Angle(self._rad + other._rad)

    def __sub__(self, other):
        if not isinstance(other, Angle):
            raise TypeError(
                "Cannot subtract %s of type %s from an Angle" % (other, type(other))
            )
        return _Angle(self._rad - other._rad)

    def __mul__(self, other):
        return _Angle(self._rad * other)

    __rmul__ = __mul__

    def __div__(self, other):
        if isinstance(other, AngleUnit):
            return self._rad / other.value
        else:
            return _Angle(self._rad / other)

    __truediv__ = __div__

    @implements(_galsim.Angle.wrap)
    def wrap(self, center=None):
        if center is None:
            center = _Angle(0.0)
        start = center._rad - jnp.pi
        offset = (self._rad - start) // (
            2.0 * jnp.pi
        )  # How many full cycles to subtract
        return _Angle(self._rad - offset * 2.0 * jnp.pi)

    @implements(_galsim.Angle.sin)
    def sin(self):
        return jnp.sin(self._rad)

    @implements(_galsim.Angle.cos)
    def cos(self):
        return jnp.cos(self._rad)

    @implements(_galsim.Angle.tan)
    def tan(self):
        return jnp.tan(self._rad)

    @implements(_galsim.Angle.sincos)
    def sincos(self):
        sin = jnp.sin(self._rad)
        cos = jnp.cos(self._rad)
        return sin, cos

    def __str__(self):
        return str(ensure_hashable(self._rad)) + " radians"

    def __repr__(self):
        return "galsim.Angle(%r, galsim.radians)" % ensure_hashable(self.rad)

    def __eq__(self, other):
        return isinstance(other, Angle) and jnp.array_equal(self.rad, other.rad)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        if not isinstance(other, Angle):
            raise TypeError(
                "Cannot compare %s of type %s to an Angle" % (other, type(other))
            )
        return self._rad <= other._rad

    def __lt__(self, other):
        if not isinstance(other, Angle):
            raise TypeError(
                "Cannot compare %s of type %s to an Angle" % (other, type(other))
            )
        return self._rad < other._rad

    def __ge__(self, other):
        if not isinstance(other, Angle):
            raise TypeError(
                "Cannot compare %s of type %s to an Angle" % (other, type(other))
            )
        return self._rad >= other._rad

    def __gt__(self, other):
        if not isinstance(other, Angle):
            raise TypeError(
                "Cannot compare %s of type %s to an Angle" % (other, type(other))
            )
        return self._rad > other._rad

    def __hash__(self):
        return hash(("galsim.Angle", ensure_hashable(self._rad)))

    @staticmethod
    def _make_dms_string(decimal, sep, prec, pad, plus_sign):
        # Account for the sign properly
        if decimal < 0:
            sign = "-"
            decimal = -decimal
        elif plus_sign:
            sign = "+"
        else:
            sign = ""

        # Figure out the 3 sep tokens
        sep1 = sep2 = ""
        sep3 = None
        if len(sep) == 1:
            sep1 = sep2 = sep
        elif len(sep) == 2:
            sep1, sep2 = sep
        elif len(sep) == 3:
            sep1, sep2, sep3 = sep

        # Round to nearest 1.e-8 seconds (or 10**-prec if given)
        round_prec = 8 if prec is None else prec
        digits = 10**round_prec

        decimal = int(3600 * digits * decimal + 0.5)

        d = decimal // (3600 * digits)
        decimal -= d * (3600 * digits)
        m = decimal // (60 * digits)
        decimal -= m * (60 * digits)
        s = decimal // digits
        decimal -= s * digits

        # Make the string
        if pad:
            d_str = "%02d" % d
            m_str = "%02d" % m
            s_str = "%02d" % s
        else:
            d_str = "%d" % d
            m_str = "%d" % m
            s_str = "%d" % s
        string = "%s%s%s%s%s%s.%0*d" % (
            sign,
            d_str,
            sep1,
            m_str,
            sep2,
            s_str,
            round_prec,
            decimal,
        )
        if not prec:
            string = string.rstrip("0")
            string = string.rstrip(".")
        if sep3:
            string = string + sep3
        return string

    @implements(_galsim.Angle.hms)
    def hms(self, sep=":", prec=None, pad=True, plus_sign=False):
        if not len(sep) <= 3:
            raise ValueError("sep must be a string or tuple of length <= 3")
        if prec is not None and not prec >= 0:
            raise ValueError("prec must be >= 0")
        return self._make_dms_string(self / hours, sep, prec, pad, plus_sign)

    @implements(_galsim.Angle.dms)
    def dms(self, sep=":", prec=None, pad=True, plus_sign=False):
        if not len(sep) <= 3:
            raise ValueError("sep must be a string or tuple of length <= 3")
        if prec is not None and not prec >= 0:
            raise ValueError("prec must be >= 0")
        return self._make_dms_string(self / degrees, sep, prec, pad, plus_sign)

    @staticmethod
    @implements(_galsim.Angle.from_hms)
    def from_hms(str):
        return Angle._parse_dms(str) * hours

    @staticmethod
    @implements(_galsim.Angle.from_dms)
    def from_dms(str):
        return Angle._parse_dms(str) * degrees

    @staticmethod
    def _parse_dms(dms):
        """Convert a string of the form dd:mm:ss.decimal into decimal degrees."""
        import re

        tokens = tuple(filter(None, re.split(r"([\.\d]+)", dms.strip())))
        if len(tokens) <= 1:
            raise ValueError("string is not of the expected format")
        sign = 1
        try:
            dd = float(tokens[0])
        except ValueError:
            if tokens[0].strip() == "-":
                sign = -1
            tokens = tokens[1:]
            dd = float(tokens[0])
            if len(tokens) <= 1:
                raise ValueError("string is not of the expected format")
        if len(tokens) <= 2:
            return sign * dd
        mm = float(tokens[2])
        if len(tokens) <= 4:
            return sign * (dd + mm / 60)
        if len(tokens) >= 7:
            raise ValueError("string is not of the expected format")
        ss = float(tokens[4])
        return sign * (dd + mm / 60.0 + ss / 3600.0)

    def tree_flatten(self):
        """This function flattens the Angle into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        children = (self._rad,)
        # Define auxiliary static data that doesn’t need to be traced
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        ret = cls.__new__(cls)
        ret._rad = children[0]
        return ret

    @staticmethod
    def from_galsim(gs_angle):
        """Create a jax_galsim `Angle` from a `galsim.Angle` object."""
        return _Angle(gs_angle._rad)

    def to_galsim(self):
        """Create a galsim `Angle` from a `jax_galsim.Angle` object."""
        return _galsim.angle._Angle(float(self._rad))


@implements(_galsim._Angle)
def _Angle(theta):
    ret = Angle.__new__(Angle)
    ret._rad = theta
    return ret
