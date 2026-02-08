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
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from jax_galsim.core.interpolate import akima_interp, akima_interp_coeffs
from jax_galsim.core.utils import ensure_hashable, implements, is_equal_with_arrays
from jax_galsim.errors import (
    GalSimIncompatibleValuesError,
    GalSimNotImplementedError,
    GalSimRangeError,
    GalSimValueError,
)


def _str_array(a):
    with np.printoptions(threshold=5, edgeitems=2, linewidth=1000):
        return repr(np.asarray(a))


@implements(
    _galsim.LookupTable,
    lax_description="""\
The JAX-GalSim version of LookupTable uses Akima (C1) cubic splines
rather than GalSim's natural cubic splines (C2) for interpolant='spline'.
GalSim Interpolant objects are not supported as the interpolant parameter;
only the string values 'floor', 'ceil', 'nearest', 'linear', 'spline' are supported.
""",
)
@register_pytree_node_class
class LookupTable:
    def __init__(self, x, f, interpolant="spline", x_log=False, f_log=False):
        self.x_log = x_log
        self.f_log = f_log

        # Only support string interpolants
        if interpolant not in ("nearest", "linear", "ceil", "floor", "spline"):
            raise GalSimNotImplementedError(
                "Only string interpolants ('nearest', 'linear', 'ceil', 'floor', 'spline') "
                "are supported in jax_galsim.LookupTable. Got %r." % interpolant
            )
        self.interpolant = interpolant

        # Sanity checks
        if len(x) != len(f):
            raise GalSimIncompatibleValuesError(
                "Input array lengths don't match", x=x, f=f
            )
        if len(x) < 2:
            raise GalSimValueError("Input arrays too small to interpolate", x)

        # Convert to JAX arrays
        x = jnp.asarray(x, dtype=float)
        f = jnp.asarray(f, dtype=float)

        # Sort if needed
        if not jnp.all(x[1:] >= x[:-1]):
            s = jnp.argsort(x)
            x = x[s]
            f = f[s]

        self.x = x
        self.f = f

        self._x_min = float(x[0])
        self._x_max = float(x[-1])
        if self._x_min == self._x_max:
            raise GalSimValueError("All x values are equal", x)
        if self.x_log and float(x[0]) <= 0.0:
            raise GalSimValueError(
                "Cannot interpolate in log(x) when table contains x<=0.", x
            )
        if self.f_log and bool(jnp.any(f <= 0.0)):
            raise GalSimValueError(
                "Cannot interpolate in log(f) when table contains f<=0.", f
            )

        # Pre-compute spline coefficients
        if interpolant == "spline":
            _x = jnp.log(x) if x_log else x
            _f = jnp.log(f) if f_log else f
            self._coeffs = akima_interp_coeffs(_x, _f)
        else:
            self._coeffs = None

    @property
    def x_min(self):
        return self._x_min

    @property
    def x_max(self):
        return self._x_max

    def __len__(self):
        return len(self.x)

    def __call__(self, x):
        x = jnp.asarray(x, dtype=float)

        # Transform to internal coordinates
        if self.x_log:
            _x = jnp.log(x)
            _xp = jnp.log(self.x)
        else:
            _x = x
            _xp = self.x

        if self.f_log:
            _fp = jnp.log(self.f)
        else:
            _fp = self.f

        if self.interpolant == "linear":
            result = jnp.interp(_x, _xp, _fp)
        elif self.interpolant == "spline":
            result = akima_interp(_x, _xp, _fp, self._coeffs)
        elif self.interpolant == "nearest":
            idx = jnp.searchsorted(_xp, _x, side="right") - 1
            idx = jnp.clip(idx, 0, len(_xp) - 1)
            # Choose nearest
            idx_next = jnp.clip(idx + 1, 0, len(_xp) - 1)
            dist_left = jnp.abs(_x - _xp[idx])
            dist_right = jnp.abs(_x - _xp[idx_next])
            idx = jnp.where(dist_right < dist_left, idx_next, idx)
            result = _fp[idx]
        elif self.interpolant == "floor":
            idx = jnp.searchsorted(_xp, _x, side="right") - 1
            idx = jnp.clip(idx, 0, len(_xp) - 1)
            result = _fp[idx]
        elif self.interpolant == "ceil":
            idx = jnp.searchsorted(_xp, _x, side="left")
            idx = jnp.clip(idx, 0, len(_xp) - 1)
            result = _fp[idx]
        else:
            raise GalSimValueError("Unknown interpolant", self.interpolant)

        if self.f_log:
            result = jnp.exp(result)

        return result

    def integrate(self, x_min=None, x_max=None):
        if self.x_log:
            raise GalSimNotImplementedError("log x spacing not implemented yet.")
        if self.f_log:
            raise GalSimNotImplementedError("log f values not implemented yet.")
        if x_min is None:
            x_min = self.x_min
        else:
            x_min = max(x_min, self.x_min)
        if x_max is None:
            x_max = self.x_max
        else:
            x_max = min(x_max, self.x_max)

        if x_min > x_max:
            return -self.integrate(x_max, x_min)
        if x_min == x_max:
            return 0.0

        if self.interpolant == "spline":
            # Analytic integration of cubic polynomial pieces
            return self._integrate_spline(x_min, x_max)
        else:
            # Trapezoidal integration using stored points
            return self._integrate_trapz(x_min, x_max)

    def _integrate_trapz(self, x_min, x_max):
        # Get all x points in range, plus endpoints
        x = np.asarray(self.x)
        f = np.asarray(self.f)

        mask = (x >= x_min) & (x <= x_max)
        xr = x[mask]
        fr = f[mask]

        # Add endpoint values if not already present
        pts_x = [float(x_min)]
        pts_f = [float(np.interp(x_min, x, f))]
        for xi, fi in zip(xr, fr):
            xi = float(xi)
            fi = float(fi)
            if xi > x_min and xi < x_max:
                pts_x.append(xi)
                pts_f.append(fi)
        pts_x.append(float(x_max))
        pts_f.append(float(np.interp(x_max, x, f)))

        pts_x = np.array(pts_x)
        pts_f = np.array(pts_f)

        return float(np.trapz(pts_f, pts_x))

    def _integrate_spline(self, x_min, x_max):
        # Analytic integration of the piecewise cubic
        # P(x) = a + b*(x-xi) + c*(x-xi)^2 + d*(x-xi)^3
        # integral from u to v of P = a*(v-u) + b/2*(v-u)^2 + c/3*(v-u)^3 + d/4*(v-u)^4
        # where u,v are relative to xi
        x = np.asarray(self.x)
        a, b, c, d = [np.asarray(ci) for ci in self._coeffs]
        n = len(x)

        total = 0.0
        for i in range(n - 1):
            seg_lo = max(x[i], x_min)
            seg_hi = min(x[i + 1], x_max)
            if seg_lo >= seg_hi:
                continue
            u = seg_lo - x[i]
            v = seg_hi - x[i]

            def _antideriv(t):
                return a[i] * t + b[i] / 2 * t**2 + c[i] / 3 * t**3 + d[i] / 4 * t**4

            total += _antideriv(v) - _antideriv(u)

        return float(total)

    def integrate_product(self, g, x_min=None, x_max=None, x_factor=1.0):
        if self.x_log:
            raise GalSimNotImplementedError("log x spacing not implemented yet.")
        if self.f_log:
            raise GalSimNotImplementedError("log f values not implemented yet.")

        if x_min is None:
            x_min = self.x_min / x_factor
        else:
            x_min = max(x_min, self.x_min / x_factor)
        if x_max is None:
            x_max = self.x_max / x_factor
        else:
            x_max = min(x_max, self.x_max / x_factor)
        if x_min > x_max:
            return -self.integrate_product(g, x_max, x_min, x_factor)
        if x_min == x_max:
            return 0.0

        if isinstance(g, LookupTable):
            x_min = max(x_min, g.x_min)
            x_max = min(x_max, g.x_max)
            if x_min >= x_max:
                return 0.0
        else:
            # Convert g into a LookupTable over the relevant range
            gx = np.asarray(self.x) / x_factor
            gx = gx[(gx >= x_min) & (gx <= x_max)]
            gx = np.unique(np.sort(np.concatenate([gx, [x_min, x_max]])))
            gf = g(gx)
            try:
                len(gf)
            except TypeError:
                gf1 = gf
                gf = np.empty_like(gx, dtype=float)
                gf.fill(gf1)
            g = _LookupTable(gx, gf, "linear")

        # Merge grids from both functions
        self_x = np.asarray(self.x) / x_factor
        g_x = np.asarray(g.x)
        all_x = np.unique(np.sort(np.concatenate([self_x, g_x])))
        all_x = all_x[(all_x >= x_min) & (all_x <= x_max)]
        # Add endpoints
        all_x = np.unique(np.sort(np.concatenate([all_x, [x_min, x_max]])))

        # Following GalSim: g is always approximated as piecewise-linear
        # between the merged abscissae, regardless of its actual interpolant.
        g_vals = np.asarray(g(all_x))

        # For self with linear interpolant: both f and g are piecewise-linear
        # on the merged grid, so f*g is piecewise-quadratic -> integrate exactly
        # via Simpson's rule.
        # For self with spline/other interpolant: refine the grid to capture
        # the curvature of f, then use the exact quadratic integration.
        if self.interpolant == "linear":
            f_vals = np.asarray(self(all_x * x_factor))
            return float(self._integrate_product_linear(all_x, f_vals, g_vals))
        else:
            # Refine grid between knots to capture curvature of f
            # Use more points for step functions (floor/ceil/nearest)
            nrefine = 5000
            refined = [all_x]
            for i in range(len(all_x) - 1):
                mid_pts = np.linspace(all_x[i], all_x[i + 1], nrefine, endpoint=False)[
                    1:
                ]
                refined.append(mid_pts)
            fine_x = np.unique(np.sort(np.concatenate(refined)))
            f_fine = np.asarray(self(fine_x * x_factor))
            # g is piecewise-linear on the coarse grid, so interpolate linearly
            g_fine = np.interp(fine_x, all_x, g_vals)
            return float(np.trapz(f_fine * g_fine, fine_x))

    @staticmethod
    def _integrate_product_linear(x, f, g):
        """Analytically integrate the product of two piecewise-linear functions.

        Between consecutive knots x[i] and x[i+1], f and g are linear, so
        f*g is quadratic. The exact integral of a quadratic on [a,b] with
        endpoint values f0*g0 and f1*g1 and midpoint value
        (f0+f1)/2 * (g0+g1)/2 is (b-a)/6 * (f0*g0 + 4*fm*gm + f1*g1)
        (Simpson's rule, which is exact for quadratics).
        """
        total = 0.0
        for i in range(len(x) - 1):
            dx = x[i + 1] - x[i]
            f0, f1 = f[i], f[i + 1]
            g0, g1 = g[i], g[i + 1]
            fm = 0.5 * (f0 + f1)
            gm = 0.5 * (g0 + g1)
            # Simpson's rule: exact for quadratic
            total += dx / 6.0 * (f0 * g0 + 4.0 * fm * gm + f1 * g1)
        return total

    def _check_range(self, x):
        slop = (self.x_max - self.x_min) * 1.0e-6
        x = np.asarray(x)
        if np.min(x) < self.x_min - slop:
            raise GalSimRangeError(
                "x value(s) below the range of the LookupTable.",
                x[x < self.x_min],
                self.x_min,
                self.x_max,
            )
        if np.max(x) > self.x_max + slop:
            raise GalSimRangeError(
                "x value(s) above the range of the LookupTable.",
                x[x > self.x_max],
                self.x_min,
                self.x_max,
            )

    def getArgs(self):
        return self.x

    def getVals(self):
        return self.f

    def getInterp(self):
        return self.interpolant

    def isLogX(self):
        return self.x_log

    def isLogF(self):
        return self.f_log

    def __eq__(self, other):
        return self is other or (
            isinstance(other, LookupTable)
            and is_equal_with_arrays(self.x, other.x)
            and is_equal_with_arrays(self.f, other.f)
            and self.x_log == other.x_log
            and self.f_log == other.f_log
            and self.interpolant == other.interpolant
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        if not hasattr(self, "_hash"):
            self._hash = hash(
                (
                    "galsim.LookupTable",
                    ensure_hashable(self.x),
                    ensure_hashable(self.f),
                    self.x_log,
                    self.f_log,
                    self.interpolant,
                )
            )
        return self._hash

    def __repr__(self):
        return (
            "galsim.LookupTable(x=array(%r), f=array(%r), interpolant=%r, x_log=%r, f_log=%r)"
            % (
                np.asarray(self.x).tolist(),
                np.asarray(self.f).tolist(),
                self.interpolant,
                self.x_log,
                self.f_log,
            )
        )

    def __str__(self):
        s = "galsim.LookupTable(x=%s, f=%s" % (
            _str_array(self.x),
            _str_array(self.f),
        )
        if self.interpolant != "spline":
            s += ", interpolant=%r" % (self.interpolant)
        if self.x_log:
            s += ", x_log=True"
        if self.f_log:
            s += ", f_log=True"
        s += ")"
        return s

    @classmethod
    def from_file(
        cls, file_name, interpolant="spline", x_log=False, f_log=False, amplitude=1.0
    ):
        data = np.loadtxt(file_name).transpose()
        if data.shape[0] != 2:
            raise GalSimValueError(
                "File provided for LookupTable does not have 2 columns", file_name
            )
        x = data[0]
        f = data[1]
        if amplitude != 1.0:
            f = f * amplitude
        return cls(x, f, interpolant=interpolant, x_log=x_log, f_log=f_log)

    @classmethod
    def from_func(
        cls,
        func,
        x_min,
        x_max,
        npoints=2000,
        interpolant="spline",
        x_log=False,
        f_log=False,
    ):
        if x_log:
            x = np.exp(np.linspace(np.log(x_min), np.log(x_max), npoints))
        else:
            x = np.linspace(x_min, x_max, npoints)
        f = np.array([func(xx) for xx in x], dtype=float)
        return cls(x, f, interpolant=interpolant, x_log=x_log, f_log=f_log)

    def tree_flatten(self):
        children = (self.x, self.f, self._coeffs)
        aux_data = {
            "interpolant": self.interpolant,
            "x_log": self.x_log,
            "f_log": self.f_log,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.x = children[0]
        obj.f = children[1]
        obj._coeffs = children[2]
        obj.interpolant = aux_data["interpolant"]
        obj.x_log = aux_data["x_log"]
        obj.f_log = aux_data["f_log"]
        obj._x_min = float(obj.x[0])
        obj._x_max = float(obj.x[-1])
        return obj


def _LookupTable(x, f, interpolant="spline", x_log=False, f_log=False):
    """Make a LookupTable but without sanity checks. Input x must be already sorted."""
    obj = LookupTable.__new__(LookupTable)
    obj.x = jnp.asarray(x, dtype=float)
    obj.f = jnp.asarray(f, dtype=float)
    obj.interpolant = interpolant
    obj.x_log = x_log
    obj.f_log = f_log
    obj._x_min = float(obj.x[0])
    obj._x_max = float(obj.x[-1])
    if interpolant == "spline":
        _xp = jnp.log(obj.x) if x_log else obj.x
        _fp = jnp.log(obj.f) if f_log else obj.f
        obj._coeffs = akima_interp_coeffs(_xp, _fp)
    else:
        obj._coeffs = None
    return obj


@implements(
    _galsim.table.trapz,
    lax_description="Uses the JAX-GalSim LookupTable for integration.",
)
def trapz(f, x, interpolant="linear"):
    if len(x) >= 2:
        return _LookupTable(x, f, interpolant=interpolant).integrate()
    else:
        return 0.0
