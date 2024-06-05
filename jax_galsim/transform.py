import galsim as _galsim
import jax
import jax.numpy as jnp
from jax._src.numpy.util import implements
from jax.tree_util import register_pytree_node_class

from jax_galsim.core.utils import compute_major_minor_from_jacobian, ensure_hashable
from jax_galsim.gsobject import GSObject
from jax_galsim.gsparams import GSParams
from jax_galsim.position import PositionD


@implements(
    _galsim.Transform,
    lax_description="Does not support Chromatic Objects or Convolutions.",
)
def Transform(
    obj,
    jac=(1.0, 0.0, 0.0, 1.0),
    offset=PositionD(0.0, 0.0),
    flux_ratio=1.0,
    gsparams=None,
    propagate_gsparams=True,
):
    if not (isinstance(obj, GSObject)):
        raise TypeError("Argument to Transform must be a GSObject.")
    elif (
        hasattr(jac, "__call__")
        or hasattr(offset, "__call__")
        or hasattr(flux_ratio, "__call__")
    ):
        raise NotImplementedError("Transform does not support callable arguments.")
    else:
        return Transformation(
            obj, jac, offset, flux_ratio, gsparams, propagate_gsparams
        )


@implements(_galsim.Transformation)
@register_pytree_node_class
class Transformation(GSObject):
    def __init__(
        self,
        obj,
        jac=(1.0, 0.0, 0.0, 1.0),
        offset=PositionD(0.0, 0.0),
        flux_ratio=1.0,
        gsparams=None,
        propagate_gsparams=True,
    ):
        self._gsparams = GSParams.check(gsparams, obj.gsparams)
        self._propagate_gsparams = propagate_gsparams
        if self._propagate_gsparams:
            obj = obj.withGSParams(self._gsparams)

        self._params = {
            "jac": jac,
            "offset": PositionD(offset),
            "flux_ratio": flux_ratio,
        }

        # this import is here to avoid circular imports
        # we do not want to mess with the transform properties of the interpolated image
        from .interpolatedimage import InterpolatedImage

        if isinstance(obj, Transformation) and not isinstance(obj, InterpolatedImage):
            # Combine the two affine transformations into one.
            dx, dy = self._fwd(obj._params["offset"].x, obj._params["offset"].y)
            self._offset.x += dx
            self._offset.y += dy
            self._params["jac"] = self._jac.dot(obj._jac)
            self._params["flux_ratio"] *= obj._flux_ratio
            self._original = obj._original
        else:
            self._original = obj

    ##############################################################
    # The internal code of the methods of the Transform class
    # should only aceess _offset, _flux_ratio, and _jac. It
    # should not pull these directly from _params.
    # Things are structured this way since the interpolated image
    # class inherits and overrides these methods.

    @property
    def _offset(self):
        return self._params["offset"]

    # we use this property so that the interpolated image can override
    # how flux ratio is computer / stored
    @property
    def _flux_ratio(self):
        return self._params["flux_ratio"]

    @property
    def _jac(self):
        jac = self._params["jac"]
        jac = jax.lax.cond(
            jac is not None,
            lambda jac: jnp.broadcast_to(jnp.array(jac, dtype=float).ravel(), (4,)),
            lambda jax: jnp.array([1.0, 0.0, 0.0, 1.0]),
            jac,
        )
        return jnp.asarray(jac, dtype=float).reshape((2, 2))

    @property
    def original(self):
        """The original object being transformed."""
        return self._original

    @property
    def jac(self):
        """The Jacobian of the transforamtion."""
        return self._jac

    @property
    def offset(self):
        """The offset of the transformation."""
        return self._offset

    @property
    def flux_ratio(self):
        """The flux ratio of the transformation."""
        return self._flux_ratio

    @property
    def _flux(self):
        return self._flux_scaling * self._original.flux

    @implements(_galsim.Transformation.withGSParams)
    def withGSParams(self, gsparams=None, **kwargs):
        """Create a version of the current object with the given gsparams

        .. note::

            Unless you set ``propagate_gsparams=False``, this method will also update the gsparams
            of the wrapped component object.
        """
        if gsparams == self._gsparams:
            return self

        chld, aux = self.tree_flatten()
        aux["gsparams"] = GSParams.check(gsparams, self._gsparams, **kwargs)
        if self._propagate_gsparams:
            new_obj = chld[0].withGSParams(aux["gsparams"])
            chld = (new_obj,) + chld[1:]

        return self.tree_unflatten(aux, chld)

    def __eq__(self, other):
        return self is other or (
            isinstance(other, Transformation)
            and self._original == other._original
            and jnp.array_equal(self._jac, other._jac)
            and self._offset == other._params["offset"]
            and self._flux_ratio == other._flux_ratio
            and self._gsparams == other._gsparams
            and self._propagate_gsparams == other._propagate_gsparams
        )

    def __hash__(self):
        return hash(
            (
                "galsim.Transformation",
                self._original,
                ensure_hashable(self._jac.ravel()),
                ensure_hashable(self._offset.x),
                ensure_hashable(self._offset.y),
                ensure_hashable(self._flux_ratio),
                self._gsparams,
                self._propagate_gsparams,
            )
        )

    def __repr__(self):
        return (
            "galsim.Transformation(%r, jac=%r, offset=%r, flux_ratio=%r, gsparams=%r, "
            "propagate_gsparams=%r)"
        ) % (
            self._original,
            ensure_hashable(self._jac.ravel()),
            self._offset,
            ensure_hashable(self._flux_ratio),
            self._gsparams,
            self._propagate_gsparams,
        )

    @classmethod
    def _str_from_jac(cls, jac):
        from jax_galsim.wcs import JacobianWCS

        dudx, dudy, dvdx, dvdy = jac.ravel()
        if dudx != 1 or dudy != 0 or dvdx != 0 or dvdy != 1:
            # Figure out the shear/rotate/dilate calls that are equivalent.
            jac = JacobianWCS(dudx, dudy, dvdx, dvdy)
            scale, shear, theta, flip = jac.getDecomposition()
            s = None
            if flip:
                s = 0  # Special value indicating to just use transform.
            if abs(theta.rad) > 1.0e-12:
                if s is None:
                    s = ".rotate(%s)" % theta
                else:
                    s = 0
            if shear.g > 1.0e-12:
                if s is None:
                    s = ".shear(%s)" % shear
                else:
                    s = 0
            if abs(scale - 1.0) > 1.0e-12:
                if s is None:
                    s = ".expand(%s)" % scale
                else:
                    s = 0
            if s == 0:
                # If flip or there are two components, then revert to transform as simpler.
                s = ".transform(%s,%s,%s,%s)" % (dudx, dudy, dvdx, dvdy)
            if s is None:
                # If nothing is large enough to show up above, give full detail of transform
                s = ".transform(%r,%r,%r,%r)" % (dudx, dudy, dvdx, dvdy)
            return s
        else:
            return ""

    def __str__(self):
        s = str(self._original)
        s += self._str_from_jac(self._jac)
        if self._offset.x != 0 or self._offset.y != 0:
            s += ".shift(%s,%s)" % (
                ensure_hashable(self._offset.x),
                ensure_hashable(self._offset.y),
            )
        if self._flux_ratio != 1.0:
            s += " * %s" % ensure_hashable(self._flux_ratio)
        return s

    @property
    def _det(self):
        return jnp.linalg.det(self._jac)

    @property
    def _invdet(self):
        return 1.0 / self._det

    @property
    def _invjac(self):
        return jnp.linalg.inv(self._jac)

    # To avoid confusion with the flux vs amplitude scaling, we use these names below, rather
    # than flux_ratio, which is really an amplitude scaling.
    @property
    def _amp_scaling(self):
        return self._flux_ratio

    @property
    def _flux_scaling(self):
        return jnp.abs(self._det) * self._flux_ratio

    def _fwd(self, x, y):
        res = jnp.dot(self._jac, jnp.array([x, y]))
        return res[0], res[1]

    def _fwdT(self, x, y):
        res = jnp.dot(self._jac.T, jnp.array([x, y]))
        return res[0], res[1]

    def _inv(self, x, y):
        res = jnp.dot(self._invjac, jnp.array([x, y]))
        return res[0], res[1]

    def _kfactor(self, kx, ky):
        kx *= -1j * self._offset.x
        ky *= -1j * self._offset.y
        kx += ky
        return self._flux_scaling * jnp.exp(kx)

    @property
    def _maxk(self):
        _, minor = compute_major_minor_from_jacobian(self._jac)
        return self._original.maxk / minor

    @property
    def _stepk(self):
        major, _ = compute_major_minor_from_jacobian(self._jac)
        stepk = self._original.stepk / major
        # If we have a shift, we need to further modify stepk
        #     stepk = Pi/R
        #     R <- R + |shift|
        #     stepk <- Pi/(Pi/stepk + |shift|)
        dr = jnp.hypot(self._offset.x, self._offset.y)
        stepk = jnp.pi / (jnp.pi / stepk + dr)
        return stepk

    @property
    def _has_hard_edges(self):
        return self._original.has_hard_edges

    @property
    def _is_axisymmetric(self):
        return bool(
            self._original.is_axisymmetric
            and self._jac[0, 0] == self._jac[1, 1]
            and self._jac[0, 1] == -self._jac[1, 0]
            and self._offset == PositionD(0.0, 0.0)
        )

    @property
    def _is_analytic_x(self):
        return self._original.is_analytic_x

    @property
    def _is_analytic_k(self):
        return self._original.is_analytic_k

    @property
    def _centroid(self):
        cen = self._original.centroid
        cen = PositionD(self._fwd(cen.x, cen.y))
        cen += self._offset
        return cen

    @property
    def _positive_flux(self):
        return self._flux_scaling * self._original.positive_flux

    @property
    def _negative_flux(self):
        return self._flux_scaling * self._original.negative_flux

    @property
    def _flux_per_photon(self):
        return self._calculate_flux_per_photon()

    @property
    def _max_sb(self):
        return self._amp_scaling * self._original.max_sb

    def _xValue(self, pos):
        pos -= self._offset
        inv_pos = PositionD(self._inv(pos.x, pos.y))
        return self._original._xValue(inv_pos) * self._amp_scaling

    def _kValue(self, kpos):
        fwdT_kpos = PositionD(self._fwdT(kpos.x, kpos.y))
        return self._original._kValue(fwdT_kpos) * self._kfactor(kpos.x, kpos.y)

    def _drawReal(self, image, jac=None, offset=(0.0, 0.0), flux_scaling=1.0):
        dx, dy = offset
        if jac is not None:
            x1 = jac.dot(self._offset._array)
            dx += x1[0]
            dy += x1[1]
        else:
            dx += self._offset.x
            dy += self._offset.y
        flux_scaling *= self._flux_scaling
        jac = (
            self._jac
            if jac is None
            else jac
            if self._jac is None
            else jac.dot(self._jac)
        )
        return self._original._drawReal(image, jac, (dx, dy), flux_scaling)

    def _drawKImage(self, image, jac=None):
        from jax_galsim.core.draw import apply_kImage_phases

        jac1 = (
            self._jac
            if jac is None
            else jac
            if self._jac is None
            else jac.dot(self._jac)
        )
        image = self._original._drawKImage(image, jac1)

        _jac = jnp.eye(2) if jac is None else jac
        image = apply_kImage_phases(self._offset, image, _jac)

        image = image * self._flux_scaling
        return image

    def _shoot(self, photons, rng):
        self._original._shoot(photons, rng)
        photons.x, photons.y = self._fwd(photons.x, photons.y)
        photons.x += self._offset.x
        photons.y += self._offset.y
        photons.scaleFlux(self._flux_scaling)

    def tree_flatten(self):
        """This function flattens the Transform into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        children = (self._original, self._params)
        # Define auxiliary static data that doesn’t need to be traced
        aux_data = {
            "gsparams": self._gsparams,
            "propagate_gsparams": self._propagate_gsparams,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        obj = cls.__new__(cls)
        obj._gsparams = aux_data["gsparams"]
        obj._propagate_gsparams = aux_data["propagate_gsparams"]
        obj._original, obj._params = children
        return obj


def _Transform(
    obj,
    jac=(1.0, 0.0, 0.0, 1.0),
    offset=PositionD(0.0, 0.0),
    flux_ratio=1.0,
    gsparams=None,
):
    """Approximately equivalent to Transform, but without some of the sanity checks (such as
    checking for chromatic options).

    For a `ChromaticObject`, you must use the regular `Transform`.
    """
    return Transformation(obj, jac, offset, flux_ratio, gsparams)
