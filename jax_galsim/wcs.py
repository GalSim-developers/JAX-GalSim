import galsim as _galsim
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from jax_galsim.angle import AngleUnit, arcsec, radians
from jax_galsim.celestial import CelestialCoord
from jax_galsim.core.utils import cast_to_python_float, ensure_hashable, implements
from jax_galsim.errors import GalSimValueError
from jax_galsim.gsobject import GSObject
from jax_galsim.position import Position, PositionD, PositionI
from jax_galsim.shear import Shear
from jax_galsim.transform import _Transform


# We inherit from the reference BaseWCS and only redefine the methods that
# make references to jax_galsim objects.
class BaseWCS(_galsim.BaseWCS):
    @implements(_galsim.BaseWCS.toWorld)
    def toWorld(self, *args, **kwargs):
        if len(args) == 1:
            if isinstance(args[0], GSObject):
                return self.profileToWorld(*args, **kwargs)
            elif isinstance(args[0], Shear):
                return self.shearToWorld(*args, **kwargs)
            else:
                return self.posToWorld(*args, **kwargs)
        elif len(args) == 2:
            if self.isCelestial():
                return self.xyToradec(*args, **kwargs)
            else:
                return self.xyTouv(*args, **kwargs)
        else:
            raise TypeError("toWorld() takes either 1 or 2 positional arguments")

    @implements(_galsim.BaseWCS.posToWorld)
    def posToWorld(self, image_pos, color=None, **kwargs):
        if color is None:
            color = self._color
        if not isinstance(image_pos, Position):
            raise TypeError("image_pos must be a PositionD or PositionI argument")
        return self._posToWorld(image_pos, color=color, **kwargs)

    @implements(_galsim.BaseWCS.profileToWorld)
    def profileToWorld(
        self,
        image_profile,
        image_pos=None,
        world_pos=None,
        color=None,
        flux_ratio=1.0,
        offset=(0, 0),
    ):
        if color is None:
            color = self._color
        return self.local(image_pos, world_pos, color=color)._profileToWorld(image_profile, flux_ratio, PositionD(offset))

    @implements(_galsim.BaseWCS.shearToWorld)
    def shearToWorld(self, image_shear, image_pos=None, world_pos=None, color=None):
        if color is None:
            color = self._color
        return self.local(image_pos, world_pos, color=color)._shearToWorld(image_shear)

    @implements(_galsim.BaseWCS.toImage)
    def toImage(self, *args, **kwargs):
        if len(args) == 1:
            if isinstance(args[0], GSObject):
                return self.profileToImage(*args, **kwargs)
            elif isinstance(args[0], Shear):
                return self.shearToImage(*args, **kwargs)
            else:
                return self.posToImage(*args, **kwargs)
        elif len(args) == 2:
            if self.isCelestial():
                return self.radecToxy(*args, **kwargs)
            else:
                return self.uvToxy(*args, **kwargs)
        else:
            raise TypeError("toImage() takes either 1 or 2 positional arguments")

    @implements(_galsim.BaseWCS.posToImage)
    def posToImage(self, world_pos, color=None):
        if color is None:
            color = self._color
        if self.isCelestial() and not isinstance(world_pos, CelestialCoord):
            raise TypeError("world_pos must be a CelestialCoord argument")
        elif not self.isCelestial() and not isinstance(world_pos, Position):
            raise TypeError("world_pos must be a PositionD or PositionI argument")
        return self._posToImage(world_pos, color=color)

    @implements(_galsim.BaseWCS.profileToImage)
    def profileToImage(
        self,
        world_profile,
        image_pos=None,
        world_pos=None,
        color=None,
        flux_ratio=1.0,
        offset=(0, 0),
    ):
        if color is None:
            color = self._color
        return self.local(image_pos, world_pos, color=color)._profileToImage(world_profile, flux_ratio, PositionD(offset))

    @implements(_galsim.BaseWCS.shearToImage)
    def shearToImage(self, world_shear, image_pos=None, world_pos=None, color=None):
        if color is None:
            color = self._color
        return self.local(image_pos, world_pos, color=color)._shearToImage(world_shear)

    @implements(_galsim.BaseWCS.local)
    def local(self, image_pos=None, world_pos=None, color=None):
        if color is None:
            color = self._color
        if world_pos is not None:
            if image_pos is not None:
                raise _galsim.GalSimIncompatibleValuesError(
                    "Only one of image_pos or world_pos may be provided",
                    image_pos=image_pos,
                    world_pos=world_pos,
                )
            image_pos = self.posToImage(world_pos, color)
        if image_pos is not None and not isinstance(image_pos, Position):
            raise TypeError("image_pos must be a PositionD or PositionI argument")
        return self._local(image_pos, color)

    @implements(_galsim.BaseWCS.jacobian)
    def jacobian(self, image_pos=None, world_pos=None, color=None):
        if color is None:
            color = self._color
        return self.local(image_pos, world_pos, color=color)._toJacobian()

    @implements(_galsim.BaseWCS.affine)
    def affine(self, image_pos=None, world_pos=None, color=None):
        if color is None:
            color = self._color
        jac = self.jacobian(image_pos, world_pos, color=color)
        # That call checked that only one of image_pos or world_pos is provided.
        if world_pos is not None:
            image_pos = self.toImage(world_pos, color=color)
        elif image_pos is None:
            # Both are None.  Must be a local WCS
            image_pos = PositionD(0, 0)

        if self._isCelestial:
            return jac.shiftOrigin(image_pos)
        else:
            if world_pos is None:
                world_pos = self.toWorld(image_pos, color=color)
            return jac.shiftOrigin(image_pos, world_pos, color=color)

    @implements(_galsim.BaseWCS.shiftOrigin)
    def shiftOrigin(self, origin, world_origin=None, color=None):
        if color is None:
            color = self._color
        if not isinstance(origin, Position):
            raise TypeError("origin must be a PositionD or PositionI argument")
        return self._shiftOrigin(origin, world_origin, color)

    @implements(_galsim.BaseWCS.withOrigin)
    def withOrigin(self, origin, world_origin=None, color=None):
        from .deprecated import depr

        depr("withOrigin", 2.3, "shiftOrigin")
        return self.shiftOrigin(origin, world_origin, color)

    # A lot of classes will need these checks, so consolidate them here
    def _set_origin(self, origin, world_origin=None):
        if origin is None:
            self._origin = PositionD(0, 0)
        else:
            if not isinstance(origin, Position):
                raise TypeError("origin must be a PositionD or PositionI argument")
            self._origin = origin
        if world_origin is None:
            self._world_origin = PositionD(0, 0)
        else:
            if not isinstance(world_origin, Position):
                raise TypeError("world_origin must be a PositionD argument")
            self._world_origin = world_origin

    def tree_flatten(self):
        """This function flattens the WCS into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        children = (self._params,)
        # Define auxiliary static data that doesn’t need to be traced
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        return cls(**(children[0]))

    @classmethod
    def from_galsim(cls, galsim_wcs):
        """Create a jax_galsim WCS object from a galsim WCS object."""
        if not isinstance(galsim_wcs, _galsim.BaseWCS):
            raise TypeError("galsim_wcs must be a galsim BaseWCS object or subclass thereof.")

        if galsim_wcs.__class__.__name__ not in globals() and galsim_wcs.__class__.__name__ != "GSFitsWCS":
            raise NotImplementedError("jax_galsim does not support the galsim WCS class %s" % galsim_wcs.__class__.__name__)

        if isinstance(galsim_wcs, _galsim.PixelScale):
            return PixelScale(galsim_wcs.scale)
        elif isinstance(galsim_wcs, _galsim.ShearWCS):
            return ShearWCS(galsim_wcs.scale, Shear.from_galsim(galsim_wcs.shear))
        elif isinstance(galsim_wcs, _galsim.JacobianWCS):
            return JacobianWCS(galsim_wcs.dudx, galsim_wcs.dudy, galsim_wcs.dvdx, galsim_wcs.dvdy)
        elif isinstance(galsim_wcs, _galsim.OffsetWCS):
            return OffsetWCS(
                galsim_wcs.scale,
                origin=Position.from_galsim(galsim_wcs.origin),
                world_origin=Position.from_galsim(galsim_wcs.world_origin),
            )
        elif isinstance(galsim_wcs, _galsim.OffsetShearWCS):
            return OffsetShearWCS(
                galsim_wcs.scale,
                Shear.from_galsim(galsim_wcs.shear),
                origin=Position.from_galsim(galsim_wcs.origin),
                world_origin=Position.from_galsim(galsim_wcs.world_origin),
            )
        elif isinstance(galsim_wcs, _galsim.AffineTransform):
            return AffineTransform(
                galsim_wcs.dudx,
                galsim_wcs.dudy,
                galsim_wcs.dvdx,
                galsim_wcs.dvdy,
                origin=Position.from_galsim(galsim_wcs.origin),
                world_origin=Position.from_galsim(galsim_wcs.world_origin),
            )
        elif isinstance(galsim_wcs, _galsim.GSFitsWCS):
            # this import goes here to avoid circular imports
            from jax_galsim.angle import radians
            from jax_galsim.celestial import CelestialCoord
            from jax_galsim.fitswcs import GSFitsWCS

            return GSFitsWCS(
                _data=[
                    galsim_wcs.wcs_type,
                    galsim_wcs.crpix,
                    galsim_wcs.cd,
                    CelestialCoord(
                        ra=galsim_wcs.center.ra.rad * radians,
                        dec=galsim_wcs.center.dec.rad * radians,
                    ),
                    galsim_wcs.pv,
                    galsim_wcs.ab,
                    galsim_wcs.abp,
                ],
            )


#########################################################################################
#
# Our class hierarchy is:
#
#    BaseWCS
#        --- EuclideanWCS
#                --- UniformWCS
#                        --- LocalWCS
#        --- CelestialWCS
#
# Here we define the rest of these classes (besides BaseWCS that is), and implement some
# functionality that is common among the subclasses of these when possible.
#
#########################################################################################


class EuclideanWCS(BaseWCS):
    """A EuclideanWCS is a `BaseWCS` whose world coordinates are on a Euclidean plane.
    We usually use the notation (u,v) to refer to positions in world coordinates, and
    they use the class `PositionD`.
    """

    # All EuclideanWCS classes must define origin and world_origin.
    # Sometimes it is convenient to access x0,y0,u0,v0 directly.
    @property
    def x0(self):
        """The x component of self.origin."""
        return self.origin.x

    @property
    def y0(self):
        """The y component of self.origin."""
        return self.origin.y

    @property
    def u0(self):
        """The x component of self.world_origin (aka u)."""
        return self.world_origin.x

    @property
    def v0(self):
        """The y component of self.world_origin (aka v)."""
        return self.world_origin.y

    def xyTouv(self, x, y, color=None):
        """Convert x,y from image coordinates to world coordinates.

        This is equivalent to ``wcs.toWorld(x,y)``.

        It is also equivalent to ``wcs.posToWorld(galsim.PositionD(x,y))`` when x and y are scalars;
        however, this routine allows x and y to be numpy arrays, in which case, the calculation
        will be vectorized, which is often much faster than using the pos interface.

        Parameters:
            x:          The x value(s) in image coordinates
            y:          The y value(s) in image coordinates
            color:      For color-dependent WCS's, the color term to use. [default: None]

        Returns:
            ra, dec
        """
        if color is None:
            color = self._color
        return self._xyTouv(x, y, color=color)

    def uvToxy(self, u, v, color=None):
        """Convert u,v from world coordinates to image coordinates.

        This is equivalent to ``wcs.toWorld(u,v)``.

        It is also equivalent to ``wcs.posToImage(galsim.PositionD(u,v))`` when u and v are scalars;
        however, this routine allows u and v to be numpy arrays, in which case, the calculation
        will be vectorized, which is often much faster than using the pos interface.

        Parameters:
            u:          The u value(s) in world coordinates
            v:          The v value(s) in world coordinates
            color:      For color-dependent WCS's, the color term to use. [default: None]
        """
        if color is None:
            color = self._color
        return self._uvToxy(u, v, color)

    # Simple.  Just call _u, _v.
    def _posToWorld(self, image_pos, color):
        x = image_pos.x - self.x0
        y = image_pos.y - self.y0
        return PositionD(self._u(x, y, color), self._v(x, y, color)) + self.world_origin

    def _xyTouv(self, x, y, color):
        x = x - self.x0  # Not -=, since don't want to modify the input arrays in place.
        y = y - self.y0
        u = self._u(x, y, color)
        v = self._v(x, y, color)
        u += self.u0
        v += self.v0
        return u, v

    # Also simple if _x,_y are implemented.  However, they are allowed to raise a
    # NotImplementedError.
    def _posToImage(self, world_pos, color):
        u = world_pos.x - self.u0
        v = world_pos.y - self.v0
        return PositionD(self._x(u, v, color), self._y(u, v, color)) + self.origin

    def _uvToxy(self, u, v, color):
        u = u - self.u0
        v = v - self.v0
        x = self._x(u, v, color)
        y = self._y(u, v, color)
        x += self.x0
        y += self.y0
        return x, y

    # Each subclass has a function _newOrigin, which just calls the constructor with new
    # values for origin and world_origin.  This function figures out what those values
    # should be to match the desired behavior of shiftOrigin.
    def _shiftOrigin(self, origin, world_origin, color):
        # Current u,v are:
        #     u = ufunc(x-x0, y-y0) + u0
        #     v = vfunc(x-x0, y-y0) + v0
        # where ufunc, vfunc represent the underlying wcs transformations.
        #
        # The _newOrigin call is expecting new values for the (x0,y0) and (u0,v0), so
        # we need to figure out how to modify the parameters given the current values.
        #
        #     Use (x1,y1) and (u1,v1) for the new values that we will pass to _newOrigin.
        #     Use (x2,y2) and (u2,v2) for the values passed as arguments.
        #
        # If world_origin is None, then we want to do basically the same thing as in the
        # non-uniform case, except that we also need to pass the function the current value of
        # wcs.world_pos to keep it from resetting the world_pos back to None.

        if world_origin is None:
            if not self._isLocal:
                origin += self.origin
                world_origin = self.world_origin
            return self._newOrigin(origin, world_origin)

        # But if world_origin is given, it isn't quite as simple.
        #
        #     u' = ufunc(x-x1, y-y1) + u1
        #     v' = vfunc(x-x1, y-y1) + v1
        #
        # We want to have:
        #     u'(x2,y2) = u2
        #     ufunc(x2-x1, y2-y1) + u1 = u2
        #
        # We don't have access to ufunc directly, just u, so
        #     (u(x2-x1+x0, y2-y1+y0) - u0) + u1 = u2
        #
        # If we take
        #     x1 = x2
        #     y1 = y2
        #
        # Then
        #     u(x0,y0) - u0 + u1 = u2
        # =>  u1 = u0 + u2 - u(x0,y0)
        #
        # And similarly,
        #     v1 = v0 + v2 - v(x0,y0)

        else:
            if not isinstance(world_origin, Position):
                raise TypeError("world_origin must be a PositionD or PositionI argument")
            if not self._isLocal:
                world_origin += self.world_origin - self._posToWorld(self.origin, color=color)
            return self._newOrigin(origin, world_origin)

    # If the class doesn't define something else, then we can approximate the local Jacobian
    # from finite differences for the derivatives.  This will be overridden by UniformWCS.
    def _local(self, image_pos, color):
        if image_pos is None:
            raise TypeError("origin must be a PositionD or PositionI argument")

        # Calculate the Jacobian using finite differences for the derivatives.
        x0 = image_pos.x - self.x0
        y0 = image_pos.y - self.y0

        # Use dx,dy = 1 pixel for numerical derivatives
        dx = 1
        dy = 1

        xlist = jnp.array([x0 + dx, x0 - dx, x0, x0], dtype=float)
        ylist = jnp.array([y0, y0, y0 + dy, y0 - dy], dtype=float)
        u = self._u(xlist, ylist, color)
        v = self._v(xlist, ylist, color)

        dudx = 0.5 * (u[0] - u[1]) / dx
        dudy = 0.5 * (u[2] - u[3]) / dy
        dvdx = 0.5 * (v[0] - v[1]) / dx
        dvdy = 0.5 * (v[2] - v[3]) / dy

        return JacobianWCS(dudx, dudy, dvdx, dvdy)

    # The naive way to make the sky image is to loop over pixels and call pixelArea(pos)
    # for that position.  This is extremely slow.  Here, we use the fact that the _u and _v
    # functions might work with numpy arrays.  If they do, this function is quite fast.
    # If not, we still get some gain from calculating u,v for each pixel and sharing some
    # of those calculations for multiple finite difference derivatives.  But the latter
    # option is still pretty slow, so it's much better to have the _u and _v work with
    # numpy arrays!
    def _makeSkyImage(self, image, sky_level, color):
        b = image.bounds
        nx = b.xmax - b.xmin + 1 + 2  # +2 more than in image to get row/col off each edge.
        ny = b.ymax - b.ymin + 1 + 2
        x, y = jnp.meshgrid(
            jnp.linspace(b.xmin - 1, b.xmax + 1, nx),
            jnp.linspace(b.ymin - 1, b.ymax + 1, ny),
        )
        x -= self.x0
        y -= self.y0
        u = self._u(x.ravel(), y.ravel(), color)
        v = self._v(x.ravel(), y.ravel(), color)
        u = jnp.reshape(u, x.shape)
        v = jnp.reshape(v, x.shape)
        # Use the finite differences to estimate the derivatives.
        dudx = 0.5 * (u[1 : ny - 1, 2:nx] - u[1 : ny - 1, 0 : nx - 2])
        dudy = 0.5 * (u[2:ny, 1 : nx - 1] - u[0 : ny - 2, 1 : nx - 1])
        dvdx = 0.5 * (v[1 : ny - 1, 2:nx] - v[1 : ny - 1, 0 : nx - 2])
        dvdy = 0.5 * (v[2:ny, 1 : nx - 1] - v[0 : ny - 2, 1 : nx - 1])

        area = jnp.abs(dudx * dvdy - dvdx * dudy)
        image._array = (area * sky_level).astype(image.dtype)

    # Each class should define the __eq__ function.  Then __ne__ is obvious.
    def __ne__(self, other):
        return not self.__eq__(other)


class UniformWCS(EuclideanWCS):
    """A UniformWCS is a `EuclideanWCS` which has a uniform pixel size and shape."""

    @property
    def _isUniform(self):
        return True

    # These can also just pass through to the _localwcs attribute.
    def _u(self, x, y, color=None):
        return self._local_wcs._u(x, y)

    def _v(self, x, y, color=None):
        return self._local_wcs._v(x, y)

    def _x(self, u, v, color=None):
        return self._local_wcs._x(u, v)

    def _y(self, u, v, color=None):
        return self._local_wcs._y(u, v)

    # For UniformWCS, the local WCS is an attribute.  Just return it.
    def _local(self, image_pos, color):
        return self._local_wcs

    # UniformWCS transformations can be inverted easily, so might as well provide that function.
    def inverse(self):
        """Return the inverse transformation, i.e. the transformation that swaps the roles of
        the "image" and "world" coordinates.
        """
        return self._inverse()

    # We'll override this for LocalWCS classes. Non-local UniformWCS classes can use that function
    # do the inversion.
    def _inverse(self):
        return self._local_wcs._inverse()._newOrigin(self.world_origin, self.origin)

    # This is very simple if the pixels are uniform.
    def _makeSkyImage(self, image, sky_level, color):
        image.fill(sky_level * self.pixelArea())

    # Just check if the locals match and if the origins match.
    def __eq__(self, other):
        return self is other or (
            isinstance(other, self.__class__)
            and self._local_wcs == other._local_wcs
            and self.origin == other.origin
            and self.world_origin == other.world_origin
        )


class LocalWCS(UniformWCS):
    """A LocalWCS is a `UniformWCS` in which (0,0) in image coordinates is at the same place
    as (0,0) in world coordinates
    """

    @implements(_galsim.wcs.LocalWCS.isLocal)
    def isLocal(self):
        return True

    # The origins are definitionally (0,0) for these.  So just define them here.
    @property
    def origin(self):
        """The image coordinate position to use as the origin."""
        return PositionD(0, 0)

    @property
    def world_origin(self):
        """The world coordinate position to use as the origin."""
        return PositionD(0, 0)

    # For LocalWCS, there is no origin to worry about.
    def _posToWorld(self, image_pos, color):
        x = image_pos.x
        y = image_pos.y
        return PositionD(self._u(x, y), self._v(x, y))

    def _xyTouv(self, x, y, color):
        return self._u(x, y), self._v(x, y)

    # For LocalWCS, there is no origin to worry about.
    def _posToImage(self, world_pos, color):
        u = world_pos.x
        v = world_pos.y
        return PositionD(self._x(u, v), self._y(u, v))

    def _uvToxy(self, u, v, color):
        return self._x(u, v), self._y(u, v)

    # For LocalWCS, this is of course trivial.
    def _local(self, image_pos, color):
        return self


class CelestialWCS(BaseWCS):
    """A CelestialWCS is a `BaseWCS` whose world coordinates are on the celestial sphere.
    We use the `CelestialCoord` class for the world coordinates.
    """

    @property
    def _isCelestial(self):
        return True

    # CelestialWCS classes still have origin, but not world_origin.
    @property
    def x0(self):
        """The x coordinate of self.origin."""
        return self.origin.x

    @property
    def y0(self):
        """The y coordinate of self.origin."""
        return self.origin.y

    def xyToradec(self, x, y, units=None, color=None):
        """Convert x,y from image coordinates to world coordinates.

        This is equivalent to ``wcs.toWorld(x,y, units=units)``.

        It is also equivalent to ``wcs.posToWorld(galsim.PositionD(x,y)).rad`` when x and y are
        scalars if units is 'radians'; however, this routine allows x and y to be numpy arrays,
        in which case, the calculation will be vectorized, which is often much faster than using
        the pos interface.

        Parameters:
            x:          The x value(s) in image coordinates
            y:          The y value(s) in image coordinates
            units:      (Only valid for `CelestialWCS`, in which case it is required)
                        The units to use for the returned ra, dec values.
            color:      For color-dependent WCS's, the color term to use. [default: None]

        Returns:
            ra, dec
        """
        if color is None:
            color = self._color
        if units is None:
            raise TypeError("units is required for CelestialWCS types")
        elif isinstance(units, str):
            units = AngleUnit.from_name(units)
        elif not isinstance(units, AngleUnit):
            raise GalSimValueError(
                "units must be either an AngleUnit or a string",
                units,
                AngleUnit.valid_names,
            )
        return self._xyToradec(x, y, units, color)

    def radecToxy(self, ra, dec, units, color=None):
        """Convert ra,dec from world coordinates to image coordinates.

        This is equivalent to ``wcs.toImage(ra,dec, units=units)``.

        It is also equivalent to ``wcs.posToImage(galsim.CelestialCoord(ra * units, dec * units))``
        when ra and dec are scalars; however, this routine allows ra and dec to be numpy arrays,
        in which case, the calculation will be vectorized, which is often much faster than using
        the pos interface.

        Parameters:
            ra:         The ra value(s) in world coordinates
            dec:        The dec value(s) in world coordinates
            units:      The units to use for the input ra, dec values.
            color:      For color-dependent WCS's, the color term to use. [default: None]
        """
        if color is None:
            color = self._color
        if isinstance(units, str):
            units = AngleUnit.from_name(units)
        elif not isinstance(units, AngleUnit):
            raise GalSimValueError(
                "units must be either an AngleUnit or a string",
                units,
                AngleUnit.valid_names,
            )
        return self._radecToxy(ra, dec, units, color)

    # This is a bit simpler than the EuclideanWCS version, since there is no world_origin.
    def _shiftOrigin(self, origin, world_origin, color):
        # We want the new wcs to have wcs.toWorld(x2,y2) match the current wcs.toWorld(0,0).
        # So,
        #
        #     u' = ufunc(x-x1, y-y1)        # In this case, there are no u0,v0
        #     v' = vfunc(x-x1, y-y1)
        #
        #     u'(x2,y2) = u(0,0)    v'(x2,y2) = v(0,0)
        #
        #     x2 - x1 = 0 - x0      y2 - y1 = 0 - y0
        # =>  x1 = x0 + x2          y1 = y0 + y2
        if world_origin is not None:
            raise TypeError("world_origin is invalid for CelestialWCS classes")
        origin += self.origin
        return self._newOrigin(origin)

    # If the class doesn't define something else, then we can approximate the local Jacobian
    # from finite differences for the derivatives of ra and dec.  Very similar to the
    # version for EuclideanWCS, but convert from dra, ddec to du, dv locallat at the given
    # position.
    def _local(self, image_pos, color):
        if image_pos is None:
            raise TypeError("origin must be a PositionD or PositionI argument")

        x0 = image_pos.x - self.x0
        y0 = image_pos.y - self.y0
        # Use dx,dy = 1 pixel for numerical derivatives
        dx = 1
        dy = 1

        xlist = jnp.array([x0, x0 + dx, x0 - dx, x0, x0], dtype=float)
        ylist = jnp.array([y0, y0, y0, y0 + dy, y0 - dy], dtype=float)
        ra, dec = self._radec(xlist, ylist, color)

        # Note: our convention is that ra increases to the left!
        # i.e. The u,v plane is the tangent plane as seen from Earth with +v pointing
        # north, and +u pointing west.
        # That means the du values are the negative of dra.
        cosdec = jnp.cos(dec[0])
        dudx = -0.5 * (ra[1] - ra[2]) / dx * cosdec
        dudy = -0.5 * (ra[3] - ra[4]) / dy * cosdec
        dvdx = 0.5 * (dec[1] - dec[2]) / dx
        dvdy = 0.5 * (dec[3] - dec[4]) / dy

        # These values are all in radians.  Convert to arcsec as per our usual standard.
        factor = radians / arcsec
        return JacobianWCS(dudx * factor, dudy * factor, dvdx * factor, dvdy * factor)

    # This is similar to the version for EuclideanWCS, but uses dra, ddec.
    # Again, it is much faster if the _radec function works with numpy arrays.
    def _makeSkyImage(self, image, sky_level, color):
        b = image.bounds
        nx = b.xmax - b.xmin + 1 + 2  # +2 more than in image to get row/col off each edge.
        ny = b.ymax - b.ymin + 1 + 2
        x, y = jnp.meshgrid(
            jnp.linspace(b.xmin - 1, b.xmax + 1, nx),
            jnp.linspace(b.ymin - 1, b.ymax + 1, ny),
        )
        x -= self.x0
        y -= self.y0
        ra, dec = self._radec(x.ravel(), y.ravel(), color)
        ra = jnp.reshape(ra, x.shape)
        dec = jnp.reshape(dec, x.shape)

        # Use the finite differences to estimate the derivatives.
        cosdec = jnp.cos(dec[1 : ny - 1, 1 : nx - 1])
        dudx = -0.5 * (ra[1 : ny - 1, 2:nx] - ra[1 : ny - 1, 0 : nx - 2])
        dudy = -0.5 * (ra[2:ny, 1 : nx - 1] - ra[0 : ny - 2, 1 : nx - 1])
        # Check for discontinuities in ra.  ra can jump by 2pi, so when it does
        # add (or subtract) pi to dudx, which is dra/2
        dudx = jnp.where(dudx > 1, dudx - jnp.pi, dudx)
        dudx = jnp.where(dudx < -1, dudx + jnp.pi, dudx)
        dudy = jnp.where(dudy > 1, dudy - jnp.pi, dudy)
        dudy = jnp.where(dudy < -1, dudy + jnp.pi, dudy)
        # Now account for the cosdec factor
        dudx *= cosdec
        dudy *= cosdec
        dvdx = 0.5 * (dec[1 : ny - 1, 2:nx] - dec[1 : ny - 1, 0 : nx - 2])
        dvdy = 0.5 * (dec[2:ny, 1 : nx - 1] - dec[0 : ny - 2, 1 : nx - 1])

        area = jnp.abs(dudx * dvdy - dvdx * dudy)
        factor = radians / arcsec
        image._array = area * sky_level * factor**2

    # Simple.  Just call _radec.
    def _posToWorld(self, image_pos, color, project_center=None, projection="gnomonic"):
        x = image_pos.x - self.x0
        y = image_pos.y - self.y0
        ra, dec = self._radec(x, y, color)
        coord = CelestialCoord(ra * radians, dec * radians)
        if project_center is None:
            return coord
        else:
            u, v = project_center.project(coord, projection=projection)
            return PositionD(u / arcsec, v / arcsec)

    def _xyToradec(self, x, y, units, color):
        x = x - self.x0  # Not -=, since don't want to modify the input arrays in place.
        y = y - self.y0
        ra, dec = self._radec(x, y, color)
        ra *= radians / units
        dec *= radians / units
        return ra, dec

    # Also simple if _xy is implemented.  However, it is allowed to raise a NotImplementedError.
    def _posToImage(self, world_pos, color):
        ra = world_pos.ra.rad
        dec = world_pos.dec.rad
        x, y = self._xy(ra, dec, color)
        return PositionD(x, y) + self.origin

    def _radecToxy(self, ra, dec, units, color):
        ra = ra * (units / radians)
        dec = dec * (units / radians)
        x, y = self._xy(ra, dec, color)
        x += self.origin.x
        y += self.origin.y
        return x, y

    # Each class should define the __eq__ function.  Then __ne__ is obvious.
    def __ne__(self, other):
        return not self.__eq__(other)


#########################################################################################
#
# Local WCS classes are those where (x,y) = (0,0) corresponds to (u,v) = (0,0).
#
# We have the following local WCS classes:
#
#     PixelScale
#     ShearWCS
#     JacobianWCS
#
# They must define the following:
#
#     origin            attribute or property returning the origin
#     world_origin      attribute or property returning the world origin
#     _u                function returning u(x,y)
#     _v                function returning v(x,y)
#     _x                function returning x(u,v)
#     _y                function returning y(u,v)
#     _profileToWorld   function converting image_profile to world_profile
#     _profileToImage   function converting world_profile to image_profile
#     _pixelArea        function returning the pixel area
#     _minScale         function returning the minimum linear pixel scale
#     _maxScale         function returning the maximum linear pixel scale
#     _toJacobian       function returning an equivalent JacobianWCS
#     _writeHeader      function that writes the WCS to a fits header.
#     _readHeader       static function that reads the WCS from a fits header.
#     _newOrigin        function returning a non-local WCS corresponding to this WCS
#     copy              return a copy
#     __eq__            check if this equals another WCS
#     __repr__          convert to string
#
#########################################################################################


@implements(_galsim.PixelScale)
@register_pytree_node_class
class PixelScale(LocalWCS):
    _isPixelScale = True

    def __init__(self, scale):
        if isinstance(scale, BaseWCS):
            raise TypeError("Cannot initialize PixelScale from a BaseWCS")
        self._params = {"scale": scale}
        self._color = None

    @property
    def _scale(self):
        return self._params["scale"]

    # Help make sure PixelScale is read-only.
    @property
    def scale(self):
        """The pixel scale"""
        return self._scale

    def _u(self, x, y, color=None):
        return x * self._scale

    def _v(self, x, y, color=None):
        return y * self._scale

    def _x(self, u, v, color=None):
        return u / self._scale

    def _y(self, u, v, color=None):
        return v / self._scale

    def _profileToWorld(self, image_profile, flux_ratio, offset):
        return _Transform(
            image_profile,
            (self._scale, 0.0, 0.0, self._scale),
            flux_ratio=self._scale**-2 * flux_ratio,
            offset=offset,
        )

    def _profileToImage(self, world_profile, flux_ratio, offset):
        return _Transform(
            world_profile,
            (1.0 / self._scale, 0.0, 0.0, 1.0 / self._scale),
            flux_ratio=self._scale**2 * flux_ratio,
            offset=offset,
        )

    def _shearToWorld(self, image_shear):
        # These are trivial for PixelScale.
        return image_shear

    def _shearToImage(self, world_shear):
        return world_shear

    def _pixelArea(self):
        return self._scale**2

    def _minScale(self):
        return self._scale

    def _maxScale(self):
        return self._scale

    def _inverse(self):
        return PixelScale(1.0 / self._scale)

    def _toJacobian(self):
        return JacobianWCS(self._scale, 0.0, 0.0, self._scale)

    def _writeHeader(self, header, bounds):
        header["GS_WCS"] = ("PixelScale", "GalSim WCS name")
        header["GS_SCALE"] = (cast_to_python_float(self.scale), "GalSim image scale")
        return self.affine()._writeLinearWCS(header, bounds)

    @staticmethod
    def _readHeader(header):
        scale = header["GS_SCALE"]
        return PixelScale(scale)

    def _newOrigin(self, origin, world_origin):
        return OffsetWCS(self._scale, origin, world_origin)

    def copy(self):
        return PixelScale(self._scale)

    def __eq__(self, other):
        return self is other or (isinstance(other, PixelScale) and self.scale == other.scale)

    def __repr__(self):
        return "galsim.PixelScale(%r)" % ensure_hashable(self.scale)

    def __hash__(self):
        return hash(repr(self))


@implements(_galsim.ShearWCS)
@register_pytree_node_class
class ShearWCS(LocalWCS):
    _req_params = {"scale": float, "shear": Shear}

    def __init__(self, scale, shear):
        self._color = None
        self._scale = scale
        self._shear = shear
        self._g1 = shear.g1
        self._g2 = shear.g2

    @property
    def _gsq(self):
        return self._g1**2 + self._g2**2

    @property
    def _gfactor(self):
        return 1.0 / jnp.sqrt(1.0 - self._gsq)

    # Help make sure ShearWCS is read-only.
    @property
    def scale(self):
        """The pixel scale."""
        return self._scale

    @property
    def shear(self):
        """The applied `Shear`."""
        return self._shear

    def _u(self, x, y, color=None):
        u = x * (1.0 - self._g1) - y * self._g2
        u *= self._gfactor * self._scale
        return u

    def _v(self, x, y, color=None):
        v = y * (1.0 + self._g1) - x * self._g2
        v *= self._gfactor * self._scale
        return v

    def _x(self, u, v, color=None):
        x = u * (1.0 + self._g1) + v * self._g2
        x *= self._gfactor / self._scale
        return x

    def _y(self, u, v, color=None):
        y = v * (1.0 - self._g1) + u * self._g2
        y *= self._gfactor / self._scale
        return y

    def _profileToWorld(self, image_profile, flux_ratio, offset):
        return image_profile.dilate(self._scale).shear(-self.shear).shift(offset) * flux_ratio

    def _profileToImage(self, world_profile, flux_ratio, offset):
        return world_profile.dilate(1.0 / self._scale).shear(self.shear).shift(offset) * flux_ratio

    def _shearToWorld(self, image_shear):
        # This isn't worth customizing.  Just use the jacobian.
        return self._toJacobian()._shearToWorld(image_shear)

    def _shearToImage(self, world_shear):
        return self._toJacobian()._shearToImage(world_shear)

    def _pixelArea(self):
        return self._scale**2

    def _minScale(self):
        return self._scale * (1.0 - jnp.sqrt(self._gsq)) * self._gfactor

    def _maxScale(self):
        # max stretch is (1+|g|) / sqrt(1-|g|^2)
        return self._scale * (1.0 + jnp.sqrt(self._gsq)) * self._gfactor

    def _inverse(self):
        return ShearWCS(1.0 / self._scale, -self._shear)

    def _toJacobian(self):
        return JacobianWCS(
            (1.0 - self._g1) * self._scale * self._gfactor,
            -self._g2 * self._scale * self._gfactor,
            -self._g2 * self._scale * self._gfactor,
            (1.0 + self._g1) * self._scale * self._gfactor,
        )

    def _newOrigin(self, origin, world_origin):
        return OffsetShearWCS(self._scale, self._shear, origin, world_origin)

    def _writeHeader(self, header, bounds):
        header["GS_WCS"] = ("ShearWCS", "GalSim WCS name")
        header["GS_SCALE"] = (cast_to_python_float(self.scale), "GalSim image scale")
        header["GS_G1"] = (cast_to_python_float(self.shear.g1), "GalSim image shear g1")
        header["GS_G2"] = (cast_to_python_float(self.shear.g2), "GalSim image shear g2")
        return self.affine()._writeLinearWCS(header, bounds)

    def copy(self):
        return ShearWCS(self._scale, self._shear)

    def __eq__(self, other):
        return self is other or (isinstance(other, ShearWCS) and self.scale == other.scale and self.shear == other.shear)

    def __repr__(self):
        return "galsim.ShearWCS(%r, %r)" % (ensure_hashable(self.scale), self.shear)

    def __hash__(self):
        return hash(repr(self))

    def tree_flatten(self):
        children = (self.scale, self.shear)
        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@implements(_galsim.JacobianWCS)
@register_pytree_node_class
class JacobianWCS(LocalWCS):
    def __init__(self, dudx, dudy, dvdx, dvdy):
        self._color = None
        self._params = {"dudx": dudx, "dudy": dudy, "dvdx": dvdx, "dvdy": dvdy}

    @property
    def _det(self):
        return self.dudx * self.dvdy - self.dudy * self.dvdx

    # Help make sure JacobianWCS is read-only.
    @property
    def dudx(self):
        """du/dx"""
        return self._params["dudx"]

    @property
    def dudy(self):
        """du/dy"""
        return self._params["dudy"]

    @property
    def dvdx(self):
        """dv/dx"""
        return self._params["dvdx"]

    @property
    def dvdy(self):
        """dv/dy"""
        return self._params["dvdy"]

    def _u(self, x, y, color=None):
        return self.dudx * x + self.dudy * y

    def _v(self, x, y, color=None):
        return self.dvdx * x + self.dvdy * y

    def _x(self, u, v, color=None):
        #  J = ( dudx  dudy )
        #      ( dvdx  dvdy )
        #  J^-1 = (1/det) (  dvdy  -dudy )
        #                 ( -dvdx   dudx )
        return (self.dvdy * u - self.dudy * v) / self._det

    def _y(self, u, v, color=None):
        return (-self.dvdx * u + self.dudx * v) / self._det

    def _profileToWorld(self, image_profile, flux_ratio, offset):
        return _Transform(
            image_profile,
            (self.dudx, self.dudy, self.dvdx, self.dvdy),
            flux_ratio=flux_ratio / self._pixelArea(),
            offset=offset,
        )

    def _profileToImage(self, world_profile, flux_ratio, offset):
        return _Transform(
            world_profile,
            (
                self.dvdy / self._det,
                -self.dudy / self._det,
                -self.dvdx / self._det,
                self.dudx / self._det,
            ),
            flux_ratio=flux_ratio * self._pixelArea(),
            offset=offset,
        )

    def _shearToWorld(self, image_shear):
        # Code from https://github.com/rmjarvis/DESWL/blob/y3a1-v23/psf/run_piff.py#L691
        e1 = image_shear.e1
        e2 = image_shear.e2

        M = jnp.array([[1 + e1, e2], [e2, 1 - e1]])
        J = self.getMatrix()
        M = J.dot(M).dot(J.T)

        e1 = (M[0, 0] - M[1, 1]) / (M[0, 0] + M[1, 1])
        e2 = (2.0 * M[0, 1]) / (M[0, 0] + M[1, 1])

        return Shear(e1=e1, e2=e2)

    def _shearToImage(self, world_shear):
        # Same as above but inverse J matrix.
        return self._inverse()._shearToWorld(world_shear)

    def _pixelArea(self):
        return abs(self._det)

    def getMatrix(self):
        """Get the Jacobian as a NumPy matrix:

        numpy.array( [[ dudx, dudy ],
                      [ dvdx, dvdy ]] )
        """
        return jnp.array([[self.dudx, self.dudy], [self.dvdx, self.dvdy]], dtype=float)

    @implements(_galsim.JacobianWCS.getDecomposition)
    def getDecomposition(self):
        from .angle import radians

        # First we need to see whether or not the transformation includes a flip.  The evidence
        # for a flip is that the determinant is negative.
        # if self._det == 0.:
        #     raise GalSimError("Transformation is singular")

        dudx, dudy, dvdx, dvdy = jnp.where(
            self._det < 0.0,
            jnp.array([self.dudy, self.dudx, self.dvdy, self.dvdx]),
            jnp.array([self.dudx, self.dudy, self.dvdx, self.dvdy]),
        )

        flip = self._det < 0.0
        scale = jnp.sqrt(jnp.abs(self._det))

        # A small bit of algebraic manipulations yield the following two equations that let us
        # determine theta:
        #
        # (dudx + dvdy) = 2 scale/sqrt(1-g^2) cos(t)
        # (dvdx - dudy) = 2 scale/sqrt(1-g^2) sin(t)

        C = dudx + dvdy
        S = dvdx - dudy
        theta = jnp.arctan2(S, C) * radians

        # The next step uses the following equations that you can get from a bit more algebra:
        #
        # cost (dudx - dvdy) - sint (dudy + dvdx) = 2 scale/sqrt(1-g^2) g1
        # sint (dudx - dvdy) + cost (dudy + dvdx) = 2 scale/sqrt(1-g^2) g2

        factor = C * C + S * S  # factor = (2 scale/sqrt(1-g^2))^2
        C /= factor  # C is now cost / (2 scale/sqrt(1-g^2))
        S /= factor  # S is now sint / (2 scale/sqrt(1-g^2))

        g1 = C * (dudx - dvdy) - S * (dudy + dvdx)
        g2 = S * (dudx - dvdy) + C * (dudy + dvdx)

        return scale, Shear(g1=g1, g2=g2), theta, flip

    def _minScale(self):
        # min scale is scale * (1-|g|) / sqrt(1-|g|^2)
        # We could get this from the decomposition, but some algebra finds that this
        # reduces to the following calculation:
        # NB: The unit tests test for the equivalence with the above formula.
        h1 = jnp.sqrt((self.dudx + self.dvdy) ** 2 + (self.dudy - self.dvdx) ** 2)
        h2 = jnp.sqrt((self.dudx - self.dvdy) ** 2 + (self.dudy + self.dvdx) ** 2)
        return 0.5 * abs(h1 - h2)

    def _maxScale(self):
        # min scale is scale * (1+|g|) / sqrt(1-|g|^2)
        # which is equivalent to the following:
        # NB: The unit tests test for the equivalence with the above formula.
        h1 = jnp.sqrt((self.dudx + self.dvdy) ** 2 + (self.dudy - self.dvdx) ** 2)
        h2 = jnp.sqrt((self.dudx - self.dvdy) ** 2 + (self.dudy + self.dvdx) ** 2)
        return 0.5 * (h1 + h2)

    def _inverse(self):
        return JacobianWCS(
            self.dvdy / self._det,
            -self.dudy / self._det,
            -self.dvdx / self._det,
            self.dudx / self._det,
        )

    def _toJacobian(self):
        return self

    def _writeHeader(self, header, bounds):
        header["GS_WCS"] = ("JacobianWCS", "GalSim WCS name")
        return self.affine()._writeLinearWCS(header, bounds)

    @staticmethod
    def _readHeader(header):
        dudx = header.get("CD1_1", 1.0)
        dudy = header.get("CD1_2", 0.0)
        dvdx = header.get("CD2_1", 0.0)
        dvdy = header.get("CD2_2", 1.0)
        return JacobianWCS(dudx, dudy, dvdx, dvdy)

    def _newOrigin(self, origin, world_origin):
        return AffineTransform(self.dudx, self.dudy, self.dvdx, self.dvdy, origin, world_origin)

    def copy(self):
        return JacobianWCS(self.dudx, self.dudy, self.dvdx, self.dvdy)

    def __eq__(self, other):
        return self is other or (
            isinstance(other, JacobianWCS)
            and self.dudx == other.dudx
            and self.dudy == other.dudy
            and self.dvdx == other.dvdx
            and self.dvdy == other.dvdy
        )

    def __repr__(self):
        return "galsim.JacobianWCS(%r, %r, %r, %r)" % (
            ensure_hashable(self.dudx),
            ensure_hashable(self.dudy),
            ensure_hashable(self.dvdx),
            ensure_hashable(self.dvdy),
        )

    def __hash__(self):
        return hash(repr(self))


#########################################################################################
#
# Non-local UniformWCS classes are those where (x,y) = (0,0) does not (necessarily)
# correspond to (u,v) = (0,0).
#
# We have the following non-local UniformWCS classes:
#
#     OffsetWCS
#     OffsetShearWCS
#     AffineTransform
#
# They must define the following:
#
#     origin            attribute or property returning the origin
#     world_origin      attribute or property returning the world origin
#     _local_wcs        property returning a local WCS with the same pixel shape
#     _writeHeader      function that writes the WCS to a fits header.
#     _readHeader       static function that reads the WCS from a fits header.
#     _newOrigin        function returning the saem WCS, but with new origin, world_origin
#     copy              return a copy
#     __repr__          convert to string
#
#########################################################################################


@implements(_galsim.OffsetWCS)
@register_pytree_node_class
class OffsetWCS(UniformWCS):
    _isPixelScale = True

    def __init__(self, scale, origin=None, world_origin=None):
        self._color = None
        self._set_origin(origin, world_origin)
        self._scale = scale
        self._params = {
            "scale": scale,
            "origin": self._origin,
            "world_origin": self._world_origin,
        }
        self._local_wcs = PixelScale(scale)

    @property
    def scale(self):
        """The pixel scale."""
        return self._scale

    @property
    def origin(self):
        """The image coordinate position to use as the origin."""
        return self._origin

    @property
    def world_origin(self):
        """The world coordinate position to use as the origin."""
        return self._world_origin

    def _writeHeader(self, header, bounds):
        header["GS_WCS"] = ("OffsetWCS", "GalSim WCS name")
        header["GS_SCALE"] = (cast_to_python_float(self.scale), "GalSim image scale")
        header["GS_X0"] = (cast_to_python_float(self.origin.x), "GalSim image origin x")
        header["GS_Y0"] = (cast_to_python_float(self.origin.y), "GalSim image origin y")
        header["GS_U0"] = (
            cast_to_python_float(self.world_origin.x),
            "GalSim world origin u",
        )
        header["GS_V0"] = (
            cast_to_python_float(self.world_origin.y),
            "GalSim world origin v",
        )
        return self.affine()._writeLinearWCS(header, bounds)

    @staticmethod
    def _readHeader(header):
        scale = header["GS_SCALE"]
        x0 = header["GS_X0"]
        y0 = header["GS_Y0"]
        u0 = header["GS_U0"]
        v0 = header["GS_V0"]
        return OffsetWCS(scale, PositionD(x0, y0), PositionD(u0, v0))

    def _newOrigin(self, origin, world_origin):
        return OffsetWCS(self._scale, origin, world_origin)

    def copy(self):
        return OffsetWCS(self._scale, self.origin, self.world_origin)

    def __repr__(self):
        return "galsim.OffsetWCS(%r, %r, %r)" % (
            ensure_hashable(self.scale),
            self.origin,
            self.world_origin,
        )

    def __hash__(self):
        return hash(repr(self))


@implements(_galsim.OffsetShearWCS)
@register_pytree_node_class
class OffsetShearWCS(UniformWCS):
    _req_params = {"scale": float, "shear": Shear}
    _opt_params = {"origin": PositionD, "world_origin": PositionD}

    def __init__(self, scale, shear, origin=None, world_origin=None):
        self._color = None
        self._set_origin(origin, world_origin)
        # The shear stuff is not too complicated, but enough so that it is worth
        # encapsulating in the ShearWCS class.  So here, we just create one of those
        # and we'll pass along any shear calculations to that.
        self._local_wcs = ShearWCS(scale, shear)

    @property
    def scale(self):
        """The pixel scale."""
        return self._local_wcs.scale

    @property
    def shear(self):
        """The applied `Shear`."""
        return self._local_wcs.shear

    @property
    def origin(self):
        """The image coordinate position to use as the origin."""
        return self._origin

    @property
    def world_origin(self):
        """The world coordinate position to use as the origin."""
        return self._world_origin

    def _newOrigin(self, origin, world_origin):
        return OffsetShearWCS(self.scale, self.shear, origin, world_origin)

    def _writeHeader(self, header, bounds):
        header["GS_WCS"] = ("OffsetShearWCS", "GalSim WCS name")
        header["GS_SCALE"] = (cast_to_python_float(self.scale), "GalSim image scale")
        header["GS_G1"] = (cast_to_python_float(self.shear.g1), "GalSim image shear g1")
        header["GS_G2"] = (cast_to_python_float(self.shear.g2), "GalSim image shear g2")
        header["GS_X0"] = (
            cast_to_python_float(self.origin.x),
            "GalSim image origin x coordinate",
        )
        header["GS_Y0"] = (
            cast_to_python_float(self.origin.y),
            "GalSim image origin y coordinate",
        )
        header["GS_U0"] = (
            cast_to_python_float(self.world_origin.x),
            "GalSim world origin u coordinate",
        )
        header["GS_V0"] = (
            cast_to_python_float(self.world_origin.y),
            "GalSim world origin v coordinate",
        )
        return self.affine()._writeLinearWCS(header, bounds)

    def copy(self):
        return OffsetShearWCS(self.scale, self.shear, self.origin, self.world_origin)

    def __repr__(self):
        return "galsim.OffsetShearWCS(%r, %r, %r, %r)" % (
            ensure_hashable(self.scale),
            self.shear,
            self.origin,
            self.world_origin,
        )

    def __hash__(self):
        return hash(repr(self))

    def tree_flatten(self):
        children = (self.scale, self.shear, self.origin, self.world_origin)
        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@implements(_galsim.AffineTransform)
@register_pytree_node_class
class AffineTransform(UniformWCS):
    def __init__(self, dudx, dudy, dvdx, dvdy, origin=None, world_origin=None):
        self._color = None
        self._set_origin(origin, world_origin)
        self._params = {
            "dudx": dudx,
            "dudy": dudy,
            "dvdx": dvdx,
            "dvdy": dvdy,
            "origin": self._origin,
            "world_origin": self._world_origin,
        }
        # As with OffsetShearWCS, we store a JacobianWCS, rather than reimplement everything.
        self._local_wcs = JacobianWCS(dudx, dudy, dvdx, dvdy)

    @property
    def dudx(self):
        """du/dx"""
        return self._local_wcs.dudx

    @property
    def dudy(self):
        """du/dy"""
        return self._local_wcs.dudy

    @property
    def dvdx(self):
        """dv/dx"""
        return self._local_wcs.dvdx

    @property
    def dvdy(self):
        """dv/dy"""
        return self._local_wcs.dvdy

    @property
    def origin(self):
        """The image coordinate position to use as the origin."""
        return self._origin

    @property
    def world_origin(self):
        """The world coordinate position to use as the origin."""
        return self._world_origin

    def _writeHeader(self, header, bounds):
        header["GS_WCS"] = ("AffineTransform", "GalSim WCS name")
        return self._writeLinearWCS(header, bounds)

    def _writeLinearWCS(self, header, bounds):
        header["CTYPE1"] = ("LINEAR", "name of the world coordinate axis")
        header["CTYPE2"] = ("LINEAR", "name of the world coordinate axis")
        header["CRVAL1"] = (
            cast_to_python_float(self.u0),
            "world coordinate at reference pixel = u0",
        )
        header["CRVAL2"] = (
            cast_to_python_float(self.v0),
            "world coordinate at reference pixel = v0",
        )
        header["CRPIX1"] = (
            cast_to_python_float(self.x0),
            "image coordinate of reference pixel = x0",
        )
        header["CRPIX2"] = (
            cast_to_python_float(self.y0),
            "image coordinate of reference pixel = y0",
        )
        header["CD1_1"] = (cast_to_python_float(self.dudx), "CD1_1 = dudx")
        header["CD1_2"] = (cast_to_python_float(self.dudy), "CD1_2 = dudy")
        header["CD2_1"] = (cast_to_python_float(self.dvdx), "CD2_1 = dvdx")
        header["CD2_2"] = (cast_to_python_float(self.dvdy), "CD2_2 = dvdy")
        return header

    @staticmethod
    def _readHeader(header):
        # We try to make this work to produce a linear WCS, no matter what kinds of key words
        # are in the header.
        if "CD1_1" in header:
            # The others should be too, but use get with a default to be safe
            dudx = header.get("CD1_1", 1.0)
            dudy = header.get("CD1_2", 0.0)
            dvdx = header.get("CD2_1", 0.0)
            dvdy = header.get("CD2_2", 1.0)
        else:
            dudx = header.get("CDELT1", 1.0)
            dudy = 0.0
            dvdx = 0.0
            dvdy = header.get("CDELT2", 1.0)
        x0 = header.get("CRPIX1", 0.0)
        y0 = header.get("CRPIX2", 0.0)
        u0 = header.get("CRVAL1", 0.0)
        v0 = header.get("CRVAL2", 0.0)

        return AffineTransform(dudx, dudy, dvdx, dvdy, PositionD(x0, y0), PositionD(u0, v0))

    def _newOrigin(self, origin, world_origin):
        return AffineTransform(self.dudx, self.dudy, self.dvdx, self.dvdy, origin, world_origin)

    def copy(self):
        return AffineTransform(self.dudx, self.dudy, self.dvdx, self.dvdy, self.origin, self.world_origin)

    def __repr__(self):
        return ("galsim.AffineTransform(%r, %r, %r, %r, origin=%r, world_origin=%r)") % (
            ensure_hashable(self.dudx),
            ensure_hashable(self.dudy),
            ensure_hashable(self.dvdx),
            ensure_hashable(self.dvdy),
            self.origin,
            self.world_origin,
        )

    def __hash__(self):
        return hash(repr(self))


def compatible(wcs1, wcs2):
    """
    A utility to check the compatibility of two WCS.  In particular, if two WCS are consistent with
    each other modulo a shifted origin, we consider them to be compatible, even though they are not
    equal.
    """
    if wcs1._isUniform and wcs2._isUniform:
        return wcs1.jacobian() == wcs2.jacobian()
    else:
        return wcs1 == wcs2.shiftOrigin(wcs1.origin, wcs1.world_origin)


def readFromFitsHeader(header, suppress_warning=True):
    """Read a WCS function from a FITS header.

    This is normally called automatically from within the `galsim.fits.read` function, but
    you can also call it directly as::

        wcs, origin = galsim.wcs.readFromFitsHeader(header)

    If the file was originally written by GalSim using one of the galsim.fits.write() functions,
    then this should always succeed in reading back in the original WCS.  It may not end up
    as exactly the same class as the original, but the underlying world coordinate system
    transformation should be preserved.

    .. note::
        For `UVFunction` and `RaDecFunction`, if the functions that were written to the FITS
        header were real python functions (rather than a string that is converted to a function),
        then the mechanism we use to write to the header and read it back in has some limitations:

        1. It apparently only works for cpython implementations.
        2. It probably won't work to write from one version of python and read from another.
           (At least for major version differences.)
        3. If the function uses globals, you'll need to make sure the globals are present
           when you read it back in as well, or it probably won't work.
        4. It looks really ugly in the header.
        5. We haven't thought much about the security implications of this, so beware using
           GalSim to open FITS files from untrusted sources.

    If the file was not written by GalSim, then this code will do its best to read the
    WCS information in the FITS header.  Depending on what kind of WCS is encoded in the
    header, this may or may not be successful.

    If there is no WCS information in the header, then this will default to a pixel scale
    of 1.

    In addition to the wcs, this function will also return the image origin that the WCS
    is assuming for the image.  If the file was originally written by GalSim, this should
    correspond to the original image origin.  If not, it will default to (1,1).

    Parameters:
        header:             The fits header with the WCS information.
        suppress_warning:   Whether to suppress a warning that the WCS could not be read from the
                            FITS header, so the WCS defaulted to either a `PixelScale` or
                            `AffineTransform`. [default: True]

    Returns:
        a tuple (wcs, origin) of the wcs from the header and the image origin.
    """
    from . import fits
    from .fitswcs import FitsWCS

    if not isinstance(header, fits.FitsHeader):
        header = fits.FitsHeader(header)
    xmin = header.get("GS_XMIN", 1)
    ymin = header.get("GS_YMIN", 1)
    origin = PositionI(xmin, ymin)
    wcs_name = header.get("GS_WCS", None)
    if wcs_name is not None:
        gdict = globals().copy()
        exec("import jax_galsim", gdict)
        wcs_type = eval("jax_galsim." + wcs_name, gdict)
        wcs = wcs_type._readHeader(header)
    else:
        # If we aren't told which type to use, this should find something appropriate
        wcs = FitsWCS(header=header, suppress_warning=suppress_warning)

    if xmin != 1 or ymin != 1:
        # ds9 always assumes the image has an origin at (1,1), so convert back to actual
        # xmin, ymin if necessary.
        delta = PositionI(xmin - 1, ymin - 1)
        wcs = wcs.shiftOrigin(delta)

    return wcs, origin
