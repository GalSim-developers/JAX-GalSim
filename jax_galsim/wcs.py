import galsim as _galsim
import jax.numpy as jnp
from jax._src.numpy.util import _wraps
from jax.tree_util import register_pytree_node_class

from jax_galsim.gsobject import GSObject
from jax_galsim.position import Position, PositionD
from jax_galsim.transform import _Transform


# We inherit from the reference BaseWCS and only redefine the methods that
# make references to jax_galsim objects.
class BaseWCS(_galsim.BaseWCS):
    @_wraps(_galsim.BaseWCS.toWorld)
    def toWorld(self, *args, **kwargs):
        if len(args) == 1:
            if isinstance(args[0], GSObject):
                return self.profileToWorld(*args, **kwargs)
            else:
                return self.posToWorld(*args, **kwargs)
        elif len(args) == 2:
            if self.isCelestial():
                return self.xyToradec(*args, **kwargs)
            else:
                return self.xyTouv(*args, **kwargs)
        else:
            raise TypeError("toWorld() takes either 1 or 2 positional arguments")

    @_wraps(_galsim.BaseWCS.posToWorld)
    def posToWorld(self, image_pos, color=None, **kwargs):
        if color is None:
            color = self._color
        if not isinstance(image_pos, Position):
            raise TypeError("image_pos must be a PositionD or PositionI argument")
        return self._posToWorld(image_pos, color=color, **kwargs)

    @_wraps(_galsim.BaseWCS.profileToWorld)
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
        return self.local(image_pos, world_pos, color=color)._profileToWorld(
            image_profile, flux_ratio, PositionD(offset)
        )

    @_wraps(_galsim.BaseWCS.toImage)
    def toImage(self, *args, **kwargs):
        if len(args) == 1:
            if isinstance(args[0], GSObject):
                return self.profileToImage(*args, **kwargs)
            else:
                return self.posToImage(*args, **kwargs)
        elif len(args) == 2:
            if self.isCelestial():
                return self.radecToxy(*args, **kwargs)
            else:
                return self.uvToxy(*args, **kwargs)
        else:
            raise TypeError("toImage() takes either 1 or 2 positional arguments")

    @_wraps(_galsim.BaseWCS.posToImage)
    def posToImage(self, world_pos, color=None):
        if color is None:
            color = self._color
        # TODO: update this to jax version of CelestialCoord when available
        if self.isCelestial() and not isinstance(world_pos, _galsim.CelestialCoord):
            raise TypeError("world_pos must be a CelestialCoord argument")
        elif not self.isCelestial() and not isinstance(world_pos, Position):
            raise TypeError("world_pos must be a PositionD or PositionI argument")
        return self._posToImage(world_pos, color=color)

    @_wraps(_galsim.BaseWCS.profileToImage)
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
        return self.local(image_pos, world_pos, color=color)._profileToImage(
            world_profile, flux_ratio, PositionD(offset)
        )

    @_wraps(_galsim.BaseWCS.local)
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

    @_wraps(_galsim.BaseWCS.affine)
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
            return jac.withOrigin(image_pos)
        else:
            if world_pos is None:
                world_pos = self.toWorld(image_pos, color=color)
            return jac.withOrigin(image_pos, world_pos, color=color)

    @_wraps(_galsim.BaseWCS.shiftOrigin)
    def shiftOrigin(self, origin, world_origin=None, color=None):
        if color is None:
            color = self._color
        if not isinstance(origin, Position):
            raise TypeError("origin must be a PositionD or PositionI argument")
        return self._shiftOrigin(origin, world_origin, color)

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
        """This function flattens the GSObject into a list of children
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
        image.array[:, :] = area * sky_level

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


@_wraps(_galsim.PixelScale)
@register_pytree_node_class
class PixelScale(LocalWCS):
    def __init__(self, scale):
        self._params = {"scale": scale}
        self._color = None

    @property
    def _scale(self):
        return jnp.asarray(self._params["scale"])

    # Help make sure PixelScale is read-only.
    @property
    def scale(self):
        """The pixel scale"""
        return self._scale

    def isPixelScale(self):
        return True

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
        header["GS_SCALE"] = (self.scale, "GalSim image scale")
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
        return "galsim.PixelScale(%r)" % self.scale

    def __hash__(self):
        return hash(repr(self))


@_wraps(_galsim.ShearWCS)
@register_pytree_node_class
class ShearWCS(LocalWCS):
    from .shear import Shear

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

    def _pixelArea(self):
        return self._scale**2

    def _minScale(self):
        return self._scale * (1.0 - jnp.sqrt(self._gsq)) * self._gfactor

    def _maxScale(self):
        # max stretch is (1+|g|) / sqrt(1-|g|^2)
        import math

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

    def copy(self):
        return ShearWCS(self._scale, self._shear)

    def __eq__(self, other):
        return self is other or (
            isinstance(other, ShearWCS) and self.scale == other.scale and self.shear == other.shear
        )

    def __repr__(self):
        return "galsim.ShearWCS(%r, %r)" % (self.scale, self.shear)

    def __hash__(self):
        return hash(repr(self))

    def tree_flatten(self):
        children = (self.scale, self.shear)
        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@_wraps(_galsim.JacobianWCS)
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

    def _pixelArea(self):
        return abs(self._det)

    def getMatrix(self):
        """Get the Jacobian as a JAX matrix:

        jax.numpy.array( [[ dudx, dudy ],
                          [ dvdx, dvdy ]] )
        """
        return jnp.array([[self.dudx, self.dudy], [self.dvdx, self.dvdy]], dtype=float)

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
            self.dudx,
            self.dudy,
            self.dvdx,
            self.dvdy,
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


@_wraps(_galsim.OffsetWCS)
@register_pytree_node_class
class OffsetWCS(UniformWCS):
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

    def isPixelScale(self):
        return True

    def _writeHeader(self, header, bounds):
        header["GS_WCS"] = ("OffsetWCS", "GalSim WCS name")
        header["GS_SCALE"] = (self.scale, "GalSim image scale")
        header["GS_X0"] = (self.origin.x, "GalSim image origin x")
        header["GS_Y0"] = (self.origin.y, "GalSim image origin y")
        header["GS_U0"] = (self.world_origin.x, "GalSim world origin u")
        header["GS_V0"] = (self.world_origin.y, "GalSim world origin v")
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
            self.scale,
            self.origin,
            self.world_origin,
        )

    def __hash__(self):
        return hash(repr(self))


@_wraps(_galsim.OffsetShearWCS)
@register_pytree_node_class
class OffsetShearWCS(UniformWCS):
    from .shear import Shear

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

    def copy(self):
        return OffsetShearWCS(self.scale, self.shear, self.origin, self.world_origin)

    def __repr__(self):
        return "galsim.OffsetShearWCS(%r, %r, %r, %r)" % (
            self.scale,
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


@_wraps(_galsim.AffineTransform)
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
        header["CRVAL1"] = (self.u0, "world coordinate at reference pixel = u0")
        header["CRVAL2"] = (self.v0, "world coordinate at reference pixel = v0")
        header["CRPIX1"] = (self.x0, "image coordinate of reference pixel = x0")
        header["CRPIX2"] = (self.y0, "image coordinate of reference pixel = y0")
        header["CD1_1"] = (self.dudx, "CD1_1 = dudx")
        header["CD1_2"] = (self.dudy, "CD1_2 = dudy")
        header["CD2_1"] = (self.dvdx, "CD2_1 = dvdx")
        header["CD2_2"] = (self.dvdy, "CD2_2 = dvdy")
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
        return AffineTransform(
            self.dudx, self.dudy, self.dvdx, self.dvdy, self.origin, self.world_origin
        )

    def __repr__(self):
        return ("galsim.AffineTransform(%r, %r, %r, %r, origin=%r, world_origin=%r)") % (
            self.dudx,
            self.dudy,
            self.dvdx,
            self.dvdy,
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
