import jax
import jax.numpy as jnp

from jax_galsim import Image, PositionD


@jax.jit
def draw_by_xValue(gsobject, image, jacobian=jnp.eye(2), offset=jnp.zeros(2), flux_scaling=1.0):
    """Utility function to draw a real-space GSObject into an Image."""
    # Applies flux scaling to compensate for pixel scale
    # See SBProfile.draw()
    flux_scaling *= image.scale**2

    # Create an array of coordinates
    coords = jnp.stack(image.get_pixel_centers(), axis=-1)
    coords = coords - image.true_center.array  # Subtract the true center
    coords = coords * image.scale  # Scale by the image pixel scale
    coords = coords - offset  # Add the offset

    # Apply the jacobian transformation
    inv_jacobian = jnp.linalg.inv(jacobian)
    _, logdet = jnp.linalg.slogdet(inv_jacobian)
    coords = jnp.dot(coords, inv_jacobian.T)
    flux_scaling *= jnp.exp(logdet)

    # Draw the object
    im = jax.vmap(lambda *args: gsobject._xValue(PositionD(*args)))(coords[..., 0], coords[..., 1])

    # Apply the flux scaling
    im = (im * flux_scaling).astype(image.dtype)

    # Return an image
    return Image(_array=im, _bounds=image.bounds, wcs=image.wcs, _dtype=image.dtype)


@jax.jit
def draw_by_kValue(gsobject, image, jacobian=jnp.eye(2)):
    # Create an array of coordinates
    coords = jnp.stack(image.get_pixel_centers(), axis=-1)
    coords = coords * image.scale  # Scale by the image pixel scale
    coords = jnp.dot(coords, jacobian.T)

    # Draw the object
    im = jax.vmap(lambda *args: gsobject._kValue(PositionD(*args)))(coords[..., 0], coords[..., 1])
    im = (im).astype(image.dtype)

    # Return an image
    return Image(_array=im, _bounds=image.bounds, wcs=image.wcs, _dtype=image.dtype)


@jax.jit
def apply_kImage_phases(gsobject, image, jacobian=jnp.eye(2)):
    # Create an array of coordinates
    kcoords = jnp.stack(image.get_pixel_centers(), axis=-1)
    kcoords = kcoords * image.scale  # Scale by the image pixel scale
    kcoords = jnp.dot(kcoords, jacobian.T)
    cenx, ceny = gsobject._offset.x, gsobject._offset.y

    #
    # flux Exp(-i (kx cx + kxy cx + kyx cy + ky cy ) )
    # NB: seems that tere is no jax.lax.polar equivalent to c++ std::polar function
    def phase(kpos):
        arg = kpos.x * cenx + kpos.y * ceny
        return jnp.cos(arg) - 1j * jnp.sin(arg)

    im_phase = jax.vmap(lambda *args: phase(PositionD(*args)))(kcoords[..., 0], kcoords[..., 1])

    return Image(
        _array=image.array * im_phase,
        _bounds=image.bounds,
        wcs=image.wcs,
        _dtype=image.dtype,
    )
