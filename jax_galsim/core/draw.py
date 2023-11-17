import galsim as _galsim
import jax
import jax.numpy as jnp
from jax._src.numpy.util import _wraps

from jax_galsim.random import PoissonDeviate


def draw_by_xValue(
    gsobject, image, jacobian=jnp.eye(2), offset=jnp.zeros(2), flux_scaling=1.0
):
    """Utility function to draw a real-space GSObject into an Image."""
    from jax_galsim import Image, PositionD

    # Applies flux scaling to compensate for pixel scale
    # See SBProfile.draw()
    flux_scaling *= image.scale**2

    # Create an array of coordinates
    coords = jnp.stack(image.get_pixel_centers(), axis=-1)
    coords = coords * image.scale  # Scale by the image pixel scale
    coords = coords - offset  # Add the offset

    # Apply the jacobian transformation
    inv_jacobian = jnp.linalg.inv(jacobian)
    _, logdet = jnp.linalg.slogdet(inv_jacobian)
    coords = jnp.dot(coords, inv_jacobian.T)
    flux_scaling *= jnp.exp(logdet)

    # Draw the object
    im = jax.vmap(lambda *args: gsobject._xValue(PositionD(*args)))(
        coords[..., 0], coords[..., 1]
    )

    # Apply the flux scaling
    im = (im * flux_scaling).astype(image.dtype)

    # Return an image
    return Image(array=im, bounds=image.bounds, wcs=image.wcs, check_bounds=False)


def draw_by_kValue(gsobject, image, jacobian=jnp.eye(2)):
    from jax_galsim import Image, PositionD

    # Create an array of coordinates
    coords = jnp.stack(image.get_pixel_centers(), axis=-1)
    coords = coords * image.scale  # Scale by the image pixel scale
    coords = jnp.dot(coords, jacobian)

    # Draw the object
    im = jax.vmap(lambda *args: gsobject._kValue(PositionD(*args)))(
        coords[..., 0], coords[..., 1]
    )
    im = (im).astype(image.dtype)

    # Return an image
    return Image(array=im, bounds=image.bounds, wcs=image.wcs, check_bounds=False)


def apply_kImage_phases(offset, image, jacobian=jnp.eye(2)):
    from jax_galsim import Image, PositionD

    # Create an array of coordinates
    kcoords = jnp.stack(image.get_pixel_centers(), axis=-1)
    kcoords = kcoords * image.scale  # Scale by the image pixel scale
    kcoords = jnp.dot(kcoords, jacobian)
    cenx, ceny = offset.x, offset.y

    #
    # flux Exp(-i (kx cx + kxy cx + kyx cy + ky cy ) )
    # NB: seems that tere is no jax.lax.polar equivalent to c++ std::polar function
    def phase(kpos):
        arg = -(kpos.x * cenx + kpos.y * ceny)
        return jnp.cos(arg) + 1j * jnp.sin(arg)

    im_phase = jax.vmap(lambda *args: phase(PositionD(*args)))(
        kcoords[..., 0], kcoords[..., 1]
    )
    return Image(
        array=image.array * im_phase,
        bounds=image.bounds,
        wcs=image.wcs,
        check_bounds=False,
    )


def sample_poisson_flux(flux, eta_factor, rng=None):
    """Sample the flux according to a Poisson distribution.

    Parameters:
        flux:           The flux of the GSObject (e.g., ``obj.flux``).
        eta_factor:     The flux per photon (e.g., ``obj._flux_per_photon``).
        rng:            If provided, a random number generator to use for photon shooting,
                        which may be any kind of `BaseDeviate` object.  If ``rng`` is None, one
                        will be automatically created, using the time as a seed.
                        [default: None]
    """
    # If we have both positive and negative photons, then the mix of these
    # already gives us some variation in the flux value from the variance
    # of how many are positive and how many are negative.
    # The number of negative photons varies as a binomial distribution.
    # <F-> = eta * Ntot * g
    # <F+> = (1-eta) * Ntot * g
    # <F+ - F-> = (1-2eta) * Ntot * g = flux
    # Var(F-) = eta * (1-eta) * Ntot * g^2
    # F+ = Ntot * g - F- is not an independent variable, so
    # Var(F+ - F-) = Var(Ntot*g - 2*F-)
    #              = 4 * Var(F-)
    #              = 4 * eta * (1-eta) * Ntot * g^2
    #              = 4 * eta * (1-eta) * flux
    # We want the variance to be equal to flux, so we need an extra:
    # delta Var = (1 - 4*eta + 4*eta^2) * flux
    #           = (1-2eta)^2 * flux
    absflux = abs(flux)
    mean = eta_factor * eta_factor * absflux
    pd = PoissonDeviate(rng, mean)
    pd_val = pd() - mean + absflux
    return pd_val


@_wraps(
    _galsim.GSObject._calculate_nphotons,
    lax_description="""\
Calculate the number of photons to shoot for photon shooting.

This routine is pure Python and is not JAX-compatible.

Parameters:
    flux:           The flux of the GSObject (e.g., ``obj.flux``).
    eta_factor:     The flux per photon (e.g., ``obj._flux_per_photon``).
    max_sb:         The maximum surface brightness of the object (e.g., ``obj.max_sb``).
    rng:            If provided, a random number generator to use for photon shooting,
                    which may be any kind of `BaseDeviate` object.  If ``rng`` is None, one
                    will be automatically created, using the time as a seed.
                    [default: None]
    max_extra_noise: If provided, the allowed extra noise in each pixel when photon
                    shooting.  This is only relevant if ``n_photons=0``, so the number of
                    photons is being automatically calculated.  In that case, if the image
                    noise is dominated by the sky background, then you can get away with
                    using fewer shot photons than the full ``n_photons = flux``.
                    Essentially each shot photon can have a ``flux > 1``, which increases
                    the noise in each pixel.  The ``max_extra_noise`` parameter specifies
                    how much extra noise per pixel is allowed because of this approximation.
                    A typical value for this might be ``max_extra_noise = sky_level / 100``
                    where ``sky_level`` is the flux per pixel due to the sky.  Note that
                    this uses a "variance" definition of noise, not a "sigma" definition.
                    [default: 0.]
    poisson_flux:   Whether to allow total object flux scaling to vary according to
                    Poisson statistics for ``n_photons`` samples when photon shooting.
                    [default: True, unless ``n_photons`` is given, in which case the default
                    is False]

""",
)
def calculate_n_photons(
    flux,
    eta_factor,
    max_sb,
    rng=None,
    max_extra_noise=0,
    poisson_flux=True,
):
    # For profiles that are positive definite, then N = flux. Easy.
    #
    # However, some profiles shoot some of their photons with negative flux. This means that
    # we need a few more photons to get the right S/N = sqrt(flux). Take eta to be the
    # fraction of shot photons that have negative flux.
    #
    # S^2 = (N+ - N-)^2 = (N+ + N- - 2N-)^2 = (Ntot - 2N-)^2 = Ntot^2(1 - 2 eta)^2
    # N^2 = Var(S) = (N+ + N-) = Ntot
    #
    # So flux = (S/N)^2 = Ntot (1-2eta)^2
    # Ntot = flux / (1-2eta)^2
    #
    # However, if each photon has a flux of 1, then S = (1-2eta) Ntot = flux / (1-2eta).
    # So in fact, each photon needs to carry a flux of g = 1-2eta to get the right
    # total flux.
    #
    # That's all the easy case. The trickier case is when we are sky-background dominated.
    # Then we can usually get away with fewer shot photons than the above.  In particular,
    # if the noise from the photon shooting is much less than the sky noise, then we can
    # use fewer shot photons and essentially have each photon have a flux > 1. This is ok
    # as long as the additional noise due to this approximation is "much less than" the
    # noise we'll be adding to the image for the sky noise.
    #
    # Let's still have Ntot photons, but now each with a flux of g. And let's look at the
    # noise we get in the brightest pixel that has a nominal total flux of Imax.
    #
    # The number of photons hitting this pixel will be Imax/flux * Ntot.
    # The variance of this number is the same thing (Poisson counting).
    # So the noise in that pixel is:
    #
    # N^2 = Imax/flux * Ntot * g^2
    #
    # And the signal in that pixel will be:
    #
    # S = Imax/flux * (N+ - N-) * g which has to equal Imax, so
    # g = flux / Ntot(1-2eta)
    # N^2 = Imax/Ntot * flux / (1-2eta)^2
    #
    # As expected, we see that lowering Ntot will increase the noise in that (and every
    # other) pixel.
    # The input max_extra_noise parameter is the maximum value of spurious noise we want
    # to allow.
    #
    # So setting N^2 = Imax + nu, we get
    #
    # Ntot = flux / (1-2eta)^2 / (1 + nu/Imax)
    # g = (1 - 2eta) * (1 + nu/Imax)
    #
    # Returns the total flux placed inside the image bounds by photon shooting.
    #

    if flux == 0.0:
        return 0, 1.0

    # The _flux_per_photon property is (1-2eta)
    # This factor will already be accounted for by the shoot function, so don't include
    # that as part of our scaling here.  There may be other adjustments though, so g=1 here.
    mod_flux = flux / (eta_factor * eta_factor)
    g = 1.0

    # If requested, let the target flux value vary as a Poisson deviate
    if poisson_flux:
        pd_val = sample_poisson_flux(flux, eta_factor, rng=rng)
        ratio = pd_val / abs(flux)
        g *= ratio
        mod_flux *= ratio

    n_photons = abs(mod_flux)
    if max_extra_noise > 0.0:
        gfactor = 1.0 + max_extra_noise / abs(max_sb)
        n_photons /= gfactor
        g *= gfactor

    # Make n_photons an integer.
    iN = int(n_photons + 0.5)

    return iN, g


# the code below is a jax version of calculate_nphotons
# that I am not sure if we need or not.
# saving in a comment for now

# def _calculate_nphotons(self, n_photons, poisson_flux, max_extra_noise, rng):
#     _n_photons, _g, _rng = jax.lax.cond(
#         self.flux == 0.0,
#         lambda n_photons, poisson_flux, max_extra_noise, rng: (0, 1.0, rng),
#         lambda n_photons, poisson_flux, max_extra_noise, rng: self._calculate_nphotons_nonzero(
#             n_photons, poisson_flux, max_extra_noise, rng
#         ),
#         n_photons,
#         poisson_flux,
#         max_extra_noise,
#         rng,
#     )
#     if rng is not None:
#         rng._state = _rng._state
#     return _n_photons, _g


# def _adjust_flux_g_poisson(self, poisson_flux, flux, mod_flux, eta_factor, rng, g):
#     from jax_galsim.random import PoissonDeviate

#     # If we have both positive and negative photons, then the mix of these
#     # already gives us some variation in the flux value from the variance
#     # of how many are positive and how many are negative.
#     # The number of negative photons varies as a binomial distribution.
#     # <F-> = eta * Ntot * g
#     # <F+> = (1-eta) * Ntot * g
#     # <F+ - F-> = (1-2eta) * Ntot * g = flux
#     # Var(F-) = eta * (1-eta) * Ntot * g^2
#     # F+ = Ntot * g - F- is not an independent variable, so
#     # Var(F+ - F-) = Var(Ntot*g - 2*F-)
#     #              = 4 * Var(F-)
#     #              = 4 * eta * (1-eta) * Ntot * g^2
#     #              = 4 * eta * (1-eta) * flux
#     # We want the variance to be equal to flux, so we need an extra:
#     # delta Var = (1 - 4*eta + 4*eta^2) * flux
#     #           = (1-2eta)^2 * flux
#     absflux = abs(flux)
#     mean = eta_factor * eta_factor * absflux
#     pd = PoissonDeviate(rng, mean)
#     pd_val = pd() - mean + absflux
#     ratio = pd_val / absflux
#     g *= ratio
#     mod_flux *= ratio
#     return jnp.abs(mod_flux), g, rng


# def _scale_extra_noise(self, max_extra_noise, mod_flux, g, max_sb):
#     gfactor = 1.0 + max_extra_noise / jnp.abs(max_sb)
#     mod_flux /= gfactor
#     g *= gfactor
#     return mod_flux, g


# def _calculate_nphotons_nonzero(self, n_photons, poisson_flux, max_extra_noise, rng):
#     # For profiles that are positive definite, then N = flux. Easy.
#     #
#     # However, some profiles shoot some of their photons with negative flux. This means that
#     # we need a few more photons to get the right S/N = sqrt(flux). Take eta to be the
#     # fraction of shot photons that have negative flux.
#     #
#     # S^2 = (N+ - N-)^2 = (N+ + N- - 2N-)^2 = (Ntot - 2N-)^2 = Ntot^2(1 - 2 eta)^2
#     # N^2 = Var(S) = (N+ + N-) = Ntot
#     #
#     # So flux = (S/N)^2 = Ntot (1-2eta)^2
#     # Ntot = flux / (1-2eta)^2
#     #
#     # However, if each photon has a flux of 1, then S = (1-2eta) Ntot = flux / (1-2eta).
#     # So in fact, each photon needs to carry a flux of g = 1-2eta to get the right
#     # total flux.
#     #
#     # That's all the easy case. The trickier case is when we are sky-background dominated.
#     # Then we can usually get away with fewer shot photons than the above.  In particular,
#     # if the noise from the photon shooting is much less than the sky noise, then we can
#     # use fewer shot photons and essentially have each photon have a flux > 1. This is ok
#     # as long as the additional noise due to this approximation is "much less than" the
#     # noise we'll be adding to the image for the sky noise.
#     #
#     # Let's still have Ntot photons, but now each with a flux of g. And let's look at the
#     # noise we get in the brightest pixel that has a nominal total flux of Imax.
#     #
#     # The number of photons hitting this pixel will be Imax/flux * Ntot.
#     # The variance of this number is the same thing (Poisson counting).
#     # So the noise in that pixel is:
#     #
#     # N^2 = Imax/flux * Ntot * g^2
#     #
#     # And the signal in that pixel will be:
#     #
#     # S = Imax/flux * (N+ - N-) * g which has to equal Imax, so
#     # g = flux / Ntot(1-2eta)
#     # N^2 = Imax/Ntot * flux / (1-2eta)^2
#     #
#     # As expected, we see that lowering Ntot will increase the noise in that (and every
#     # other) pixel.
#     # The input max_extra_noise parameter is the maximum value of spurious noise we want
#     # to allow.
#     #
#     # So setting N^2 = Imax + nu, we get
#     #
#     # Ntot = flux / (1-2eta)^2 / (1 + nu/Imax)
#     # g = (1 - 2eta) * (1 + nu/Imax)
#     #
#     # Returns the total flux placed inside the image bounds by photon shooting.
#     #

#     flux = self.flux

#     # The _flux_per_photon property is (1-2eta)
#     # This factor will already be accounted for by the shoot function, so don't include
#     # that as part of our scaling here.  There may be other adjustments though, so g=1 here.
#     eta_factor = self._flux_per_photon
#     mod_flux = flux / (eta_factor * eta_factor)
#     g = 1.0

#     # If requested, let the target flux value vary as a Poisson deviate
#     mod_flux, g, _rng = jax.lax.cond(
#         poisson_flux,
#         lambda poisson_flux, flux, mod_flux, eta_factor, rng, g: self._adjust_flux_g_poisson(
#             poisson_flux, flux, mod_flux, eta_factor, rng, g
#         ),
#         lambda poisson_flux, flux, mod_flux, eta_factor, rng, g: (mod_flux, g, rng),
#         poisson_flux,
#         flux,
#         mod_flux,
#         eta_factor,
#         rng,
#         g,
#     )
#     if rng is not None:
#         rng._state = _rng._state

#     mod_flux, g = jax.lax.cond(
#         max_extra_noise > 0.0,
#         lambda max_extra_noise, mod_flux, g, max_sb: self._scale_extra_noise(
#             max_extra_noise, mod_flux, g, max_sb
#         ),
#         lambda max_extra_noise, mod_flux, g, max_sb: (mod_flux, g),
#         max_extra_noise,
#         mod_flux,
#         g,
#         self.max_sb,
#     )

#     # Make n_photons an integer and use input if requested
#     n_photons = jax.lax.cond(
#         n_photons == 0.0,
#         lambda n_photons, mod_flux: jnp.ceil(mod_flux).astype(int),
#         lambda n_photons, mod_flux: jnp.ceil(n_photons).astype(int),
#         n_photons,
#         mod_flux,
#     )

#     return n_photons, g, rng
