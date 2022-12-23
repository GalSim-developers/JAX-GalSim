import jax
from jax.config import config
config.update("jax_enable_x64", True)


import jax.numpy as jnp

import numpy as np
import jax_galsim as galsim

from jax_galsim import helpers
from jax_galsim import noise

####
# Modified version of JaxGalsim.gsobject.drawImage similar to the Gaslim case 
# modified in the previous cell. It does nto include the convolution by Pixel
# and is used with method = 'real_space' or 'fft'

def jaxgalsim_drawImage(
        self,
        image=None,
        nx=None,
        ny=None,
        bounds=None,
        scale=None,
        wcs=None,
        dtype=None,
        method="real_space",
        area=1.0,
        exptime=1.0,
        gain=1.0,
        add_to_image=False,
        center=None,
        use_true_center=True,
        offset=None,
        n_photons=0.0,
        rng=None,
        max_extra_noise=0.0,
        poisson_flux=None,
        sensor=None,
        photon_ops=(),
        n_subsample=3,
        maxN=None,
        save_photons=False,
        bandpass=None,
        setup_only=False,
        surface_ops=None,
    ):
  
        assert method == 'real_space' or method == 'fft'

        from jax_galsim.wcs import PixelScale

        # Figure out what wcs we are going to use.
        wcs = self._determine_wcs(scale, wcs, image)

        # Make sure offset and center are PositionD, converting from other formats (tuple, array,..)
        # Note: If None, offset is converted to PositionD(0,0), but center will remain None.
        offset = self._parse_offset(offset)
        center = self._parse_center(center)

        # Determine the bounds of the new image for use below (if it can be known yet)
        new_bounds = self._get_new_bounds(image, nx, ny, bounds, center)

        # Get the local WCS, accounting for the offset correctly.
        local_wcs = self._local_wcs(
            wcs, image, offset, center, use_true_center, new_bounds
        )

        # Account for area and exptime.
        flux_scale = area * exptime
        # For surface brightness normalization, also scale by the pixel area.
        if method == "sb":
            flux_scale /= local_wcs.pixelArea()
        # Only do the gain here if not photon shooting, since need the number of photons to
        # reflect that actual photons, not ADU.
        if gain != 1 and method != "phot" and sensor is None:
            flux_scale /= gain

        # Determine the offset, and possibly fix the centering for even-sized images
        offset = self._adjust_offset(new_bounds, offset, center, use_true_center)

        # Convert the profile in world coordinates to the profile in image coordinates:
        prof = local_wcs.profileToImage(self, flux_ratio=flux_scale, offset=offset)

        local_wcs = local_wcs.shiftOrigin(offset)

        # Make sure image is setup correctly
        image = prof._setup_image(image, nx, ny, bounds, add_to_image, dtype, center)
        image.wcs = wcs

        if setup_only:
            image.added_flux = 0.0
            return image

        # Making a view of the image lets us change the center without messing up the original.
        wcs = image.wcs
        image.setCenter(0, 0)
        image.wcs = PixelScale(1.0)
        original_center = image.center


        draw_image = image
        add = add_to_image


        if method == 'real_space':
          added_photons = prof.drawReal(draw_image, add_to_image)
        else: # method == 'fft':
          added_photons = prof.drawFFT(draw_image, add_to_image)


        draw_image.added_flux = added_photons / flux_scale
        draw_image.shift(original_center)
        draw_image.wcs = wcs
        return draw_image

galsim.gsobject.GSObject.jaxgalsim_drawImage = jaxgalsim_drawImage
####


gal= galsim.Gaussian(sigma=1)
img_real = gal.jaxgalsim_drawImage(scale=0.1, method='real_space')
img_rfft = gal.jaxgalsim_drawImage(scale=0.1, method="fft") # nb. with scale=1.0 => detect possible aliasing due to non wrapping implementation

print("Gauss: min/max real-rFFT / max real: ",np.max((img_real-img_rfft).array)/np.max(img_real.array), np.min((img_real-img_rfft).array)/np.max(img_real.array))

obj1 = galsim.Gaussian(half_light_radius=1.)
obj2 = galsim.Exponential(half_light_radius=0.5)

# Rescale the flux of one object
obj2 = obj2.withFlux(0.4)

# Sum the two components of my galaxy
gal = obj1 + obj2

im_r = gal.jaxgalsim_drawImage(nx=128, ny=128, scale=0.02, method='real_space')
im_rf = gal.jaxgalsim_drawImage(nx=128, ny=128, scale=0.02, method='fft')
print("Gauss+Exp: min/max real-rFFT / max real: ",np.max((im_r-im_rf).array)/np.max(im_r.array),
      np.min((im_r-im_rf).array)/np.max(im_r.array))
