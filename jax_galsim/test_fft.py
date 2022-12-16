import jax
from jax.config import config
config.update("jax_enable_x64", True)


import jax.numpy as jnp

import numpy as np
import jax_galsim as galsim

from jax_galsim import helpers
from jax_galsim import noise


# Define a galsim galaxy as the sum of two objects
obj1 = galsim.Gaussian(half_light_radius=1.)
obj2 = galsim.Exponential(half_light_radius=0.5)

# Rescale the flux of one object
obj2 = obj2.withFlux(0.4)

# Sum the two components of my galaxy
gal = obj1 + obj2

#FFT image 
imFFT= galsim.Image(128,128, dtype=jnp.float64, init_value=0, wcs=galsim.wcs.PixelScale(1.0))

kimg, N = gal.drawFFT_makeKImage(image=imFFT)
print("CD: ",kimg, N, kimg.dtype)

imFFT= galsim.Image(128,128, dtype=jnp.float32, init_value=0, wcs=galsim.wcs.PixelScale(1.0))
print("imFFT.scale= ", imFFT.scale, "type: ",imFFT.dtype)
kimg, N = gal.drawFFT_makeKImage(image=imFFT)
print("CF: ",kimg, N, kimg.dtype)



print("_drawKImage (1): ",kimg,kimg.array)
kimg = gal._drawKImage(kimg)
print("_drawKImage (2): ",kimg,kimg.array)
