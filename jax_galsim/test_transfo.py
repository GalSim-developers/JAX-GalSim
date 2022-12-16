import jax
from jax.config import config
config.update("jax_enable_x64", True)


import jax.numpy as jnp

import numpy as np
import jax_galsim as galsim

obj1 = galsim.Gaussian(half_light_radius=1.)
trans = galsim.Transformation(obj1)
im = galsim.Image(128,128, dtype=jnp.float64, init_value=0, wcs=galsim.wcs.PixelScale(1.0))
imK = trans._drawKImage(im)
print(imK.array)
