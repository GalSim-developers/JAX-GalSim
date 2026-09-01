Wavelength-dependent Profiles
=============================

.. currentmodule:: jax_galsim

JAX-GalSim supports a JAX-native subset of GalSim chromatic rendering.  The
core use case is a wavelength-dependent PSF convolved with a source whose SED
is a traced JAX array:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import jax_galsim as galsim

   wave = jnp.linspace(400.0, 900.0, 256)
   sed = galsim.SED(wave, jnp.ones_like(wave))
   bandpass = galsim.Bandpass.tophat(550.0, 750.0)

   gal = galsim.Gaussian(half_light_radius=0.5) * sed
   psf = galsim.ChromaticAtmosphere(fwhm_ref=0.7, lam_ref=700.0, alpha=-0.2)
   final = galsim.ChromaticConvolution([gal, psf])
   image = final.drawImage(bandpass, nx=64, ny=64, scale=0.2, n_waves=32)

Separable chromatic objects, such as ``GSObject * SED``, are rendered by
integrating only the scalar spectral weight, then drawing one unit-flux spatial
profile.  Non-separable chromatic objects, such as a chromatic PSF whose size
changes with wavelength, are rendered by integrating their Fourier-space values
over a fixed wavelength grid.

The wavelength grid size is static, while SED flux values are traced.  This
keeps the rendering path compatible with ``jax.jit`` and ``jax.grad``.

Spectral objects
----------------

.. autoclass:: SED
   :members:
   :show-inheritance:

.. autoclass:: Bandpass
   :members:
   :show-inheritance:

Chromatic objects
-----------------

.. autoclass:: ChromaticObject
   :members:
   :show-inheritance:

.. autoclass:: SimpleChromaticTransformation
   :members:
   :show-inheritance:

.. autoclass:: ChromaticAtmosphere
   :members:
   :show-inheritance:

.. autoclass:: ChromaticConvolution
   :members:
   :show-inheritance:

.. autoclass:: ChromaticSum
   :members:
   :show-inheritance:

Compatibility notes
-------------------

``galsim.Convolve`` dispatches to ``ChromaticConvolution`` when any input is
chromatic.  Multiplication follows GalSim's common pattern:

.. code-block:: python

   chromatic_gal = galsim.Gaussian(half_light_radius=0.5) * sed
   same_object = sed * galsim.Gaussian(half_light_radius=0.5)

The current implementation focuses on differentiable array-backed SEDs,
array-backed bandpasses, separable chromatic sources, and non-separable
Gaussian/Moffat atmospheric PSFs.  Full GalSim spectral I/O, lookup-table
metadata, Airy/OpticalPSF chromatic optics, and photon shooting are still
outside this JAX-native subset.
