# Quick Start

A complete galaxy image simulation, then JAX transformations (`jit`, `grad`, `vmap`) on top.

## A Simple Simulation

A Gaussian galaxy convolved with a Gaussian PSF, drawn and noised -- equivalent to GalSim's `demo1.py`.

```python
import jax_galsim

# Galaxy parameters
gal_flux = 1e5      # total counts
gal_sigma = 2.0     # arcsec
psf_sigma = 1.0     # arcsec
pixel_scale = 0.2   # arcsec/pixel
noise_sigma = 30.0  # counts per pixel

# Define profiles
gal = jax_galsim.Gaussian(flux=gal_flux, sigma=gal_sigma)
psf = jax_galsim.Gaussian(flux=1.0, sigma=psf_sigma)

# Convolve galaxy with PSF
final = jax_galsim.Convolve([gal, psf])

# Draw the image
image = final.drawImage(scale=pixel_scale)

# Add Gaussian noise
image = image.addNoise(jax_galsim.GaussianNoise(sigma=noise_sigma))

# Write to FITS
image.write("output/demo1.fits")
```

Most GalSim code translates directly by replacing `import galsim` with `import jax_galsim`.

## JIT Compilation

Wrap your simulation in `jax.jit` to compile it into an optimized XLA computation:

```python
import jax

@jax.jit
def simulate(flux, sigma):
    gal = jax_galsim.Gaussian(flux=flux, sigma=sigma)
    psf = jax_galsim.Gaussian(flux=1.0, sigma=1.0)
    final = jax_galsim.Convolve([gal, psf])
    return final.drawImage(scale=0.2)

# First call compiles; subsequent calls are fast
image = simulate(1e5, 2.0)
```

## Automatic Differentiation

Compute gradients of any scalar output with respect to parameters:

```python
def total_flux(gal_sigma, psf_sigma):
    gal = jax_galsim.Gaussian(flux=1e5, sigma=gal_sigma)
    psf = jax_galsim.Gaussian(flux=1.0, sigma=psf_sigma)
    final = jax_galsim.Convolve([gal, psf])
    image = final.drawImage(scale=0.2)
    return image.array.sum()

# Gradient of total image flux with respect to both sigmas
grad_fn = jax.grad(total_flux, argnums=(0, 1))
d_gal, d_psf = grad_fn(2.0, 1.0)
```

Useful for fitting galaxy models to data via gradient-based optimization.

## Vectorization with vmap

Batch-simulate galaxies with different parameters without explicit loops:

```python
import jax.numpy as jnp

sigmas = jnp.linspace(1.0, 4.0, 10)

@jax.vmap
def batch_simulate(sigma):
    gal = jax_galsim.Gaussian(flux=1e5, sigma=sigma)
    psf = jax_galsim.Gaussian(flux=1.0, sigma=1.0)
    final = jax_galsim.Convolve([gal, psf])
    return final.drawImage(scale=0.2, nx=64, ny=64).array

# Simulate all 10 galaxies in parallel
images = batch_simulate(sigmas)  # shape: (10, 64, 64)
```

## Next Steps

- [Key Concepts](key-concepts.md) — Understand the design behind JAX-GalSim
- [Notable Differences](../notable-differences.md) — What's different from GalSim
- [API Reference](../api/profiles/gaussian.md) — Full API documentation
