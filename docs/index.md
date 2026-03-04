# JAX-GalSim

**JAX port of GalSim, for parallelized, GPU accelerated, and differentiable galaxy image simulations.**

[![Python package](https://github.com/GalSim-developers/JAX-GalSim/actions/workflows/python_package.yaml/badge.svg)](https://github.com/GalSim-developers/JAX-GalSim/actions/workflows/python_package.yaml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/GalSim-developers/JAX-GalSim/main.svg)](https://results.pre-commit.ci/latest/github/GalSim-developers/JAX-GalSim/main)

!!! warning "Early Development"

    This project is still in an early development phase. Please use the
    [reference GalSim implementation](https://github.com/GalSim-developers/GalSim)
    for any scientific applications.

---

## Why JAX-GalSim?

JAX-GalSim reimplements [GalSim](https://github.com/GalSim-developers/GalSim) in pure JAX, unlocking:

!!! tip "JIT Compilation"

    Compile simulation pipelines with `jax.jit` for significant speedups, especially on GPU.

!!! tip "Automatic Differentiation"

    Compute gradients of simulation outputs with respect to galaxy parameters using `jax.grad`.

!!! tip "Vectorization"

    Batch simulations over parameter grids with `jax.vmap` --- no explicit loops needed.

---

## Quick Install

```bash
pip install jax-galsim
```

## Minimal Example

```python
import jax
import jax_galsim

# Define a galaxy and PSF
gal = jax_galsim.Gaussian(flux=1e5, sigma=2.0)
psf = jax_galsim.Gaussian(flux=1.0, sigma=1.0)

# Convolve and draw
final = jax_galsim.Convolve([gal, psf])
image = final.drawImage(scale=0.2)

# Add noise
image = image.addNoise(jax_galsim.GaussianNoise(sigma=30.0))
```

JAX-GalSim objects are JAX pytrees, so you can JIT-compile and differentiate the entire pipeline:

```python
@jax.jit
def simulate(flux, sigma):
    gal = jax_galsim.Gaussian(flux=flux, sigma=sigma)
    psf = jax_galsim.Gaussian(flux=1.0, sigma=1.0)
    return jax_galsim.Convolve([gal, psf]).drawImage(scale=0.2).array.sum()

# Compute gradients with respect to galaxy parameters
grad_fn = jax.grad(simulate, argnums=(0, 1))
dflux, dsigma = grad_fn(1e5, 2.0)
```

---

## Next Steps

- [Installation](getting-started/installation.md) --- Set up JAX-GalSim with GPU support
- [Quick Start](getting-started/quickstart.md) --- Walk through a complete simulation
- [Notable Differences](notable-differences.md) --- What changes when GalSim runs on JAX
- [API Reference](api/profiles/gaussian.md) --- Browse the full API
