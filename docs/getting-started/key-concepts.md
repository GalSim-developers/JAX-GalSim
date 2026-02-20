# Key Concepts

## GSObject: The Building Block

Every galaxy profile, PSF, and optical component is a **GSObject**. GSObjects support:

- **Arithmetic**: `+` (sum), `*` (scalar multiply)
- **Transformations**: `.shear()`, `.shift()`, `.dilate()`, `.rotate()`
- **Convolution**: `jax_galsim.Convolve([obj1, obj2, ...])`
- **Drawing**: `.drawImage(scale=...)` renders the profile to a pixel grid

```python
import jax_galsim

# A Gaussian galaxy, sheared and shifted
gal = jax_galsim.Gaussian(flux=1e5, sigma=2.0)
gal = gal.shear(g1=0.2, g2=0.1)
gal = gal.shift(dx=0.5, dy=-0.3)
```

## Images Are Immutable

JAX arrays cannot be modified in-place, so operations that would mutate an image
in GalSim return a new image instead:

```python
# GalSim:   image.addNoise(noise)          # modifies in-place
# JAX-GalSim: image = image.addNoise(noise)  # returns new image
```

This is the most common difference when porting GalSim code.
See [Notable Differences](../notable-differences.md) for the full list.

## PyTree Registration

All JAX-GalSim objects are registered as JAX **PyTrees**, so you can pass them
directly to `jit`, `grad`, and `vmap`:

```python
import jax

@jax.jit
def render(gal):
    return gal.drawImage(scale=0.2).array

gal = jax_galsim.Gaussian(flux=1e5, sigma=2.0)
result = render(gal)  # gal is automatically flattened/unflattened
```

Profile parameters (flux, sigma, etc.) are **traced children** that JAX can
differentiate through. Configuration like `GSParams` is **static auxiliary data**
that triggers recompilation when changed. See [PyTree Registration](../architecture/pytree.md)
for details.

## GSParams: Numerical Configuration

`GSParams` controls numerical accuracy and performance trade-offs for rendering:

```python
# Use tighter tolerances for high-precision work
gsparams = jax_galsim.GSParams(maximum_fft_size=8192, kvalue_accuracy=1e-6)
gal = jax_galsim.Gaussian(flux=1e5, sigma=2.0, gsparams=gsparams)
```

`GSParams` is treated as **static** auxiliary data in the PyTree, so changing it
triggers recompilation under `jit`.

## Functional Random Numbers

JAX uses a functional PRNG -- random state is explicit and never mutated.
JAX-GalSim wraps this in GalSim's familiar noise API:

```python
noise = jax_galsim.GaussianNoise(sigma=30.0)
image = image.addNoise(noise)
```

See [Notable Differences](../notable-differences.md) for details on RNG behavior.
