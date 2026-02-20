# Key Concepts

This page covers the core ideas you need to understand when using JAX-GalSim.

## GSObject: The Building Block

Every galaxy profile, PSF model, and optical component is a **GSObject** — the
base class for all surface brightness profiles. GSObjects support:

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

Unlike GalSim (which uses mutable NumPy arrays), JAX-GalSim images are
**immutable** because JAX arrays cannot be modified in-place. Operations that
would mutate an image in GalSim return a new image instead:

```python
# GalSim (mutable):
#   image.addNoise(noise)  # modifies image in-place
#
# JAX-GalSim (immutable):
image = image.addNoise(noise)  # returns a new image
```

This is the most common source of differences when porting GalSim code.

## PyTree Registration

JAX transformations (`jit`, `grad`, `vmap`) need to understand how to
decompose objects into arrays and metadata. JAX-GalSim objects are registered
as **PyTrees** with two components:

- **Children** (traced): Parameters that can be differentiated — stored in
  `obj._params` (e.g., flux, sigma, half_light_radius)
- **Auxiliary data** (static): Configuration that doesn't change — stored in
  `obj._gsparams` and similar fields

This means you can pass JAX-GalSim objects directly to `jit`, `grad`, and `vmap`
without any special handling:

```python
import jax

@jax.jit
def render(gal):
    return gal.drawImage(scale=0.2).array

gal = jax_galsim.Gaussian(flux=1e5, sigma=2.0)
result = render(gal)  # gal is automatically flattened/unflattened
```

See [PyTree Registration](../architecture/pytree.md) for implementation details.

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

JAX uses a functional PRNG — random state is explicit and never mutated:

```python
import jax

# Create a random key
key = jax.random.PRNGKey(42)

# Create a noise model (JAX-GalSim wraps this in GalSim-compatible API)
noise = jax_galsim.GaussianNoise(sigma=30.0)
image = image.addNoise(noise)
```

See [Notable Differences](../notable-differences.md) for more on how the RNG
differs from GalSim.
