# Notable Differences from GalSim

JAX-GalSim is designed as a drop-in replacement for GalSim --- replacing
`import galsim` with `import jax_galsim` works for all supported features.
However, JAX's execution model introduces several fundamental differences
that you should understand before porting code or writing new simulations.

---

## Immutability

JAX arrays are **immutable**. Any GalSim operation that modifies data in-place
returns a new object in JAX-GalSim instead.

```python
# GalSim — mutates the image in-place
image.addNoise(noise)
image.array[10, 10] = 0.0

# JAX-GalSim — returns a new image each time
image = image.addNoise(noise)

# Direct array element mutation is not supported.
# Use jax.numpy operations to produce a new array:
new_array = image.array.at[10, 10].set(0.0)
```

This is the most common change when porting GalSim code. Every call that
modifies an image, adds noise, or updates a value must capture the return value.
If you forget the assignment, the original object is unchanged and no error is
raised --- a subtle source of bugs.

---

## Array Views

NumPy supports **array views** --- slices that share memory with the original
array. JAX does not. In GalSim, you can obtain a real-valued view of a complex
image (e.g., the real part shares memory with the underlying complex buffer).
In JAX-GalSim, these operations return **copies** instead. Modifying the copy
does not affect the original.

```python
# GalSim — real_part is a view, shares memory with complex_image
real_part = complex_image.real

# JAX-GalSim — real_part is a copy
real_part = complex_image.real  # independent array
```

---

## Random Number Generation

JAX uses a **functional PRNG** --- random state is explicit and must be passed
through computations. This has several consequences:

**Determinism**: Given the same seed, JAX-GalSim produces identical results
across runs and platforms (CPU, GPU, TPU). GalSim's results may vary by platform.

**Explicit state**: Random deviates carry their state explicitly. Under the hood,
JAX-GalSim wraps JAX's key-based PRNG in GalSim's familiar noise API, so the
user-facing interface looks the same:

```python
noise = jax_galsim.GaussianNoise(sigma=30.0)
image = image.addNoise(noise)  # state is managed internally
```

**Different sequences**: Even with the same seed value, the actual random number
sequences differ from GalSim. Results will not match GalSim number-for-number.
This is expected --- the underlying PRNG algorithms are completely different.

**No in-place fill**: GalSim deviates can "fill" existing arrays. JAX deviates
always return new arrays, consistent with JAX's immutability model.

---

## PyTree Registration

All JAX-GalSim objects are registered as JAX **PyTrees**. This is what allows
you to pass them directly to `jax.jit`, `jax.grad`, and `jax.vmap`.

A PyTree splits each object into two parts:

| Part | What it contains | Examples | Effect of changing |
|------|-----------------|----------|--------------------|
| **Children** (traced) | Values JAX differentiates through | `flux`, `sigma`, `half_light_radius` | Re-evaluation, not recompilation |
| **Auxiliary data** (static) | Structure and configuration | `GSParams`, enum flags | Full recompilation under `jit` |

In practice, profile parameters live in a `_params` dict (children) and
numerical configuration lives in `_gsparams` (auxiliary):

```python
gal = jax_galsim.Gaussian(flux=1e5, sigma=2.0)
# gal._params = {"flux": 1e5, "sigma": 2.0}  — traced by JAX
# gal._gsparams = GSParams(...)               — static, triggers recompile
```

Because `GSParams` is static auxiliary data, changing it between calls to a
`jit`-compiled function triggers recompilation. Keep `GSParams` constant across
calls when possible.

```python
import jax

gsparams = jax_galsim.GSParams(maximum_fft_size=8192)

@jax.jit
def simulate(flux, sigma):
    gal = jax_galsim.Gaussian(flux=flux, sigma=sigma, gsparams=gsparams)
    return gal.drawImage(scale=0.2).array.sum()

# Changing gsparams here would cause recompilation on next call
```

---

## Control Flow and Tracing

JAX's JIT compiler works by **tracing** --- it records operations on abstract
values to build a computation graph. This restricts what Python code can do
inside `jit`-compiled functions.

### No branching on traced values

You cannot use Python `if`/`else` on values that JAX is tracing (e.g., profile
parameters passed into a `jit`-compiled function):

```python
@jax.jit
def bad(sigma):
    if sigma > 1.0:       # ERROR: sigma is a tracer, not a concrete value
        return sigma * 2
    return sigma

@jax.jit
def good(sigma):
    return jax.lax.cond(sigma > 1.0, lambda s: s * 2, lambda s: s, sigma)
```

JAX-GalSim uses an internal `has_tracers()` utility to detect tracing and
avoid problematic control flow in its own implementations.

### Fixed output shapes

Under `jit`, the **shape** of every array must be determinable at compile time.
Operations whose output size depends on input values (e.g., adaptive image
sizing based on a traced parameter) may not work. When using `jax.vmap`, you
must specify fixed image dimensions:

```python
@jax.vmap
def batch(sigma):
    gal = jax_galsim.Gaussian(flux=1e5, sigma=sigma)
    # Must specify nx, ny so all images have the same shape
    return gal.drawImage(scale=0.2, nx=64, ny=64).array
```

### The `__init__` gotcha

During `jit` tracing, JAX calls constructors with **tracer objects** rather than
concrete Python numbers. Type checks like `isinstance(sigma, float)` will fail
on tracers. JAX-GalSim handles this internally, but if you subclass any
JAX-GalSim object, be aware that `__init__` may receive tracers:

```python
from jax_galsim.core.utils import has_tracers

class MyProfile(jax_galsim.GSObject):
    def __init__(self, sigma, gsparams=None):
        if not has_tracers(sigma):
            # Only validate with concrete values
            if sigma <= 0:
                raise ValueError("sigma must be positive")
        ...
```

---

## Profile Restrictions

Some GalSim features are not yet implemented in JAX-GalSim:

- **Truncated Moffat profiles**: The `trunc` parameter is not supported.
- **ChromaticObject**: All chromatic functionality (wavelength-dependent profiles) is not available.
- **InterpolatedKImage**: Not implemented.
- **Airy, Kolmogorov, OpticalPSF, RealGalaxy**: And other profiles --- see [API Coverage](api-coverage.md) for the full list.

The project currently implements **22.5%** of the GalSim public API, focused on
the most commonly used profiles and operations. Coverage is expanding.

---

## Numerical Precision

Simulation results may differ slightly from GalSim at the floating-point level:

- **Operation reordering**: JAX (via XLA) may reorder floating-point operations for performance. Floating-point addition is not associative, so different orderings produce slightly different results.
- **Different math kernels**: XLA-compiled math kernels may differ from system math libraries (e.g., `libm`) that GalSim uses via NumPy/C++.
- **Gradient-safe functions**: JAX-GalSim uses special implementations (e.g., `safe_sqrt` that avoids `NaN` gradients at zero) where GalSim uses standard library functions. These may produce slightly different values at edge cases.
- **Default precision**: JAX defaults to 32-bit floats. Enable 64-bit with `jax.config.update("jax_enable_x64", True)` for higher precision matching GalSim's default behavior.

These differences are typically at the level of floating-point round-off
($\sim 10^{-7}$ for float32, $\sim 10^{-15}$ for float64) and should not
affect scientific conclusions.

---

## The `@implements` Decorator

JAX-GalSim reuses GalSim's docstrings rather than duplicating them. Every public
class and function uses an `@implements` decorator that copies the docstring from
the corresponding GalSim object and appends a note about JAX-specific differences:

```python
from jax_galsim.core.utils import implements
import galsim as _galsim

@implements(_galsim.Gaussian,
    lax_description="LAX: Does not support ChromaticObject.")
class Gaussian(GSObject):
    ...
```

This means the [API Reference](api/profiles/gaussian.md) shows GalSim's
documentation with an added "LAX-backend" note. If you see RST-formatted cross-references
like `:func:` or `:class:` in the docs, they come from GalSim's original docstrings.
