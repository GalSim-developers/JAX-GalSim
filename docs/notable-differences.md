# Notable Differences from GalSim

JAX-GalSim strives to be a drop-in replacement for GalSim, but JAX's design
imposes some fundamental differences. This page documents them.

## Immutability

JAX arrays are immutable — you cannot modify them in-place. This affects all
image operations:

```python
# GalSim: mutates image in-place
image.addNoise(noise)
image.array[10, 10] = 0.0

# JAX-GalSim: returns a new image
image = image.addNoise(noise)
# Direct array mutation is not supported
```

Any GalSim code that relies on in-place modification of images needs to be
rewritten to use the return value.

## Array Views

JAX does not support all the array view semantics that NumPy provides. In
particular:

- **Real views of complex images** are not available. In GalSim, you can get a
  real-valued view of a complex image's real or imaginary part that shares
  memory. JAX-GalSim returns copies instead.

## Random Number Generation

JAX uses a **functional PRNG** — random state is explicit and must be threaded
through computations. Key differences:

- JAX's PRNG is deterministic and reproducible across platforms
- Random deviates cannot "fill" an existing array; they return new arrays
- The sequence of random numbers differs from GalSim's RNG, so results will
  not be numerically identical even with the same seed
- Different RNG classes may have different stability properties for discarding

## Profile Restrictions

Some GalSim features are not yet implemented:

- **Truncated Moffat profiles** are not supported (the `trunc` parameter)
- **ChromaticObject** and all chromatic functionality is not available
- **InterpolatedKImage** is not implemented
- See [API Coverage](api-coverage.md) for the full list of supported APIs

## Control Flow and Tracing

JAX's tracing system places restrictions on Python control flow:

- **`if`/`else` on traced values**: You cannot branch on values that JAX is
  tracing (e.g., profile parameters inside a `jit`-compiled function). Use
  `jax.lax.cond` instead.
- **Variable-size operations**: Operations whose output shape depends on input
  values (e.g., adaptive image sizing) may not work under `jit`.

JAX-GalSim uses `has_tracers()` internally to detect when code is being traced
and avoid problematic control flow patterns.

## Numerical Precision

Some operations may produce slightly different numerical results due to:

- Different order of floating-point operations (JAX may reorder for performance)
- Use of XLA-compiled math kernels instead of system math libraries
- Custom gradient-safe implementations (e.g., `safe_sqrt` in `core/math.py`)
  that handle edge cases differently for differentiability
