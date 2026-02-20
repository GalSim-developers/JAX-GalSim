# Notable Differences from GalSim

JAX-GalSim is a drop-in replacement for most GalSim code, but JAX's design
imposes some fundamental differences.

## Immutability

JAX arrays are immutable. Operations that mutate in GalSim return new objects instead:

```python
# GalSim: mutates image in-place
image.addNoise(noise)
image.array[10, 10] = 0.0

# JAX-GalSim: returns a new image
image = image.addNoise(noise)
# Direct array mutation is not supported
```

## Array Views

JAX does not support NumPy's array view semantics. Real-valued views of complex
images (sharing memory) are not available; JAX-GalSim returns copies instead.

## Random Number Generation

JAX uses a **functional PRNG** -- random state is explicit and must be threaded through computations:

- Deterministic and reproducible across platforms
- Deviates return new arrays (cannot "fill" existing ones)
- Number sequences differ from GalSim's RNG, even with the same seed
- Different RNG classes may have different stability properties for discarding

## Profile Restrictions

Some GalSim features are not yet implemented:

- **Truncated Moffat profiles** are not supported (the `trunc` parameter)
- **ChromaticObject** and all chromatic functionality is not available
- **InterpolatedKImage** is not implemented
- See [API Coverage](api-coverage.md) for the full list of supported APIs

## Control Flow and Tracing

JAX's tracing system restricts Python control flow:

- **`if`/`else` on traced values**: Cannot branch on values JAX is tracing (e.g., profile parameters inside `jit`). Use `jax.lax.cond` instead.
- **Variable-size operations**: Operations whose output shape depends on input values (e.g., adaptive image sizing) may not work under `jit`.

JAX-GalSim uses `has_tracers()` internally to detect tracing and avoid problematic control flow.

## Numerical Precision

Results may differ slightly from GalSim due to:

- Different floating-point operation ordering (JAX may reorder for performance)
- XLA-compiled math kernels instead of system math libraries
- Gradient-safe implementations (e.g., `safe_sqrt` in `core/math.py`) that handle edge cases for differentiability
