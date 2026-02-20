# The @implements Decorator

JAX-GalSim reuses GalSim's docstrings via the `@implements` decorator.

## Usage

```python
import galsim as _galsim
from jax_galsim.core.utils import implements

@implements(_galsim.Gaussian,
    lax_description="LAX: Does not support ChromaticObject.")
class Gaussian(GSObject):
    ...
```

This does three things:

1. **Copies the docstring** from `_galsim.Gaussian` to `jax_galsim.Gaussian`
2. **Adds a LAX note** (the `lax_description`) documenting JAX-specific
   differences or limitations
3. **Sets `__galsim_wrapped__`** on the class, linking back to the original
   GalSim implementation

## The `lax_description` Parameter

Documents differences from the reference GalSim implementation:

```python
@implements(_galsim.Add,
    lax_description="Does not support ChromaticObject at this point.")
def Add(*args, **kwargs):
    return Sum(*args, **kwargs)
```

## How It Works

The decorator (in `jax_galsim/core/utils.py`):

1. Copies the original GalSim docstring
2. Inserts a "LAX-backend implementation of `:func:original.name`" note after the summary line
3. Appends the `lax_description` text (if provided)
4. Assigns the combined docstring to the wrapped class/function

## When to Use It

- **Always** when implementing a GalSim class or function — this is the standard
  pattern in JAX-GalSim
- The `lax_description` should note any restricted functionality, different
  behavior, or missing parameters compared to GalSim
- If there is no corresponding GalSim function (e.g., JAX-specific utilities),
  write a normal docstring instead
