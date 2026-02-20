# The @implements Decorator

JAX-GalSim reuses GalSim's docstrings rather than duplicating them. The
`@implements` decorator handles this automatically.

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

Use this to document any differences from the reference GalSim implementation:

```python
@implements(_galsim.Add,
    lax_description="Does not support ChromaticObject at this point.")
def Add(*args, **kwargs):
    return Sum(*args, **kwargs)
```

The description is inserted at the top of the docstring, after the summary line,
as a "LAX-backend implementation" note.

## How It Works

The decorator (defined in `jax_galsim/core/utils.py`) does the following:

1. Retrieves the original function's docstring
2. Parses the GalSim-style numpydoc format (summary, parameters, etc.)
3. Reconstructs the docstring with:
   - The original summary line
   - A "LAX-backend implementation of `:func:original.name`" note
   - The `lax_description` text (if provided)
   - The rest of the original docstring body
4. Assigns the combined docstring to the wrapped function

## When to Use It

- **Always** when implementing a GalSim class or function — this is the standard
  pattern in JAX-GalSim
- The `lax_description` should note any restricted functionality, different
  behavior, or missing parameters compared to GalSim
- If there is no corresponding GalSim function (e.g., JAX-specific utilities),
  write a normal docstring instead
