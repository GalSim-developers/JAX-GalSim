# PyTree Registration

JAX transformations decompose objects into arrays (leaves) and static structure
(treedef). Every JAX-GalSim class uses `@register_pytree_node_class` and
implements two methods:

```python
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class Gaussian(GSObject):
    def tree_flatten(self):
        # Children: values that JAX can trace (differentiate through)
        children = (self.params,)
        # Aux data: static values that trigger recompilation if changed
        aux_data = (self.gsparams,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Reconstruct the object from its parts
        return cls(params=children[0], gsparams=aux_data[0])
```

## Children vs Auxiliary Data

| Component | Role | Examples | Effect of changing |
|-----------|------|----------|--------------------|
| **Children** (traced) | Values JAX differentiates through | `flux`, `sigma`, `half_light_radius` | Triggers re-evaluation, not recompilation |
| **Auxiliary data** (static) | Structure and configuration | `GSParams`, enum flags | Triggers full recompilation under `jit` |

In practice, profile parameters live in a `_params` dict (children) and numerical
configuration lives in `_gsparams` (auxiliary):

```python
gal = jax_galsim.Gaussian(flux=1e5, sigma=2.0)
# gal._params = {"flux": 1e5, "sigma": 2.0}  — traced
# gal._gsparams = GSParams(...)               — static
```

## The `__init__` Gotcha

During `tree_unflatten`, JAX calls the constructor with traced values, not
concrete Python numbers. Type checks like `isinstance(sigma, float)` will fail
on tracers. Use `has_tracers()` to skip validation during tracing:

```python
from jax_galsim.core.utils import has_tracers

class MyProfile(GSObject):
    def __init__(self, sigma, gsparams=None):
        if not has_tracers(sigma):
            # Only validate with concrete values
            if sigma <= 0:
                raise ValueError("sigma must be positive")
        ...
```

## Practical Implications

1. **`jax.grad` works**: Because profile parameters are traced children, you get
   gradients for free.

2. **`GSParams` changes recompile**: Changing `GSParams` between calls to a
   `jit`-compiled function triggers recompilation, since it's static auxiliary data.

3. **No mutable state**: PyTree flattening and unflattening means all state must
   be reconstructable from children + aux_data. There's no hidden mutable state.
