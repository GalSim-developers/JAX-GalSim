# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is JAX-GalSim?

JAX-GalSim is a pure-JAX reimplementation of [GalSim](https://github.com/GalSim-developers/GalSim), the modular galaxy image simulation toolkit. It provides a near-identical API to GalSim but with full support for JAX transformations (`jit`, `vmap`, `grad`), enabling differentiable and GPU-accelerated galaxy simulations. Currently ~22.5% of the GalSim API is implemented.

## Build and Development Commands

```bash
# Setup (requires --recurse-submodules for test submodules)
git clone --recurse-submodules <repo-url>
pip install -e ".[dev]"
pre-commit install

# Run all tests
pytest

# Run a single test file
pytest tests/jax/test_jitting.py

# Run a specific test
pytest tests/jax/test_jitting.py::test_name -v

# Lint and format
ruff check . --fix
ruff format .

# Or via pre-commit
pre-commit run --all-files
```

## Architecture

### GSObject System

The core abstraction is `GSObject` (in `gsobject.py`), the base class for all galaxy/PSF profiles. Key design:

- **Parameters**: Traced (differentiable) values stored in `self._params` dict. Static config in `self._gsparams`.
- **Pytree protocol**: Every GSObject subclass is decorated with `@register_pytree_node_class` and implements `tree_flatten()`/`tree_unflatten()` so JAX can trace through them.
- **Drawing**: Profiles implement `_xValue(pos)` for real-space and `_kValue(kpos)` for Fourier-space evaluation. Drawing is dispatched through `core/draw.py` which uses `jax.vmap` over pixel centers.
- **Composition**: `Convolution`, `Sum`, `Transform` compose GSObjects into new ones while preserving the pytree structure.

Concrete profiles: `Gaussian`, `Exponential`, `Moffat`, `Spergel`, `Box`/`Pixel`, `DeltaFunction`, `InterpolatedImage`.

### Key Patterns

**Implementing a new GSObject**: Every profile follows this pattern:
```python
import galsim as _galsim
from jax.tree_util import register_pytree_node_class
from jax_galsim.core.utils import implements

@implements(_galsim.ProfileName)
@register_pytree_node_class
class ProfileName(GSObject):
    def __init__(self, ..., flux=1.0, gsparams=None):
        super().__init__(param1=val1, flux=flux, gsparams=gsparams)

    def tree_flatten(self):
        children = (self.params,)
        aux_data = {"gsparams": self.gsparams}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        ...
```

**`@implements` decorator** (in `core/utils.py`): Copies docstrings from the reference GalSim object and optionally appends a `lax_description` noting JAX-specific differences. Use this on all public classes/functions that mirror GalSim.

**`from_galsim()`/`to_galsim()` methods**: Many classes provide conversion to/from GalSim equivalents (Position, Bounds, Image, Shear, GSParams, WCS, CelestialCoord).

### Image System

`Image` (in `image.py`) wraps `jnp.ndarray` with bounds and WCS metadata. Unlike GalSim, JAX arrays are immutable — all operations return new Image instances instead of modifying in place.

### Testing Architecture

Tests come from three sources, configured in `tests/galsim_tests_config.yaml`:
- **`tests/GalSim/tests/`** — GalSim's own test suite (git submodule), run against jax_galsim via `conftest.py` module-swapping hooks
- **`tests/Coord/tests/`** — Coord package test suite (git submodule)
- **`tests/jax/`** — JAX-specific tests (jit, vmap, grad, benchmarks)

The `conftest.py` replaces `galsim` imports with `jax_galsim` in collected test modules. Tests for unimplemented features auto-pass via the `allowed_failures` list in the YAML config.

CI splits tests into 4 parallel groups using `pytest-split`.

## Key Differences from GalSim

- JAX arrays are **immutable** — no in-place operations on images
- `jax.config.update("jax_enable_x64", True)` is required for 64-bit precision (set in conftest.py)
- Random number generation uses JAX's functional RNG, not GalSim's C++ RNG
- Reference GalSim is always imported as `_galsim` (private) to avoid confusion with the jax_galsim namespace

## Ruff Configuration

Selects rules: E, F, I, W. Ignores: C901 (complexity), E203, E501 (line length). `__init__.py` ignores F401 (unused imports) and I001 (import order). Ruff excludes `tests/GalSim/`, `tests/Coord/`, `tests/jax/galsim/`, `dev/notebooks/`.
