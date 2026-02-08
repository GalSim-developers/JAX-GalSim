# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAX-GalSim is a JAX port of [GalSim](https://github.com/GalSim-developers/GalSim) (Galaxy Simulation toolkit) providing parallelized, GPU-accelerated, and differentiable galaxy image simulations. It aims to be a drop-in replacement for GalSim with a pure JAX backend. Currently implements ~22.5% of the GalSim API.

## Build & Development Commands

```bash
# Install in development mode (requires git submodules for tests)
git clone --recurse-submodules <repo-url>
pip install -e ".[dev]"
pre-commit install

# Run all tests
pytest

# Run tests verbose with timing
pytest -vv --durations=100 --randomly-seed=42

# Run a specific test file
pytest tests/jax/test_jitting.py

# Run a specific test function
pytest tests/jax/test_jitting.py::test_gaussian_jit -v

# Lint and format
ruff check . --fix
ruff format .

# Or run all pre-commit hooks
pre-commit run --all-files

# Update API coverage list in README
python scripts/update_api_coverage.py
```

## Architecture

### Core Design Patterns

**1. `@implements` decorator** (`jax_galsim/core/utils.py`): Every JAX-GalSim class/function that reimplements a GalSim equivalent uses this decorator. It copies the GalSim docstring and appends a `lax_description` documenting JAX-specific differences:
```python
@implements(_galsim.Gaussian,
    lax_description="Does not support ChromaticObject.")
class Gaussian(GSObject):
    ...
```

**2. PyTree registration**: All JAX-compatible classes use `@register_pytree_node_class` and implement `tree_flatten()` / `tree_unflatten()`. This enables `jax.jit`, `jax.vmap`, and `jax.grad` transformations. Parameters in `_params` dict are traced; static config in `_gsparams` is auxiliary data.

**3. Immutability**: JAX arrays are immutable—in-place operations return new objects. Image `.view()` and `.subImage()` create copies, not views. RNG methods return new arrays.

### Class Hierarchy

- **GSObject** (`gsobject.py`): Base class for all galaxy profiles. Subclasses implement `_xValue()` (real-space), `_kValue()` (Fourier-space), `_drawReal()`, `_drawKImage()`.
  - Profiles: `Gaussian`, `Exponential`, `Moffat`, `Spergel`, `Box`/`Pixel`, `DeltaFunction`
  - Composites: `Convolution` (FFT-based), `Sum`/`Add`, `Transform`, `Deconvolution`
- **Image** (`image.py`): Wraps JAX arrays with `Bounds` and `WCS` metadata.
- **WCS** (`wcs.py`): Coordinate systems — `PixelScale`, `JacobianWCS`, `AffineTransform`, `ShearWCS`, `OffsetWCS`, `TanWCS`, `GSFitsWCS`.
- **PhotonArray** (`photon_array.py`): Photon positions/fluxes for photon-shooting rendering.
- **Random** (`random.py`): `BaseDeviate` and subclasses (`GaussianDeviate`, `PoissonDeviate`, etc.) using JAX PRNG.

### Drawing Pipeline

`core/draw.py` contains `draw_by_xValue()` (real-space rendering) and `draw_by_kValue()` (Fourier-space rendering). Objects are drawn via `GSObject.drawImage()`.

## Test Structure

Tests live in three directories configured in `pyproject.toml`:

- **`tests/GalSim/`** — Git submodule containing the reference GalSim implementation and its tests. The `conftest.py` monkey-patches `galsim` imports to use `jax_galsim`, so these tests validate API compatibility automatically. This is useful as a reference when implementing new features.
- **`tests/jax/`** — JAX-specific tests (JIT, vmap, grad, benchmarks).
- **`tests/Coord/tests/`** — Coordinate system tests (git submodule).

**`tests/galsim_tests_config.yaml`** controls which GalSim tests are enabled and defines `allowed_failures` for unimplemented features (tests matching these error messages are reported as "allowed failure" instead of errors).

**`tests/conftest.py`** enables `jax_enable_x64` globally and handles the GalSim→jax_galsim module patching.

## Linting

Ruff with rules `E`, `F`, `I`, `W`; ignores `C901`, `E203`, `E501`. `__init__.py` files allow unused imports (`F401`) and unsorted imports (`I001`). Tests in `tests/GalSim/`, `tests/Coord/`, and `dev/notebooks/` are excluded from linting.

## Key Constraints

- JAX ≥0.8.0, GalSim ≥2.7.0, Python 3.12 (CI target)
- JAX x64 mode is always enabled for tests (set in `conftest.py`)
- No compilation step—pure Python package
- New files should have BSD 3-clause license header
- NumPy/SciPy docstring format; prefer `@implements` over copy-pasting GalSim docs
