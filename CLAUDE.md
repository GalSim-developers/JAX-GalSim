# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAX-GalSim is a JAX port of GalSim (Galaxy Image Simulation toolkit) that enables parallelized, GPU-accelerated, and differentiable galaxy image simulations. This is an early-stage project aiming to reimplement GalSim functionalities in pure JAX.

**Key Design Principles:**
- Drop-in replacement for GalSim with a close API match
- Each function/feature is tested against the reference GalSim implementation
- This is a **subset** of GalSim (only includes functions with a reference implementation)
- Code should be readable and pip-installable without compilation

**Current Status:** Early development phase (v0.0.1rc1). Not for scientific applications yet - use the reference GalSim implementation for production work.

## Installation and Setup

**Recommended:** Use a virtual environment to isolate dependencies:

```bash
# Clone with submodules (required for tests)
git clone --recurse-submodules https://github.com/YOUR_USERNAME/JAX-GalSim
cd JAX-GalSim

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .

# Install development tools
pip install black pre-commit pytest
pre-commit install
```

## Testing

```bash
# Run all tests (includes both GalSim reference tests and JAX-specific tests)
pytest

# Test paths are configured in pytest.ini:
# - tests/GalSim/tests/ (reference GalSim tests)
# - tests/jax (JAX-specific tests like test_jitting.py)

# Run specific test file
pytest tests/jax/test_jitting.py
```

## Code Formatting

This project uses Black for code formatting:

```bash
# Format all code
black .

# Black excludes tests/GalSim/ directory (configured in .pre-commit-config.yaml)
```

**Important:** CI will fail if code is not formatted with Black. Use pre-commit hooks to automate this.

## Architecture

### Core Structure

- `jax_galsim/` - Main package implementing JAX versions of GalSim objects
  - `gsobject.py` - Base `GSObject` class that all galaxy profile objects inherit from
  - `gsparams.py` - `GSParams` class for speed/accuracy trade-off parameters
  - `gaussian.py`, `exponential.py` - Specific galaxy profile implementations
  - `sum.py` - Composite objects (e.g., `Add`, `Sum`)
  - `core/` - Core utilities (currently empty/minimal)

- `tests/` - Test suite
  - `GalSim/` - Git submodule containing the reference GalSim implementation for testing
  - `jax/` - JAX-specific tests (e.g., JIT compilation tests)

### JAX Pytree Registration

All GSObject classes must be registered as JAX pytrees to support JIT compilation and automatic differentiation:

```python
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class MyGSObject(GSObject):
    def tree_flatten(self):
        # Return (children, aux_data) where children are JAX arrays
        # and aux_data contains static information
        ...

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Reconstruct object from aux_data and children
        ...
```

### Documentation Pattern

Avoid duplicating documentation. Use JAX's `_wraps` utility to inherit docs from GalSim:

```python
from jax._src.numpy.util import _wraps
import galsim as _galsim

@_wraps(_galsim.Gaussian)
@register_pytree_node_class
class Gaussian(GSObject):
    ...

# Or for functions with differences:
@_wraps(_galsim.Add, lax_description="Does not support `ChromaticObject` at this point.")
def Add(*args, **kwargs):
    return Sum(*args, **kwargs)
```

The `lax_description` parameter documents any differences or limitations compared to GalSim.

### GSObject Parameter Management

GSObjects use a dual-parameter system:
- `_params` dict: Traced parameters (JAX arrays) that can be differentiated
- `_gsparams`: Static parameters (`GSParams` object) for numerical configurations

Properties like `flux`, `sigma`, `half_light_radius` etc. are accessed via `self.params` dictionary.

### Testing Against Reference GalSim

Tests in `tests/GalSim/tests/` are from the reference GalSim implementation. JAX-GalSim objects are tested against these to ensure API compatibility and numerical accuracy.

JAX-specific tests in `tests/jax/` verify JAX functionality like JIT compilation, differentiation, and pytree behavior.

### How the Testing Infrastructure Works

JAX-GalSim uses a **pytest hook system** to automatically run GalSim's test suite against JAX-GalSim implementations. This means you can reuse all of GalSim's existing tests without modification!

#### The Mechanism

**1. Import Replacement (`tests/conftest.py`)**
   - The `pytest_pycollect_makemodule` hook intercepts test file loading
   - Automatically replaces `import galsim` with `import jax_galsim` in all GalSim test files
   - This happens transparently - no modification to GalSim test files needed!

**2. Test Configuration (`tests/galsim_tests_config.yaml`)**
   ```yaml
   enabled_tests:
     galsim:
       - test_gaussian.py
       - test_exponential.py
       - "*"  # Enable all GalSim tests

   allowed_failures:
     - "module 'jax_galsim' has no attribute 'Airy'"
     - "module 'jax_galsim.bessel' has no attribute 'j1'"
     # ... list of expected failures for unimplemented features
   ```

   - `enabled_tests`: Lists which GalSim test files to run (`"*"` means all)
   - `allowed_failures`: Error messages that won't fail the test suite (for tracking unimplemented features)

**3. Test Execution Flow**
   ```
   pytest tests/GalSim/tests/test_bessel.py
           ↓
   pytest hook replaces: import galsim → import jax_galsim
           ↓
   GalSim tests run against JAX-GalSim implementation
           ↓
   Results compared with scipy.special / reference values
           ↓
   PASS / FAIL / ALLOWED FAILURE
   ```

#### Enabling Tests for New Functions

When you implement a new function in JAX-GalSim, follow these steps to enable its tests:

**Example: Adding `bessel.kn` function**

1. **Implement the function** in `jax_galsim/bessel.py`:
   ```python
   from jax_galsim.core.utils import implements
   import galsim as _galsim

   @implements(_galsim.bessel.kn)
   @jax.jit
   def kn(n, x):
       """Modified Bessel function K_n for integer n"""
       # ... implementation ...
       return result
   ```

2. **Remove from allowed_failures** in `tests/galsim_tests_config.yaml`:
   ```yaml
   allowed_failures:
     # DELETE or comment out this line:
     # - "module 'jax_galsim.bessel' has no attribute 'kn'"
   ```

3. **Run the tests**:
   ```bash
   pytest tests/GalSim/tests/test_bessel.py::test_kn -v
   ```

4. **Test outcomes**:
   - **PASS**: Your implementation matches GalSim's accuracy ✅
   - **FAIL**: Numerical accuracy issues - fix your implementation
   - **ERROR**: API mismatch - check function signature and behavior

#### Finding Which Tests Will Run

To see what GalSim tests exist for a module:

```bash
# List all bessel tests
grep "^def test_" tests/GalSim/tests/test_bessel.py

# Example output:
# def test_j0():
# def test_j1():
# def test_kn():
# def test_kv():
# ... etc
```

Each `test_*` function will automatically run against your JAX-GalSim implementation when enabled!

#### Tracking Progress

```bash
# Run all tests and see summary
pytest tests/GalSim/tests/ -v

# Common output:
# ✅ 25 passed        - Implementations working correctly
# ❌ 3 failed         - Implementations with accuracy issues
# ⚠️  100 allowed     - Features not yet implemented
```

This gives you clear visibility into:
- What's working (passing tests)
- What needs fixing (failing tests)
- What's not implemented yet (allowed failures)

#### Debugging Failed Tests

When a test fails, pytest shows:
- **Expected values**: From GalSim/scipy
- **Actual values**: From your JAX-GalSim implementation
- **Tolerance**: Typically `rtol=1e-10` (10 decimal places)

Example failure:
```python
AssertionError:
Not equal to tolerance rtol=1e-10
ACTUAL:  [18.24, 2.146, 45.04, ...]
DESIRED: [11.90, 2.146, 37.79, ...]
```

This tells you exactly which test cases have accuracy problems.

## Contributing Workflow

1. Fork and clone with `--recurse-submodules`
2. Create a feature branch: `git checkout -b descriptive-name`
3. Make changes and ensure tests pass: `pytest`
4. Format code: `black .`
5. Update `CHANGELOG.md`
6. Add BSD license header to new files
7. Squash commits if needed: `git rebase -i`
8. Open PR against `main` branch

**Before submitting:**
- Ensure tests pass
- Code is Black-formatted
- PR is self-contained and focused
- New functionality has tests
- Branch is up-to-date with upstream `main`

## Submodule Management

The `tests/GalSim` directory is a git submodule pointing to the reference GalSim implementation. When tests fail to run:

```bash
# Initialize/update submodules
git submodule update --init --recursive
```

## Documentation Style

Follow NumPy/SciPy documentation format: https://numpydoc.readthedocs.io/en/latest/format.html

Prefer using `_wraps` to inherit GalSim documentation rather than copy/pasting.
