# Installation

## Quick Install

```bash
pip install jax-galsim
```

This installs JAX-GalSim and its dependencies (JAX, NumPy, GalSim, Astropy).

## GPU Support

JAX-GalSim inherits GPU support from JAX. To use NVIDIA GPUs, install the appropriate JAX variant:

```bash
pip install -U "jax[cuda12]"
```

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for other accelerators and platform-specific instructions.

## Development Install

To contribute to JAX-GalSim or run the test suite:

```bash
# Clone with submodules (required for GalSim reference tests)
git clone --recurse-submodules https://github.com/GalSim-developers/JAX-GalSim
cd JAX-GalSim

# Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/jax/test_api.py

# Run a specific test
pytest tests/jax/test_api.py::test_api_same

# Verbose output with timing
pytest -vv --durations=100
```

### Linting and Formatting

```bash
# Lint
ruff check . --fix

# Format
ruff format .

# Or run both via pre-commit
pre-commit run --all-files
```

See [CONTRIBUTING.md](https://github.com/GalSim-developers/JAX-GalSim/blob/main/CONTRIBUTING.md) for full contribution guidelines.
