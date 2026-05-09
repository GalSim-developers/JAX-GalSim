JAX-GalSim
==========

.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   quickstart
   sharp-bits
   api-coverage
   versioning
   api/index

|ci-badge| |ruff-badge| |precommit-badge|

.. |ci-badge| image:: https://github.com/GalSim-developers/JAX-GalSim/actions/workflows/python_package.yaml/badge.svg
   :target: https://github.com/GalSim-developers/JAX-GalSim/actions/workflows/python_package.yaml

.. |ruff-badge| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff

.. |precommit-badge| image:: https://results.pre-commit.ci/badge/github/GalSim-developers/JAX-GalSim/main.svg
   :target: https://results.pre-commit.ci/latest/github/GalSim-developers/JAX-GalSim/main

.. warning::

   This project is still in an early development phase. Please use the
   `reference GalSim implementation <https://github.com/GalSim-developers/GalSim>`_
   for any scientific applications.

**JAX-GalSim** is a JAX re-implementation of the `GalSim
<https://github.com/GalSim-developers/GalSim>`_ galaxy image simulation
toolkit.  It exposes (nearly) the same API as GalSim while enabling
automatic differentiation, JIT compilation, and hardware acceleration via
`JAX <https://github.com/google/jax>`_.

Why JAX-GalSim?
---------------

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: ⚡ JIT Compilation
      :class-card: sd-border-0

      Compile simulation pipelines with ``jax.jit`` for significant
      speedups, especially on GPU.

   .. grid-item-card:: 🔁 Automatic Differentiation
      :class-card: sd-border-0

      Compute gradients of simulation outputs with respect to galaxy
      parameters using ``jax.grad``.

   .. grid-item-card:: 🔀 Vectorization
      :class-card: sd-border-0

      Batch simulations over parameter grids with ``jax.vmap`` — no
      explicit loops needed.

Quick Install
-------------

.. code-block:: bash

   pip install jax-galsim

.. code-block:: bash

   conda install -c conda-forge jax-galsim

See :doc:`installation` for GPU support and development setup.

Minimal Example
---------------

.. code-block:: python

   import jax
   import jax_galsim

   # Define a galaxy and PSF
   gal = jax_galsim.Gaussian(flux=1e5, sigma=2.0)
   psf = jax_galsim.Gaussian(flux=1.0, sigma=1.0)

   # Convolve and draw
   final = jax_galsim.Convolve([gal, psf])
   image = final.drawImage(scale=0.2)

JAX-GalSim objects are JAX pytrees, so you can JIT-compile and differentiate
the entire pipeline:

.. code-block:: python

   @jax.jit(static_argnames=['slen', 'fft_size'])
   def simulate(flux, sigma, *, slen=21, fft_size=128):
       gsparams = jax_galsim.GSParams(minimum_fft_size=fft_size, maximum_fft_size=fft_size)
       gal = jax_galsim.Gaussian(flux=flux, sigma=sigma)
       psf = jax_galsim.Gaussian(flux=1.0, sigma=1.0)
       return jax_galsim.Convolve([gal, psf]).withGSParams(gsparams) \
                         .drawImage(nx=slen, ny=slen, scale=0.2).array.sum()

   # Compute gradients with respect to galaxy parameters
   dflux, dsigma = jax.grad(simulate, argnums=(0, 1))(1e5, 2.0)


Getting Started
---------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 📖 API Reference
      :link: api/index
      :link-type: doc

      Auto-generated documentation for every public class, function, and
      module in ``jax_galsim``.

   .. grid-item-card:: 🔗 GalSim upstream
      :link: https://galsim-developers.github.io/GalSim/_build/html
      :link-type: url

      The original GalSim documentation.  Many docstrings in JAX-GalSim
      are derived from GalSim and expanded with JAX-specific notes.

   .. grid-item-card:: 🚀 Quick Start
      :link: quickstart
      :link-type: doc

      Walk through a complete simulation with JIT, grad, and vmap.

   .. grid-item-card:: 🔪 JAX-GalSim - The Sharp Bits 🔪
      :link: sharp-bits
      :link-type: doc

      What changes when GalSim runs on JAX — immutability, tracing,
      PyTrees, and more.

About the Documentation
------------------------

Each class and function that mirrors an upstream GalSim object is annotated
with :func:`jax_galsim.core.utils.implements`.  This decorator copies the
original GalSim docstring and prepends any JAX-specific caveats.  In the :doc:`api/index` you will therefore find:

* A **summary** and optional **🔪 JAX-GalSim - The Sharp Bits 🔪** block at the top of
  each entry highlighting important caveats.
* An explicit **Parameters** table derived from the original GalSim
  documentation.
* A collapsible **Original GalSim Documentation** block containing the full
  upstream narrative.
