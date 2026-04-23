JAX-GalSim
==========

.. toctree::
   :maxdepth: 1
   :hidden:

   api/index

**JAX-GalSim** is a JAX re-implementation of the `GalSim
<https://github.com/GalSim-developers/GalSim>`_ galaxy image simulation
toolkit.  It exposes (nearly) the same API as GalSim while enabling
automatic differentiation, JIT compilation, and hardware acceleration via
`JAX <https://github.com/google/jax>`_.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc

      Auto-generated documentation for every public class, function, and
      module in ``jax_galsim``.

   .. grid-item-card:: GalSim upstream
      :link: https://galsim-developers.github.io/GalSim/_build/html
      :link-type: url

      The original GalSim documentation.  Many docstrings in JAX-GalSim
      are derived from GalSim and expanded with JAX-specific notes.

About JAX-GalSim
----------------

Each class and function that mirrors an upstream GalSim object is annotated
with :func:`jax_galsim.core.utils.implements`.  This decorator copies the
original GalSim docstring and prepends any JAX-specific caveats.  In the API
docs you will therefore find:

* A **summary** and optional **JAX-specific notes** at the top of each entry.
* An explicit **Parameters** table derived from the original GalSim
  documentation.
* A collapsible **Original GalSim Documentation** block containing the full
  upstream narrative.
