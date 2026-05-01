Versioning and API Policy
---------------------------

JAX-GalSim follows `Calver <https://calver.org/>`_ with a version number ``YYYY.MM.MICRO`` with ``MICRO`` resetting to ``0`` at the start of each month.

For APIs which are also present in GalSim (e.g., you can import the same thing by substituting galsim for jax_galsim), JAX-GalSim is a strict subset of the GalSim APIs. All other APIs may change without notice for any version part increment. We thus recommend pinning the entire JAX-GalSim version if you use this code in your work.
