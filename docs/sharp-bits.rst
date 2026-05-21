🔪 JAX-GalSim - The Sharp Bits 🔪
==================================

JAX-GalSim is designed as a drop-in replacement for GalSim — replacing
``import galsim`` with ``import jax_galsim`` works for all supported features.
However, JAX's execution model introduces several fundamental differences
that you should understand before porting code or writing new simulations.

Immutability
------------

JAX arrays are **immutable**. Any GalSim operation that originally modified data
in-place now creates a new array that overwrites the original one. Take
``__iadd__`` as an example:

.. code-block:: python

   # GalSim — mutates the image in-place
   # i.e. no new numpy array is created
   image += 1.0
   # under the hood: self.array[:,:] += a  (no new array)

   # JAX-GalSim — creates a new array and overwrites original one
   image += 1.0
   # under the hood: image._array = image._array + 1.0  (new JAX array)

This can be a subtle source of bugs if you are used to NumPy in-place
mutability. Here is a concrete illustration:

.. code-block:: python

   # galsim
   image = galsim.ImageD(11, 11)
   arr1 = image.array
   image += 1.0
   arr1.sum(), image.array.sum()   # -> 121.0, 121.0

   # jax-galsim
   image = jax_galsim.ImageD(11, 11)
   arr1 = image.array
   image += 1.0
   arr1.sum(), image.array.sum()   # -> 0.0, 121.0  (original unmodified!)

For more details, see the `JAX Sharp Bits page on in-place updates
<https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#in-place-updates>`_.

Array Views
-----------

NumPy supports **array views** — slices that share memory with the original
array. JAX does not. In GalSim, you can obtain a real-valued view of a complex
image (the real part shares memory with the underlying complex buffer).
In JAX-GalSim these operations return **copies** instead. Modifying the copy
does not affect the original.

.. code-block:: python

   # GalSim — real_part is a view, shares memory with complex_image
   real_part = complex_image.real

   # JAX-GalSim — real_part is a copy
   real_part = complex_image.real  # independent array

Fixed Array Shapes in JAX Function Transformations
--------------------------------------------------

JAX function transformations (e.g., ``jax.jit``, ``jax.vmap``, etc.) require statically known
array shapes in order to support tracing. To support this, the JAX-GalSim ``BoundsI`` class must
have a statically known shape. Further this class can be instantiated via the syntax
``BoundsI(xmin=..., deltax=..., ymin=..., deltay=...)`` where ``deltax/y`` are the statically defined
shape. ``BoundsI`` classes may have dynamically set ``x/ymin`` values. However, in this case the ``&``
and ``+`` operations, which can change the shape of the ``BoundsI`` instance are not allowed in
JAX-traced code. ``BoundsI`` instances have a special method ``isStatic()`` which returns ``True``
if the object was instantiated with statically know ``x/ymin`` values. A static ``BoundsI`` class
cannot be converted to a dynamic one via assignment and an attempt to do so will raise an exception.

Scalar Types, Array Types, and Type Casting
-------------------------------------------

With the use of JAX, there are now many possible types for numeric data. These include

- **Python scalars**: Objects with types that are ``float``, ``int``, or ``complex``.
- **NumPy scalars**: Objects with types that are subclasses of ``np.floating``, ``np.integer``, etc.
- **NumPy array scalars**: Objects with a type that is ``np.ndarray`` and have ``np.ndim(...) == 0``.
- **NumPy arrays**: Objects with a type that is ``np.ndarray`` and have ``np.ndim(...) > 0``.
- **JAX array scalars**: Objects with a type that is ``jax.numpy.ndarray`` and have ``jax.numpy.ndim(...) == 0``.
- **JAX arrays**: Objects with a type that is ``jax.numpy.ndarray`` and have ``jax.numpy.ndim(...) > 0``.

**JAX does not have pure scalar types like NumPy. JAX uses array scalars for those instead.**

JAX-GalSim uses the following rules when handling data types and casting.

- If the item is a Python numeric type (i.e., ``int`` or ``float``) or a
  NumPy scalar type (i.e., ``isinstance(x, np.number)``, ``isinstance(x, np.integer)``, etc.),
  convert it to a Python type of the appropriate kind.
- For all other array-like types, cast to the correct type via ``jax.numpy.astype(x, ...)``.
- For putting data into FITS headers only, JAX-GalSim converts of NumPy/JAX arrays to Python
  numeric types as long as there is one element in the array (i.e., it is a NumPy scalar type,
  an array scalar, or a 1D array with one element).

These rules allow JAX-GalSim to transparently handle JAX's tracing operations, but can result in
the code raising generic ``Exception`` instances instead of more specific ``GalSim`` exceptions in
some cases.

Object Comparison with ``==``
-----------------------------

In JAX-GalSim, all objects which define arrays to be traced by JAX will return JAX boolean
array scalars (i.e., ``jax.numpy.array(True)`` or ``jax.numpy.array(False)``) as the result
of the ``==`` operator. Important cases of this rule are static ``BoundsI`` objects and
``Interpolant`` objects (and their subclasses), which return Python boolean values (i.e.
``True`` and ``False``). These difference can be a source of subtle bugs since the negation
of JAX array boolean values is typically done with ``~``, while for Python boolean values it is
done with ``not``. Mixing these two forms can cause unexpected and incorrect results since

.. code-block:: python

   >>> ~True is False
   <python-input-0>:1: SyntaxWarning: "is" with 'int' literal. Did you mean "=="?
   False

Random Number Generation
------------------------

JAX uses a **functional PRNG** — random state is explicit and must be passed
through computations. This has several consequences:

**Determinism**: Given the same seed, JAX-GalSim produces identical results
across runs and platforms (CPU, GPU, TPU). GalSim's results may vary by
platform.

**Explicit state**: Random deviates carry their state explicitly. Under the
hood, JAX-GalSim wraps JAX's key-based PRNG in GalSim's familiar noise API,
so the user-facing interface looks the same:

.. code-block:: python

   noise = jax_galsim.GaussianNoise(sigma=30.0)
   image.addNoise(noise)  # state is managed internally

**Different sequences**: Even with the same seed value, the actual random
number sequences differ from GalSim. Results will not match GalSim
number-for-number. This is expected — the underlying PRNG algorithms are
completely different.

**No in-place fill**: GalSim deviates can "fill" existing arrays. JAX deviates
always return new arrays, consistent with JAX's immutability model.

PyTree Registration
-------------------

All JAX-GalSim objects are registered as JAX **PyTrees**. This is what allows
you to pass them directly to ``jax.jit``, ``jax.grad``, and ``jax.vmap``.

A PyTree splits each object into two parts:

.. list-table::
   :header-rows: 1
   :widths: 20 35 30 15

   * - Part
     - What it contains
     - Examples
     - Effect of changing
   * - **Children** (traced)
     - Values JAX differentiates through
     - ``flux``, ``sigma``, ``half_light_radius``
     - Re-evaluation, not recompilation
   * - **Auxiliary data** (static)
     - Structure and configuration
     - ``GSParams``, enum flags
     - Full recompilation under ``jit``

For ``GSObject``, profile parameters live in a ``_params`` dict (children) and
numerical configuration lives in ``_gsparams`` (auxiliary):

.. code-block:: python

   gal = jax_galsim.Gaussian(flux=1e5, sigma=2.0)
   # gal._params    = {"flux": 1e5, "sigma": 2.0}  — traced by JAX
   # gal._gsparams  = GSParams(...)                 — static, triggers recompile

Because ``GSParams`` is static auxiliary data, changing it between calls to a
``jit``-compiled function triggers recompilation. Keep ``GSParams`` constant
across calls when possible:

.. code-block:: python

   import jax

   gsparams = jax_galsim.GSParams(minimum_fft_size=8192, maximum_fft_size=8192)
   slen = 21

   @jax.jit
   def simulate(flux, sigma):
       gal = jax_galsim.Gaussian(flux=flux, sigma=sigma, gsparams=gsparams)
       return gal.drawImage(nx=slen, ny=slen, scale=0.2).array.sum()

Control Flow and Tracing
------------------------

JAX's JIT compiler works by **tracing** — it records operations on abstract
values to build a computation graph. This restricts what Python code can do
inside ``jit``-compiled functions.

No branching on traced values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You cannot use Python ``if``/``else`` on values that JAX is tracing (e.g.,
profile parameters passed into a ``jit``-compiled function):

.. code-block:: python

   @jax.jit
   def bad(sigma):
       if sigma > 1.0:       # ERROR: sigma is a tracer, not a concrete value
           return sigma * 2
       return sigma

   @jax.jit
   def good(sigma):
       return jax.lax.cond(sigma > 1.0, lambda s: s * 2, lambda s: s, sigma)

Fixed output shapes
^^^^^^^^^^^^^^^^^^^

Under ``jit``, the **shape** of every array must be determinable at compile time.
When using ``jax.jit`` or ``jax.vmap`` you must specify fixed image dimensions:

.. code-block:: python

   @jax.jit
   @jax.vmap
   def batch(sigma):
       gsparams = jax_galsim.GSParams(minimum_fft_size=256, maximum_fft_size=256)
       gal = jax_galsim.Gaussian(flux=1e5, sigma=sigma).withGSParams(gsparams)
       # Must specify nx, ny so all images have the same shape
       return gal.drawImage(scale=0.2, nx=64, ny=64).array

The default drawing procedure uses an FFT whose k-space image size normally
depends on traced galaxy parameters (e.g. size). Fix it explicitly via
``GSParams``:

.. code-block:: python

   gsparams = jax_galsim.GSParams(minimum_fft_size=256, maximum_fft_size=256)

Both ``minimum_fft_size`` and ``maximum_fft_size`` must be set to the same value.

The ``__init__`` gotcha
^^^^^^^^^^^^^^^^^^^^^^^

During ``jit`` tracing, JAX calls constructors with **tracer objects** rather
than concrete Python numbers. Type checks like ``isinstance(sigma, float)`` will
return ``False`` on tracers, and you cannot check correctness of values (e.g.,
``if sigma > 0: ...```). JAX-GalSim handles this internally, but if you subclass any
JAX-GalSim object, be aware that ``__init__`` may receive tracers.

Profile Restrictions
--------------------

Some GalSim features are not yet implemented in JAX-GalSim:

- **Truncated Moffat profiles**: The ``trunc`` parameter is not supported.
- **ChromaticObject**: All chromatic functionality (wavelength-dependent
  profiles) is not available.
- **InterpolatedKImage**: Not implemented.
- **Airy, Kolmogorov, OpticalPSF, RealGalaxy, etc.**: See :doc:`api-coverage` for
  the full list.

Numerical Precision
-------------------

Simulation results may differ slightly from GalSim at the floating-point level:

- **Operation reordering**: JAX (via XLA) may reorder floating-point operations
  for performance. Floating-point addition is not associative, so different
  orderings produce slightly different results.
- **Different math kernels**: XLA-compiled math kernels may differ from system
  math libraries (e.g. ``libm``) used by GalSim via NumPy/C++.
- **Gradient-safe functions**: JAX-GalSim uses special implementations (e.g.
  ``safe_sqrt`` to avoid ``NaN`` gradients at zero) where GalSim uses standard
  library functions. These may produce slightly different values at edge cases.
- **Default precision**: JAX defaults to 32-bit floats. Enable 64-bit with
  ``jax.config.update("jax_enable_x64", True)`` for higher precision matching
  GalSim's default behaviour.

These differences are typically at the level of floating-point round-off
(:math:`{\sim}10^{-7}` for float32, :math:`{\sim}10^{-15}` for float64) and
should not affect scientific conclusions.

⚠️ Additional Sharp Bits
------------------------

In the :doc:`api/index` you will find **🔪 JAX-GalSim - The Sharp Bits 🔪** blocks highlighting additional important caveats for specific classes and or methods. These could include things like:

- Some classes do not perform some of Galsim's test for correctness during initialization (e.g., :meth:`~jax_galsim.InterpolatedImage`).
- Certain profiles might not be auto-differentiable with respect to some of their parameters (e.g., :class:`~jax_galsim.Spergel`, :class:`~jax_galsim.Moffat`)
- Limitations regarding what types of inputes are handled (e.g., :meth:`~jax_galsim.Image.calculate_fft` does not accept complex dtypes.)

