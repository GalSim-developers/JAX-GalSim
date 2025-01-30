import secrets
from functools import partial

import galsim as _galsim
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.tree_util import register_pytree_node_class

from jax_galsim.core.utils import implements

try:
    from jax.extend.random import wrap_key_data
except (ModuleNotFoundError, ImportError):
    from jax.random import wrap_key_data

from jax_galsim.core.utils import ensure_hashable

LAX_FUNCTIONAL_RNG = """\
JAX-GalSim PRNGs have some support for linking states, but it may not always work and/or may cause issues.

 - Linked states across JIT boundaries or devices will not work.
 - Within a single routine linking may work.
 - You may encounter errors related to global side effects for some combinations of linked states
   and jitted/vmapped routines.

Seeding the JAX-GalSim PRNG can be done in a few ways:

  - pass seed=None (This is equivalent to passing seed=0.)
  - pass an integer seed (This method will throw errors if the integer is traced by JAX.)
  - pass another JAX-GalSim PRNG
  - pass a JAX PRNG key made via `jax.random.key`.

**JAX PRNG keys made via `jax.random.PRNGKey` are not supported.**

When using JAX-GalSim PRNGs and JIT, you should always return the PRNG from the function
and then set the state on input PRNG via `prng.reset(new_prng)`. This will ensure that the
PRNG state is propagated correctly outside the JITed code.
"""


@register_pytree_node_class
class _DeviateState:
    """This class holds the RNG state for a JAX-GalSim PRNG.

    **This class is not intended to be used directly.**

    Parameters
    ----------
    key : key data with dtype `jax.dtypes.prng_key`
        The JAX PRNG key made via `jrandom.key`
    """

    def __init__(self, key):
        self.key = key

    def split_one(self):
        self.key, subkey = jrandom.split(self.key)
        return subkey

    def tree_flatten(self):
        children = (self.key,)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0])


@implements(
    _galsim.BaseDeviate,
    lax_description=LAX_FUNCTIONAL_RNG,
)
@register_pytree_node_class
class BaseDeviate:
    # always the case for JAX
    def __init__(self, seed=None):
        self.reset(seed=seed)
        self._params = {}

    @property
    @implements(_galsim.BaseDeviate.has_reliable_discard)
    def has_reliable_discard(self):
        return True

    @property
    @implements(_galsim.BaseDeviate.generates_in_pairs)
    def generates_in_pairs(self):
        return False

    @implements(
        _galsim.BaseDeviate.seed,
        lax_description="The JAX version of this method does no type checking.",
    )
    def seed(self, seed=None):
        self._seed(seed=seed)

    @implements(_galsim.BaseDeviate._seed)
    def _seed(self, seed=None):
        _initial_seed = seed or secrets.randbelow(2**31)
        self._state.key = jrandom.key(_initial_seed)

    @implements(
        _galsim.BaseDeviate.reset,
        lax_description=("The JAX version of this method does no type checking."),
    )
    def reset(self, seed=None):
        if isinstance(seed, _DeviateState):
            self._state = seed
        elif isinstance(seed, BaseDeviate):
            self._state = seed._state
        elif hasattr(seed, "dtype") and jax.dtypes.issubdtype(
            seed.dtype, jax.dtypes.prng_key
        ):
            self._state = _DeviateState(seed)
        elif isinstance(seed, str):
            self._state = _DeviateState(
                wrap_key_data(jnp.array(eval(seed), dtype=jnp.uint32))
            )
        elif isinstance(seed, tuple):
            self._state = _DeviateState(
                wrap_key_data(jnp.array(seed, dtype=jnp.uint32))
            )
        else:
            _initial_seed = seed or secrets.randbelow(2**31)
            self._state = _DeviateState(jrandom.key(_initial_seed))

    @property
    def _key(self):
        return self._state.key

    @_key.setter
    def _key(self, key):
        self._state.key = key

    @implements(_galsim.BaseDeviate.serialize)
    def serialize(self):
        return repr(ensure_hashable(jrandom.key_data(self._key)))

    @property
    @implements(_galsim.BaseDeviate.np)
    def np(self):
        raise NotImplementedError(
            "The JAX galsim.BaseDeviate does not support being used as a numpy PRNG."
        )

    @implements(_galsim.BaseDeviate.as_numpy_generator)
    def as_numpy_generator(self):
        raise NotImplementedError(
            "The JAX galsim.BaseDeviate does not support being used as a numpy PRNG."
        )

    @implements(
        _galsim.BaseDeviate.clearCache,
        lax_description="This method is a no-op for the JAX version of this class.",
    )
    def clearCache(self):
        pass

    @implements(
        _galsim.BaseDeviate.discard,
        lax_description=(
            "The JAX version of this class has reliable discarding and uses one key per value "
            "so it never generates in pairs. Thus this method will never raise an error."
        ),
    )
    def discard(self, n, suppress_warnings=False):
        self._key = self.__class__._discard(self._key, n)

    @jax.jit
    def _discard(key, n):
        def __discard(i, key):
            key, _ = jrandom.split(key)
            return key

        return jax.lax.fori_loop(0, n, __discard, key, unroll=True)

    @implements(
        _galsim.BaseDeviate.raw,
        lax_description=(
            "The JAX version of this class does not use the raw value to "
            "generate the next value of a specific kind."
        ),
    )
    def raw(self):
        self._key, subkey = jrandom.split(self._key)
        return jrandom.bits(subkey, dtype=jnp.uint32)

    @implements(
        _galsim.BaseDeviate.generate,
        lax_description=(
            "JAX arrays cannot be changed in-place, so the JAX version of "
            "this method returns a new array."
        ),
    )
    def generate(self, array):
        self._key, array = self.__class__._generate(self._key, array)
        return array

    @implements(
        _galsim.BaseDeviate.add_generate,
        lax_description=(
            "JAX arrays cannot be changed in-place, so the JAX version of "
            "this method returns a new array."
        ),
    )
    def add_generate(self, array):
        return self.generate(array) + array

    def __call__(self):
        self._key, val = self.__class__._generate_one(self._key, None)
        return val

    @implements(_galsim.BaseDeviate.duplicate)
    def duplicate(self):
        ret = self.__class__.__new__(self.__class__)
        ret._state = _DeviateState(self._state.key)
        ret._params = self._params.copy()
        return ret

    def __copy__(self):
        return self.duplicate()

    def __eq__(self, other):
        return self is other or (
            isinstance(other, self.__class__)
            and jnp.array_equal(
                jrandom.key_data(self._key), jrandom.key_data(other._key)
            )
            and self._params == other._params
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = None

    def tree_flatten(self):
        """This function flattens the PRNG into a list of children
        nodes that will be traced by JAX and auxiliary static data."""
        # Define the children nodes of the PyTree that need tracing
        children = (self._state, self._params)
        # Define auxiliary static data that doesn’t need to be traced
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        return cls(children[0], **(children[1]))

    def __repr__(self):
        return "galsim.BaseDeviate(seed=%r)" % (
            ensure_hashable(jrandom.key_data(self._key)),
        )

    def __str__(self):
        return self.__repr__()


@implements(
    _galsim.UniformDeviate,
    lax_description=LAX_FUNCTIONAL_RNG,
)
@register_pytree_node_class
class UniformDeviate(BaseDeviate):
    @jax.jit
    def _generate(key, array):
        # we do it this way so that the RNG appears to have a fixed state that is advanced per value drawn
        key, res = jax.lax.scan(
            UniformDeviate._generate_one, key, None, length=array.ravel().shape[0]
        )
        return key, res.reshape(array.shape)

    @jax.jit
    def _generate_one(key, x):
        _key, subkey = jrandom.split(key)
        return _key, jrandom.uniform(subkey, dtype=float)

    def __repr__(self):
        return "galsim.UniformDeviate(seed=%r)" % (
            ensure_hashable(jrandom.key_data(self._key)),
        )

    def __str__(self):
        return "galsim.UniformDeviate()"


@implements(
    _galsim.GaussianDeviate,
    lax_description=LAX_FUNCTIONAL_RNG,
)
@register_pytree_node_class
class GaussianDeviate(BaseDeviate):
    def __init__(self, seed=None, mean=0.0, sigma=1.0):
        super().__init__(seed=seed)
        self._params["mean"] = mean
        self._params["sigma"] = sigma

    @property
    @implements(_galsim.GaussianDeviate.mean)
    def mean(self):
        return self._params["mean"]

    @property
    @implements(_galsim.GaussianDeviate.sigma)
    def sigma(self):
        return self._params["sigma"]

    @implements(
        _galsim.GaussianDeviate.generate,
        lax_description=(
            "JAX arrays cannot be changed in-place, so the JAX version of "
            "this method returns a new array."
        ),
    )
    def generate(self, array):
        self._key, array = self.__class__._generate(self._key, array)
        return array * self.sigma + self.mean

    @jax.jit
    def _generate(key, array):
        # we do it this way so that the RNG appears to have a fixed state that is advanced per value drawn
        key, res = jax.lax.scan(
            GaussianDeviate._generate_one, key, None, length=array.ravel().shape[0]
        )
        return key, res.reshape(array.shape)

    def __call__(self):
        self._key, val = self.__class__._generate_one(self._key, None)
        return val * self.sigma + self.mean

    @jax.jit
    def _generate_one(key, x):
        _key, subkey = jrandom.split(key)
        return _key, jrandom.normal(subkey, dtype=float)

    @implements(_galsim.GaussianDeviate.generate_from_variance)
    def generate_from_variance(self, array):
        self._key, _array = self.__class__._generate(self._key, array)
        return _array * jnp.sqrt(array)

    def __repr__(self):
        return "galsim.GaussianDeviate(seed=%r, mean=%r, sigma=%r)" % (
            ensure_hashable(jrandom.key_data(self._key)),
            ensure_hashable(self.mean),
            ensure_hashable(self.sigma),
        )

    def __str__(self):
        return "galsim.GaussianDeviate(mean=%r, sigma=%r)" % (
            ensure_hashable(self.mean),
            ensure_hashable(self.sigma),
        )


@implements(
    _galsim.BinomialDeviate,
    lax_description=LAX_FUNCTIONAL_RNG,
)
@register_pytree_node_class
class BinomialDeviate(BaseDeviate):
    def __init__(self, seed=None, N=1, p=0.5):
        super().__init__(seed=seed)
        self._params["N"] = N
        self._params["p"] = p

    @property
    @implements(_galsim.BinomialDeviate.n)
    def n(self):
        return self._params["N"]

    @property
    @implements(_galsim.BinomialDeviate.p)
    def p(self):
        return self._params["p"]

    @implements(
        _galsim.BinomialDeviate.generate,
        lax_description=(
            "JAX arrays cannot be changed in-place, so the JAX version of "
            "this method returns a new array."
        ),
    )
    def generate(self, array):
        self._key, array = BinomialDeviate._generate(self._key, array, self.n, self.p)
        return array

    @partial(jax.jit, static_argnums=(2,))
    def _generate(key, array, n, p):
        # we do it this way so that the RNG appears to have a fixed state that is advanced per value drawn
        carry, res = jax.lax.scan(
            BinomialDeviate._generate_one,
            (key, jnp.broadcast_to(p, (n,))),
            None,
            length=array.ravel().shape[0],
        )
        key = carry[0]
        return key, res.reshape(array.shape)

    def __call__(self):
        carry, val = BinomialDeviate._generate_one(
            (self._key, jnp.broadcast_to(self.p, (self.n,))), None
        )
        self._key = carry[0]
        return val

    @jax.jit
    def _generate_one(args, x):
        key, p = args
        _key, subkey = jrandom.split(key)
        # argument order is scale, concentration
        return (_key, p), jnp.sum(jrandom.bernoulli(subkey, p))

    def __repr__(self):
        return "galsim.BinomialDeviate(seed=%r, N=%r, p=%r)" % (
            ensure_hashable(jrandom.key_data(self._key)),
            ensure_hashable(self.n),
            ensure_hashable(self.p),
        )

    def __str__(self):
        return "galsim.BinomialDeviate(N=%r, p=%r)" % (
            ensure_hashable(self.n),
            ensure_hashable(self.p),
        )


@implements(
    _galsim.PoissonDeviate,
    lax_description=LAX_FUNCTIONAL_RNG,
)
@register_pytree_node_class
class PoissonDeviate(BaseDeviate):
    def __init__(self, seed=None, mean=1.0):
        super().__init__(seed=seed)
        self._params["mean"] = mean

    @property
    @implements(_galsim.PoissonDeviate.mean)
    def mean(self):
        return self._params["mean"]

    @implements(
        _galsim.PoissonDeviate.generate,
        lax_description=(
            "JAX arrays cannot be changed in-place, so the JAX version of "
            "this method returns a new array."
        ),
    )
    def generate(self, array):
        self._key, array = self.__class__._generate(self._key, array, self.mean)
        return array

    @jax.jit
    def _generate(key, array, mean):
        # we do it this way so that the RNG appears to have a fixed state that is advanced per value drawn
        key, res = jax.lax.scan(
            PoissonDeviate._generate_one,
            key,
            jnp.broadcast_to(mean, array.ravel().shape),
            length=array.ravel().shape[0],
        )
        return key, res.reshape(array.shape)

    def __call__(self):
        self._key, val = self.__class__._generate_one(self._key, self.mean)
        return val

    @jax.jit
    def _generate_one(key, mean):
        _key, subkey = jrandom.split(key)
        val = jax.lax.cond(
            mean < 2**17,
            lambda subkey, mean: jrandom.poisson(subkey, mean, dtype=int).astype(float),
            lambda subkey, mean: (
                jrandom.normal(subkey, dtype=float) * jnp.sqrt(mean) + mean
            ),
            subkey,
            mean,
        )
        return _key, val

    @implements(_galsim.PoissonDeviate.generate_from_expectation)
    def generate_from_expectation(self, array):
        self._key, _array = self.__class__._generate_from_exp(self._key, array)
        return _array

    @jax.jit
    def _generate_from_exp(key, array):
        # we do it this way so that the RNG appears to have a fixed state that is advanced per value drawn
        key, res = jax.lax.scan(
            PoissonDeviate._generate_one,
            key,
            array.ravel(),
            length=array.ravel().shape[0],
        )
        return key, res.reshape(array.shape)

    def __repr__(self):
        return "galsim.PoissonDeviate(seed=%r, mean=%r)" % (
            ensure_hashable(jrandom.key_data(self._key)),
            ensure_hashable(self.mean),
        )

    def __str__(self):
        return "galsim.PoissonDeviate(mean=%r)" % (ensure_hashable(self.mean),)


@implements(
    _galsim.WeibullDeviate,
    lax_description=LAX_FUNCTIONAL_RNG,
)
@register_pytree_node_class
class WeibullDeviate(BaseDeviate):
    def __init__(self, seed=None, a=1.0, b=1.0):
        super().__init__(seed=seed)
        self._params["a"] = a
        self._params["b"] = b

    @property
    @implements(_galsim.WeibullDeviate.a)
    def a(self):
        return self._params["a"]

    @property
    @implements(_galsim.WeibullDeviate.b)
    def b(self):
        return self._params["b"]

    @implements(
        _galsim.WeibullDeviate.generate,
        lax_description=(
            "JAX arrays cannot be changed in-place, so the JAX version of "
            "this method returns a new array."
        ),
    )
    def generate(self, array):
        self._key, array = self.__class__._generate(self._key, array, self.a, self.b)
        return array

    @jax.jit
    def _generate(key, array, a, b):
        # we do it this way so that the RNG appears to have a fixed state that is advanced per value drawn
        carry, res = jax.lax.scan(
            WeibullDeviate._generate_one,
            (key, a, b),
            None,
            length=array.ravel().shape[0],
        )
        key, _, _ = carry
        return key, res.reshape(array.shape)

    def __call__(self):
        carry, val = self.__class__._generate_one((self._key, self.a, self.b), None)
        self._key, _, _ = carry
        return val

    @jax.jit
    def _generate_one(args, x):
        key, a, b = args
        _key, subkey = jrandom.split(key)
        # argument order is scale, concentration
        return (_key, a, b), jrandom.weibull_min(subkey, b, a, dtype=float)

    def __repr__(self):
        return "galsim.WeibullDeviate(seed=%r, a=%r, b=%r)" % (
            ensure_hashable(jrandom.key_data(self._key)),
            ensure_hashable(self.a),
            ensure_hashable(self.b),
        )

    def __str__(self):
        return "galsim.WeibullDeviate(a=%r, b=%r)" % (
            ensure_hashable(self.a),
            ensure_hashable(self.b),
        )


@implements(
    _galsim.GammaDeviate,
    lax_description=LAX_FUNCTIONAL_RNG,
)
@register_pytree_node_class
class GammaDeviate(BaseDeviate):
    def __init__(self, seed=None, k=1.0, theta=1.0):
        super().__init__(seed=seed)
        self._params["k"] = k
        self._params["theta"] = theta

    @property
    @implements(_galsim.GammaDeviate.k)
    def k(self):
        return self._params["k"]

    @property
    @implements(_galsim.GammaDeviate.theta)
    def theta(self):
        return self._params["theta"]

    @implements(
        _galsim.GammaDeviate.generate,
        lax_description=(
            "JAX arrays cannot be changed in-place, so the JAX version of "
            "this method returns a new array."
        ),
    )
    def generate(self, array):
        self._key, array = self.__class__._generate(self._key, array, self.k)
        return array * self.theta

    @jax.jit
    def _generate(key, array, k):
        # we do it this way so that the RNG appears to have a fixed state that is advanced per value drawn
        key, res = jax.lax.scan(
            GammaDeviate._generate_one,
            key,
            jnp.broadcast_to(k, array.ravel().shape),
            length=array.ravel().shape[0],
        )
        return key, res.reshape(array.shape)

    def __call__(self):
        self._key, val = self.__class__._generate_one(self._key, self.k)
        return val * self.theta

    @jax.jit
    def _generate_one(key, k):
        _key, subkey = jrandom.split(key)
        return _key, jrandom.gamma(subkey, k, dtype=float)

    def __repr__(self):
        return "galsim.GammaDeviate(seed=%r, k=%r, theta=%r)" % (
            ensure_hashable(jrandom.key_data(self._key)),
            ensure_hashable(self.k),
            ensure_hashable(self.theta),
        )

    def __str__(self):
        return "galsim.GammaDeviate(k=%r, theta=%r)" % (
            ensure_hashable(self.k),
            ensure_hashable(self.theta),
        )


@implements(
    _galsim.Chi2Deviate,
    lax_description=LAX_FUNCTIONAL_RNG,
)
@register_pytree_node_class
class Chi2Deviate(BaseDeviate):
    def __init__(self, seed=None, n=1.0):
        super().__init__(seed=seed)
        self._params["n"] = n

    @property
    @implements(_galsim.Chi2Deviate.n)
    def n(self):
        return self._params["n"]

    @implements(
        _galsim.Chi2Deviate.generate,
        lax_description=(
            "JAX arrays cannot be changed in-place, so the JAX version of "
            "this method returns a new array."
        ),
    )
    def generate(self, array):
        self._key, array = self.__class__._generate(self._key, array, self.n)
        return array

    @jax.jit
    def _generate(key, array, n):
        # we do it this way so that the RNG appears to have a fixed state that is advanced per value drawn
        key, res = jax.lax.scan(
            Chi2Deviate._generate_one,
            key,
            jnp.broadcast_to(n, array.ravel().shape),
            length=array.ravel().shape[0],
        )
        return key, res.reshape(array.shape)

    def __call__(self):
        self._key, val = self.__class__._generate_one(self._key, self.n)
        return val

    @jax.jit
    def _generate_one(key, n):
        _key, subkey = jrandom.split(key)
        return _key, jrandom.chisquare(subkey, n, dtype=float)

    def __repr__(self):
        return "galsim.Chi2Deviate(seed=%r, n=%r)" % (
            ensure_hashable(jrandom.key_data(self._key)),
            ensure_hashable(self.n),
        )

    def __str__(self):
        return "galsim.Chi2Deviate(n=%r)" % (ensure_hashable(self.n),)


@implements(
    _galsim.random.permute,
    lax_description="The JAX implementation of this function cannot operate in-place and so returns a new list of arrays.",
)
def permute(rng, *args):
    rng = BaseDeviate(rng)
    arrs = []
    for arr in args:
        arrs.append(jrandom.permutation(rng._key, arr))
    rng.discard(1)
    return arrs
