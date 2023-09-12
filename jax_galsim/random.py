import jax
import jax.numpy as jnp
import numpy as np

from jax._src.numpy.util import _wraps
import galsim as _galsim


class BaseDeviate:
    @_wraps(_galsim.BaseDeviate)
    def __init__(self, seed=None):
        self._rng_args = ()
        self.reset(seed)

    def seed(self, seed=0):
        """Seed the pseudo-random number generator with a given integer value.

        Parameters:
            seed:       An int value to be used to seed the random number generator.  Using 0
                        means to generate a seed from the system. [default: 0]
        """
        if seed == int(seed):
            self._seed(int(seed))
        else:
            raise TypeError("BaseDeviate seed must be an integer.  Got %s" % seed)

    def _seed(self, seed=0):
        """Equivalent to `seed`, but without any type checking."""
        self._rng = jax.random.PRNGKey(seed)

    def reset(self, seed=None):
        """Reset the pseudo-random number generator, severing connections to any other deviates.
        Providing another `BaseDeviate` object as the seed connects this deviate with the other
        one, so they will both use the same underlying random number generator.

        Parameters:
            seed:       Something that can seed a `BaseDeviate`: an integer seed or another
                        `BaseDeviate`.  Using None means to generate a seed from the system.
                        [default: None]
        """
        if isinstance(seed, BaseDeviate):
            self._reset(seed)
        elif seed is None:
            self._rng = jax.random.PRNGKey(0)
        elif seed == int(seed):
            self._rng = jax.random.PRNGKey(seed)
        else:
            raise TypeError(
                "BaseDeviate must be initialized with either a jax or another "
                "BaseDeviate"
            )

    def _reset(self, rng):
        """Equivalent to `reset`, but rng must be a `BaseDeviate` (not an int), and there
        is no type checking.
        """
        # TODO: Check whether we want to allow this pattern, because it
        # will create two identical objects that will be incrementing their
        # seeds in parallel, with the same values
        self._rng = rng._rng

    def duplicate(self):
        """Create a duplicate of the current `BaseDeviate` object.

        The subsequent series from each copy of the `BaseDeviate` will produce identical values::

            >>> u = galsim.UniformDeviate(31415926)
            >>> u()
            0.17100770119577646
            >>> u2 = u.duplicate()
            >>> u()
            0.49095047544687986
            >>> u()
            0.10306670609861612
            >>> u2()
            0.49095047544687986
            >>> u2()
            0.10306670609861612
            >>> u2()
            0.13129289541393518
            >>> u()
            0.13129289541393518
        """
        ret = BaseDeviate.__new__(self.__class__)
        ret.__dict__.update(self.__dict__)
        ret._rng = self._rng
        return ret

    def __copy__(self):
        return self.duplicate()

    def __getstate__(self):
        d = self.__dict__.copy()
        d["rng_str"] = self.serialize()
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._rng = d["_rng"]

    def clearCache(self):
        """Clear the internal cache of the `BaseDeviate`, if any.  This is currently only relevant
        for `GaussianDeviate`, since it generates two values at a time, saving the second one to
        use for the next output value.
        """
        pass  # Does nothing for the JAX backend

    def discard(self, n, suppress_warnings=False):
        """Discard n values from the current sequence of pseudo-random numbers.

        This is typically used to keep two random number generators in sync when one of them
        is used to generate some random values.  The other can quickly discard the same number
        of random values to stay in sync with the first one.

        Parameters:
            n:                  The number of raw random numbers to discard.
            suppress_warnings:  Whether to suppress warnings related to detecting when this
                                action is not likely to reliably keep two random number
                                generators in sync. [default: False]
        """
        if not self.has_reliable_discard and not suppress_warnings:
            _galsim.galsim_warn(
                self.__class__.__name__
                + " does not use a consistent number of randoms per generated value, "
                + "so discard cannot be guaranteed to keep two random deviates in sync."
            )
        if n % 2 == 1 and self.generates_in_pairs and not suppress_warnings:
            _galsim.galsim_warn(
                self.__class__.__name__
                + " uses two randoms per pair of generated values, so discarding "
                + "an odd number of randoms probably doesn't make sense."
            )
        self._rng = jax.random.split(self._rng, int(n) + 1)[-1]

    @property
    def has_reliable_discard(self):
        """Indicates whether the generator always uses 1 rng per value.

        If it does, then `discard` can reliably be used to keep two generators in sync when one
        of them generated some values and the other didn't.

        This is False for PoissonDeviate, Chi2Deviate, and GammaDeviate.

        See also `BaseDeviate.generates_in_pairs`.
        """
        return True

    @property
    def generates_in_pairs(self):
        """Indicates whether the generator uses 2 rngs values per 2 returned values.

        GaussianDeviate has a slight wrinkle to the `BaseDeviate.has_reliable_discard` story.
        It always uses two rng values to generate two Gaussian deviates.  So if the number of
        generated values is even, then discard can keep things in sync.  However, if an odd
        number of values are generated, then you to generate one more value (and discard it)
        to keep things synced up.

        This is only True for GaussianDeviate.
        """
        return False

    def generate(self, array):
        """Generate many pseudo-random values, filling in the values of a numpy array."""
        raise NotImplementedError

    def add_generate(self, array):
        """Generate many pseudo-random values, adding them to the values of a numpy array."""
        return array + self.generate(array)

    def __eq__(self, other):
        return self is other or (
            isinstance(other, self.__class__)
            and self._rng_args == other._rng_args
            and self.serialize() == other.serialize()
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = None

    def serialize(self):
        return str(self._rng)

    def _seed_repr(self):
        s = self.serialize().split(" ")
        return " ".join(s)

    def __repr__(self):
        return "galsim.BaseDeviate(%r)" % self._seed_repr()

    def __str__(self):
        return "galsim.BaseDeviate(%r)" % self._seed_repr()


class GaussianDeviate(BaseDeviate):
    """Pseudo-random number generator with Gaussian distribution.

    See http://en.wikipedia.org/wiki/Gaussian_distribution for further details.

    Successive calls to ``g()`` generate pseudo-random values distributed according to a Gaussian
    distribution with the provided ``mean``, ``sigma``::

        >>> g = galsim.GaussianDeviate(31415926)
        >>> g()
        0.5533754000847082
        >>> g()
        1.0218588970190354

    Parameters:
        seed:       Something that can seed a `BaseDeviate`: an integer seed or another
                    `BaseDeviate`.  Using 0 means to generate a seed from the system.
                    [default: None]
        mean:       Mean of Gaussian distribution. [default: 0.]
        sigma:      Sigma of Gaussian distribution. [default: 1.; Must be > 0]
    """

    def __init__(self, seed=None, mean=0.0, sigma=1.0):
        if sigma < 0.0:
            raise _galsim.GalSimRangeError(
                "GaussianDeviate sigma must be > 0.", sigma, 0.0
            )
        self._rng_args = (float(mean), float(sigma))
        self.reset(seed)

    @property
    def mean(self):
        """The mean of the Gaussian distribution."""
        return self._rng_args[0]

    @property
    def sigma(self):
        """The sigma of the Gaussian distribution."""
        return self._rng_args[1]

    def __call__(self):
        """Draw a new random number from the distribution.

        Returns a Gaussian deviate with the given mean and sigma.
        """
        seed, self._rng = jax.random.split(self._rng)
        return self.mean + self.sigma * jax.random.normal(seed)

    def generate(self, array):
        seed, self._rng = jax.random.split(self._rng)
        return self.mean + self.sigma * jax.random.normal(seed, shape=array.shape)

    def generate_from_variance(self, array):
        """Generate many Gaussian deviate values using the existing array values as the
        variance for each.
        """
        seed, self._rng = jax.random.split(self._rng)
        return self.mean + jnp.sqrt(array) * jax.random.normal(seed, shape=array.shape)

    def __repr__(self):
        return "galsim.GaussianDeviate(seed=%r, mean=%r, sigma=%r)" % (
            self._seed_repr(),
            self.mean,
            self.sigma,
        )

    def __str__(self):
        return "galsim.GaussianDeviate(mean=%r, sigma=%r)" % (self.mean, self.sigma)
