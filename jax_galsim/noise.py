import jax
import jax.numpy as jnp


from jax_galsim.utils import *
from jax_galsim.helpers import _MSG_STACK, apply_stack, seed


def sample(
    name, fn,  rng_key=None, sample_shape=()
):
    """
    """
    assert isinstance(
        sample_shape, tuple
    ), "sample_shape needs to be a tuple of integers"
    if not isinstance(fn, BaseNoise):
        type_error = TypeError(
            "only Normal implemeneted"
        )

    # if no active Messengers, draw a sample or return obs as expected:
    if not _MSG_STACK:
        return fn(rng_key=rng_key, sample_shape=sample_shape)


    # Otherwise, we initialize a message...
    initial_msg = {
        "type": "sample",
        "name": name,
        "fn": fn,
        "value": None,
        "args": (),
        "kwargs": {"rng_key": rng_key, "sample_shape": sample_shape},
    }

    # ...and use apply_stack to send it to the Messengers
    msg = apply_stack(initial_msg)
    return msg["value"]



class BaseNoise():

    def __init__(self, batch_shape=(), event_shape=()):
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        super(BaseNoise, self).__init__()

    @property
    def batch_shape(self):
        """
        Returns the shape over which the distribution parameters are batched.

        :return: batch shape of the distribution.
        :rtype: tuple
        """
        return self._batch_shape

    @property
    def event_shape(self):
        """
        Returns the shape of a single sample from the distribution without
        batching.

        :return: event shape of the distribution.
        :rtype: tuple
        """
        return self._event_shape

    @property
    def event_dim(self):
        """
        :return: Number of dimensions of individual events.
        :rtype: int
        """
        return len(self.event_shape)


    def shape(self, sample_shape=()):
        """
        The tensor shape of samples from this distribution.

        Samples are of shape::

            d.shape(sample_shape) == sample_shape + d.batch_shape + d.event_shape

        :param tuple sample_shape: the size of the iid batch to be drawn from the
            distribution.
        :return: shape of samples.
        :rtype: tuple
        """
        return sample_shape + self.batch_shape + self.event_shape

    def sample(self, key, sample_shape=()):
        """
        Returns a sample from the distribution having shape given by
        `sample_shape + batch_shape + event_shape`. Note that when `sample_shape` is non-empty,
        leading dimensions (of size `sample_shape`) of the returned sample will
        be filled with iid draws from the distribution instance.

        :param jax.random.PRNGKey key: the rng_key key to be used for the distribution.
        :param tuple sample_shape: the sample shape for the distribution.
        :return: an array of shape `sample_shape + batch_shape + event_shape`
        :rtype: numpy.ndarray
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        key = kwargs.pop("rng_key")
        return self.sample(key, *args, **kwargs)



class GaussianNoise(BaseNoise):

    def __init__(self, loc=0.0, scale=1.0):
        self.loc, self.scale = promote_shapes(loc, scale)
        batch_shape = jax.lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        super(GaussianNoise, self).__init__(
            batch_shape = batch_shape
            )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key), 'not valid key'
        eps = jax.random.normal(
            key, shape=sample_shape + self.batch_shape
        )
        return self.loc + eps * self.scale

    @property
    def mean(self):
        return jnp.broadcast_to(self.loc, self.batch_shape)

    @property
    def variance(self):
        return jnp.broadcast_to(self.scale**2, self.batch_shape)
