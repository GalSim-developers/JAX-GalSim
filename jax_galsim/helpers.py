import numpy as np
import jax
import jax.numpy as jnp



_MSG_STACK = []

def default_process_message(msg):
   msg["value"] = msg["fn"](*msg["args"], **msg["kwargs"])

def apply_stack(msg):
    """
    Execute the effect stack at a single site according to the following scheme:
        1. For each ``Messenger`` in the stack from bottom to top,
           execute ``Messenger.process_message`` with the message;
           if the message field "stop" is True, stop;
           otherwise, continue
        2. Apply default behavior (``default_process_message``) to finish remaining
           site execution
        3. For each ``Messenger`` in the stack from top to bottom,
           execute ``Messenger.postprocess_message`` to update the message
           and internal messenger state with the site results
    """
    pointer = 0
    for pointer, handler in enumerate(reversed(_MSG_STACK)):
        handler.process_message(msg)
        # When a Messenger sets the "stop" field of a message,
        # it prevents any Messengers above it on the stack from being applied.
        if msg.get("stop"):
            break

    default_process_message(msg)

    # A Messenger that sets msg["stop"] == True also prevents application
    # of postprocess_message by Messengers above it on the stack
    # via the pointer variable from the process_message loop
    for handler in _MSG_STACK[-pointer - 1 :]:
        handler.postprocess_message(msg)
    return msg

class Messenger(object):
    def __init__(self, fn=None):
        if fn is not None and not callable(fn):
            raise ValueError(
                "Expected `fn` to be a Python callable object; "
                "instead found type(fn) = {}.".format(type(fn))
            )
        self.fn = fn

    def __enter__(self):
        _MSG_STACK.append(self)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            assert _MSG_STACK[-1] is self
            _MSG_STACK.pop()
        else:
            if self in _MSG_STACK:
                loc = _MSG_STACK.index(self)
                for i in range(loc, len(_MSG_STACK)):
                    _MSG_STACK.pop()

    def process_message(self, msg):
        pass

    def postprocess_message(self, msg):
        pass

    def __call__(self, *args, **kwargs):
        if self.fn is None:
            # Assume self is being used as a decorator.
            assert len(args) == 1 and not kwargs
            self.fn = args[0]
            return self
        with self:
            return self.fn(*args, **kwargs)


class seed(Messenger):
    """
    Helper to jax glue random keys feature
    """

    def __init__(self, fn=None, rng_seed=None):
        if isinstance(rng_seed, int) or (
            isinstance(rng_seed, (np.ndarray, jnp.ndarray)) and not jnp.shape(rng_seed)
        ):
            rng_seed = jax.random.PRNGKey(rng_seed)
        if not (
            isinstance(rng_seed, (np.ndarray, jnp.ndarray))
            and rng_seed.dtype == jnp.uint32
            and rng_seed.shape == (2,)
        ):
            raise TypeError("Incorrect type for rng_seed: {}".format(type(rng_seed)))
        self.rng_key = rng_seed
        super(seed, self).__init__(fn)

    def process_message(self, msg):
        if (
            msg["type"] == "sample" and msg["kwargs"]["rng_key"] is None
        ):
            if msg["value"] is not None:
                # no need to create a new key when value is available
                return
            self.rng_key, rng_key_sample = jax.random.split(self.rng_key)
            msg["kwargs"]["rng_key"] = rng_key_sample



