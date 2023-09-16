import jax.numpy as jnp


def tree_data_is_equal(ad1, ad2):
    """Utility function to compare two flattened pytrees
    even if they have JAX arrays.
    """
    if isinstance(ad1, tuple):
        if not isinstance(ad2, tuple):
            return False
        if len(ad1) != len(ad2):
            return False
        for i in range(len(ad1)):
            if not tree_data_is_equal(ad1[i], ad2[i]):
                return False
    elif isinstance(ad1, list):
        if not isinstance(ad2, list):
            return False
        if len(ad1) != len(ad2):
            return False
        for i in range(len(ad1)):
            if not tree_data_is_equal(ad1[i], ad2[i]):
                return False
    elif isinstance(ad1, set):
        if not isinstance(ad2, set):
            return False
        if len(ad1) != len(ad2):
            return False
        if ad1 != ad2:
            return False
    elif isinstance(ad1, dict):
        if not isinstance(ad2, dict):
            return False
        if set(ad1.keys()) != set(ad2.keys()):
            return False
        for k in ad1.keys():
            if not tree_data_is_equal(ad1[k], ad2[k]):
                return False
    elif isinstance(ad1, jnp.ndarray):
        if not isinstance(ad2, jnp.ndarray):
            return False
        if not jnp.array_equal(ad1, ad2):
            return False
    elif ad1 != ad2:
        return False

    # if we get here things are true
    return True
