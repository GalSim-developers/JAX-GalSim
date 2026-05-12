import numpy as np
import pytest

import jax_galsim


def test_image_jax_view_raises():
    im = jax_galsim.ImageD(np.arange(20).reshape(4, 5))
    with pytest.raises(NotImplementedError):
        im.view()
