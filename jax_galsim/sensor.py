import galsim as _galsim
from jax._src.numpy.util import implements
from jax.tree_util import register_pytree_node_class

from jax_galsim.errors import GalSimUndefinedBoundsError
from jax_galsim.position import PositionI


@implements(_galsim.Sensor)
@register_pytree_node_class
class Sensor:
    def __init__(self):
        pass

    @implements(_galsim.Sensor.accumulate)
    def accumulate(self, photons, image, orig_center=None, resume=False):
        if not image.bounds.isDefined():
            raise GalSimUndefinedBoundsError(
                "Calling accumulate on image with undefined bounds"
            )
        return photons.addTo(image)

    @implements(_galsim.Sensor.calculate_pixel_areas)
    def calculate_pixel_areas(self, image, orig_center=PositionI(0, 0), use_flux=True):
        return 1.0

    def updateRNG(self, rng):
        pass

    def __repr__(self):
        return "galsim.Sensor()"

    def __eq__(self, other):
        return self is other or (
            isinstance(other, Sensor) and repr(self) == repr(other)
        )  # Checks that neither is a subclass

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(repr(self))

    def tree_flatten(self):
        children = tuple()
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()
