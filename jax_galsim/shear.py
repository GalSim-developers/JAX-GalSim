import galsim as _galsim
import jax.numpy as jnp
from galsim.errors import GalSimIncompatibleValuesError
from jax.tree_util import register_pytree_node_class

from jax_galsim.angle import Angle, _Angle, radians
from jax_galsim.core.utils import ensure_hashable, implements


@register_pytree_node_class
@implements(
    _galsim.Shear,
    lax_description="""\
The jax_galsim implementation of ``Shear`` does not perform range checking of the \
shear (e.g., ``|g| <= 1``) upon construction.""",
)
class Shear(object):
    def __init__(self, *args, **kwargs):
        # There is no valid set of >2 keyword arguments, so raise an exception in this case:
        if len(kwargs) > 2:
            raise TypeError(
                "Shear constructor received >2 keyword arguments: %s" % kwargs.keys()
            )

        if len(args) > 1:
            raise TypeError(
                "Shear constructor received >1 non-keyword arguments: %s" % args
            )

        # If a component of e, g, or eta, then require that the other component is zero if not set,
        # and don't allow specification of mixed pairs like e1 and g2.
        # Also, require a position angle if we didn't get g1/g2, e1/e2, or eta1/eta2

        # Unnamed arg must be a complex shear
        if len(args) == 1:
            self._g = args[0]
            if not (jnp.all(jnp.iscomplex(self._g)) or isinstance(self._g, complex)):
                raise TypeError(
                    "Non-keyword argument to Shear must be complex g1 + 1j * g2"
                )

        # Empty constructor means shear == (0,0)
        elif not kwargs:
            self._g = 0j

        # g1,g2
        elif "g1" in kwargs or "g2" in kwargs:
            g1 = kwargs.pop("g1", 0.0)
            g2 = kwargs.pop("g2", 0.0)
            self._g = g1 + 1j * g2

        # e1,e2
        elif "e1" in kwargs or "e2" in kwargs:
            e1 = kwargs.pop("e1", 0.0)
            e2 = kwargs.pop("e2", 0.0)
            absesq = e1**2 + e2**2
            self._g = (e1 + 1j * e2) * self._e2g(absesq)

        # eta1,eta2
        elif "eta1" in kwargs or "eta2" in kwargs:
            eta1 = kwargs.pop("eta1", 0.0)
            eta2 = kwargs.pop("eta2", 0.0)
            eta = eta1 + 1j * eta2
            abseta = abs(eta)
            self._g = eta * self._eta2g(abseta)

        # g,beta
        elif "g" in kwargs:
            if "beta" not in kwargs:
                raise GalSimIncompatibleValuesError(
                    "Shear constructor requires beta when g is specified.",
                    g=kwargs["g"],
                    beta=None,
                )
            beta = kwargs.pop("beta")
            if not isinstance(beta, Angle):
                raise TypeError("beta must be an Angle instance.")
            g = kwargs.pop("g")
            self._g = g * jnp.exp(2j * beta.rad)

        # e,beta
        elif "e" in kwargs:
            if "beta" not in kwargs:
                raise GalSimIncompatibleValuesError(
                    "Shear constructor requires beta when e is specified.",
                    e=kwargs["e"],
                    beta=None,
                )
            beta = kwargs.pop("beta")
            if not isinstance(beta, Angle):
                raise TypeError("beta must be an Angle instance.")
            e = kwargs.pop("e")
            self._g = self._e2g(e**2) * e * jnp.exp(2j * beta.rad)

        # eta,beta
        elif "eta" in kwargs:
            if "beta" not in kwargs:
                raise GalSimIncompatibleValuesError(
                    "Shear constructor requires beta when eta is specified.",
                    eta=kwargs["eta"],
                    beta=None,
                )
            beta = kwargs.pop("beta")
            if not isinstance(beta, Angle):
                raise TypeError("beta must be an Angle instance.")
            eta = kwargs.pop("eta")
            self._g = self._eta2g(eta) * eta * jnp.exp(2j * beta.rad)

        # q,beta
        elif "q" in kwargs:
            if "beta" not in kwargs:
                raise GalSimIncompatibleValuesError(
                    "Shear constructor requires beta when q is specified.",
                    q=kwargs["q"],
                    beta=None,
                )
            beta = kwargs.pop("beta")
            if not isinstance(beta, Angle):
                raise TypeError("beta must be an Angle instance.")
            q = kwargs.pop("q")
            eta = -jnp.log(q)
            self._g = self._eta2g(eta) * eta * jnp.exp(2j * beta.rad)

        elif "beta" in kwargs:
            raise GalSimIncompatibleValuesError(
                "beta provided to Shear constructor, but not g/e/eta/q",
                beta=kwargs["beta"],
                e=None,
                g=None,
                q=None,
                eta=None,
            )

        # check for the case where there are 1 or 2 kwargs that are not valid ones for
        # initializing a Shear
        if kwargs:
            raise TypeError(
                "Shear constructor got unexpected extra argument(s): %s" % kwargs.keys()
            )

    @property
    @implements(_galsim.Shear.g1)
    def g1(self):
        return self._g.real

    @property
    @implements(_galsim.Shear.g2)
    def g2(self):
        return self._g.imag

    @property
    @implements(_galsim.Shear.g)
    def g(self):
        return jnp.abs(self._g)

    @property
    @implements(_galsim.Shear.beta)
    def beta(self):
        return _Angle(0.5 * jnp.angle(self._g))

    @property
    @implements(_galsim.Shear.shear)
    def shear(self):
        return self._g

    @property
    @implements(_galsim.Shear.e1)
    def e1(self):
        return self._g.real * self._g2e(self.g**2)

    @property
    @implements(_galsim.Shear.e2)
    def e2(self):
        return self._g.imag * self._g2e(self.g**2)

    @property
    @implements(_galsim.Shear.e)
    def e(self):
        return self.g * self._g2e(self.g**2)

    @property
    @implements(_galsim.Shear.esq)
    def esq(self):
        return self.e**2

    @property
    @implements(_galsim.Shear.eta1)
    def eta1(self):
        return self._g.real * self._g2eta(self.g)

    @property
    @implements(_galsim.Shear.eta2)
    def eta2(self):
        return self._g.imag * self._g2eta(self.g)

    @property
    @implements(_galsim.Shear.eta)
    def eta(self):
        return self.g * self._g2eta(self.g)

    @property
    @implements(_galsim.Shear.q)
    def q(self):
        return (1.0 - self.g) / (1.0 + self.g)

    # Helpers to convert between different conventions
    # Note: These return the scale factor by which to multiply.  Not the final value.
    def _g2e(self, absgsq):
        return 2.0 / (1.0 + absgsq)

    def _e2g(self, absesq):
        return jnp.where(
            absesq > 1.0e-4,
            1.0 / (1.0 + jnp.sqrt(1.0 - absesq)),
            # Avoid numerical issues near e=0 using Taylor expansion
            0.5 + absesq * (0.125 + absesq * (0.0625 + absesq * 0.0390625)),
        )

    def _g2eta(self, absg):
        absgsq = absg * absg
        return jnp.where(
            absg > 1.0e-4,
            2.0 * jnp.arctanh(absg) / absg,
            # This doesn't have as much trouble with accuracy, but have to avoid absg=0,
            # so might as well Taylor expand for small values.
            2.0 + absgsq * ((2.0 / 3.0) + absgsq * 0.4),
        )

    def _eta2g(self, abseta):
        absetasq = abseta * abseta
        return jnp.where(
            abseta > 1.0e-4,
            jnp.tanh(0.5 * abseta) / abseta,
            0.5 + absetasq * ((-1.0 / 24.0) + absetasq * (1.0 / 240.0)),
        )

    # define all the various operators on Shear objects
    def __neg__(self):
        return _Shear(-self._g)

    # order of operations: shear by other._shear, then by self._shear
    def __add__(self, other):
        return _Shear((self._g + other._g) / (1.0 + self._g.conjugate() * other._g))

    # order of operations: shear by -other._shear, then by self._shear
    def __sub__(self, other):
        return self + (-other)

    def __eq__(self, other):
        return self is other or (isinstance(other, Shear) and self._g == other._g)

    def __ne__(self, other):
        return not self.__eq__(other)

    @implements(_galsim.Shear.getMatrix)
    def getMatrix(self):
        return jnp.array(
            [[1.0 + self.g1, self.g2], [self.g2, 1.0 - self.g1]]
        ) / jnp.sqrt(1.0 - self.g**2)

    @implements(_galsim.Shear.rotationWith)
    def rotationWith(self, other):
        # Save a little time by only working on the first column.
        S3 = self.getMatrix().dot(other.getMatrix()[:, :1])
        R = (-(self + other)).getMatrix().dot(S3)
        theta = jnp.arctan2(R[1, 0], R[0, 0])
        return theta * radians

    def __repr__(self):
        return "galsim.Shear(%r)" % (ensure_hashable(self.shear))

    def __str__(self):
        return "galsim.Shear(g1=%s,g2=%s)" % (
            ensure_hashable(self.g1),
            ensure_hashable(self.g2),
        )

    def __hash__(self):
        return hash(ensure_hashable(self._g))

    def tree_flatten(self):
        children = (self._g,)
        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreates an instance of the class from flatten representation"""
        del aux_data  # unused in this class
        obj = cls.__new__(cls)
        obj._g = children[0]
        return obj

    @classmethod
    def from_galsim(cls, galsim_shear):
        """Create a jax_galsim `Shear` from a `galsim.Shear` object."""
        return cls(g1=galsim_shear.g1, g2=galsim_shear.g2)

    def to_galsim(self):
        """Create a galsim `Shear` from a `jax_galsim.Shear` object."""
        return _galsim.Shear(g1=float(self.g1), g2=float(self.g2))


@implements(_galsim._Shear)
def _Shear(shear):
    ret = Shear.__new__(Shear)
    ret._g = shear
    return ret
