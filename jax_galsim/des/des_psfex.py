# This is a JAX port of galsim.des.des_psfex (galsim/des/des_psfex.py).
# The reading of the PSFEx file is unchanged host-side I/O; the per-position
# PSF evaluation (getPSFArray) is reimplemented in JAX so it can be jitted,
# vmapped, and differentiated with respect to the image position.
import os

import galsim as _galsim
import galsim.des  # noqa: F401  (populates _galsim.des for @implements below)
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from jax_galsim._pyfits import pyfits
from jax_galsim.core.utils import (
    cast_numpy_array_to_native_byte_order,
    cast_to_float,
    implements,
)
from jax_galsim.errors import GalSimIncompatibleValuesError
from jax_galsim.fits import FitsHeader
from jax_galsim.image import Image
from jax_galsim.interpolant import Lanczos
from jax_galsim.interpolatedimage import InterpolatedImage
from jax_galsim.wcs import readFromFitsHeader

LAX_DES_PSFEX = """\
The JAX-GalSim version of ``DES_PSFEx`` does not register itself with the
GalSim config framework (the ``des_psfex`` input type and ``DES_PSFEx`` object
type are not available), since JAX-GalSim does not implement config processing.

As a PyTree, the data read from the PSFEx file (the PCA basis and the
polynomial zero points and scales) together with the ``wcs`` are traced
children, so a batch of PSFEx models can be stacked and evaluated in a single
``jit``/``vmap`` call. Only ``fit_order`` and ``fit_size`` are static
auxiliary data, since they set the number of polynomial terms and hence the
shapes of the traced arrays; batching therefore applies to models sharing a
polynomial degree, which is the usual case within one instrument and reduction.

``file_name`` is deliberately *not* part of the PyTree, so that models read
from different files share a tree structure and can be batched together. An
instance rebuilt by ``tree_unflatten`` has ``file_name`` set to ``None``.
"""


@implements(_galsim.des.DES_PSFEx, lax_description=LAX_DES_PSFEX, module="galsim.des")
@register_pytree_node_class
class DES_PSFEx:
    _req_params = {"file_name": str}
    _opt_params = {"dir": str, "image_file_name": str}
    _single_params = []
    _takes_rng = False

    def __init__(self, file_name, image_file_name=None, wcs=None, dir=None):
        if dir:
            if not isinstance(file_name, str):
                raise TypeError("file_name must be a string")
            file_name = os.path.join(dir, file_name)
            if image_file_name is not None:
                image_file_name = os.path.join(dir, image_file_name)
        self.file_name = file_name
        if image_file_name:
            if wcs is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot provide both image_file_name and wcs",
                    image_file_name=image_file_name,
                    wcs=wcs,
                )
            header = FitsHeader(file_name=image_file_name)
            wcs, origin = readFromFitsHeader(header)
            self.wcs = wcs
        elif wcs:
            self.wcs = wcs
        else:
            self.wcs = None
        self.read()

    def read(self):
        if isinstance(self.file_name, str):
            hdu_list = pyfits.open(self.file_name)
            hdu = hdu_list[1]
        else:
            hdu = self.file_name
            hdu_list = None
        pol_naxis = hdu.header["POLNAXIS"]

        pol_name1 = hdu.header["POLNAME1"]
        pol_name2 = hdu.header["POLNAME2"]

        pol_zero1 = hdu.header["POLZERO1"]
        pol_zero2 = hdu.header["POLZERO2"]
        pol_scal1 = hdu.header["POLSCAL1"]
        pol_scal2 = hdu.header["POLSCAL2"]

        pol_ngrp = hdu.header["POLNGRP"]
        pol_group1 = hdu.header["POLGRP1"]
        pol_group2 = hdu.header["POLGRP2"]
        pol_deg = hdu.header["POLDEG1"]

        psf_naxis = hdu.header["PSFNAXIS"]
        psf_axis1 = hdu.header["PSFAXIS1"]
        psf_axis2 = hdu.header["PSFAXIS2"]
        psf_axis3 = hdu.header["PSFAXIS3"]
        psf_samp = hdu.header["PSF_SAMP"]

        basis = hdu.data.field("PSF_MASK")[0]

        if hdu_list:
            hdu_list.close()

        try:
            assert pol_naxis == 2
            assert pol_name1.startswith("X") and pol_name1.endswith("IMAGE")
            assert pol_name2.startswith("Y") and pol_name2.endswith("IMAGE")
            assert pol_ngrp == 1
            assert pol_group1 == 1
            assert pol_group2 == 1
            assert psf_naxis == 3
            assert psf_axis3 == ((pol_deg + 1) * (pol_deg + 2)) // 2
            assert basis.shape[0] == psf_axis3
            assert basis.shape[1] == psf_axis2
            assert basis.shape[2] == psf_axis1
        except AssertionError as e:
            raise OSError("PSFEx file %s is not as expected.\n%r" % (self.file_name, e))

        # The basis and the polynomial zero points/scales are traced children,
        # so that models from different files can be batched together. PSFEx
        # stores the cube big-endian, which JAX will not accept, so convert to
        # the native byte order first.
        self.basis = jnp.asarray(cast_numpy_array_to_native_byte_order(basis))
        # fit_order/fit_size stay static: they set the number of polynomial
        # terms and hence the shapes of the traced arrays.
        self.fit_order = int(pol_deg)
        self.fit_size = int(psf_axis3)
        self.x_zero = cast_to_float(pol_zero1)
        self.y_zero = cast_to_float(pol_zero2)
        self.x_scale = cast_to_float(pol_scal1)
        self.y_scale = cast_to_float(pol_scal2)
        self.sample_scale = cast_to_float(psf_samp)

    @implements(_galsim.des.DES_PSFEx.getSampleScale)
    def getSampleScale(self):
        return self.sample_scale

    @implements(_galsim.des.DES_PSFEx.getLocalWCS)
    def getLocalWCS(self, image_pos):
        if self.wcs:
            return self.wcs.local(image_pos)
        else:
            return None

    @implements(_galsim.des.DES_PSFEx.getPSF)
    def getPSF(self, image_pos, gsparams=None):
        im = Image(self.getPSFArray(image_pos))
        psf = InterpolatedImage(
            im,
            scale=self.sample_scale,
            flux=1,
            x_interpolant=Lanczos(3),
            gsparams=gsparams,
        )
        if self.wcs:
            psf = self.wcs.toWorld(psf, image_pos=image_pos)
        return psf

    @implements(_galsim.des.DES_PSFEx.getPSFArray)
    def getPSFArray(self, image_pos):
        xto = self._powers((image_pos.x - self.x_zero) / self.x_scale)
        yto = self._powers((image_pos.y - self.y_zero) / self.y_scale)
        order = self.fit_order
        # order is a static Python int, so this comprehension is unrolled at
        # trace time; it mirrors galsim's ordering of the polynomial terms.
        P = jnp.stack(
            [
                xto[nx] * yto[ny]
                for ny in range(order + 1)
                for nx in range(order + 1 - ny)
            ]
        )
        return jnp.tensordot(P, self.basis, (0, 0)).astype(jnp.float32)

    def _powers(self, x):
        # JAX-safe replacement for galsim's ``np.empty`` + in-place loop: build
        # [1, x, x**2, ..., x**order] via a cumulative product (same recurrence
        # as galsim, but without an in-place update, which JAX forbids). The
        # leading 1 takes x's dtype so the concatenation cannot silently promote
        # the result to a wider type than the input position.
        x = jnp.asarray(x)
        return jnp.concatenate(
            [
                jnp.ones((1,), dtype=x.dtype),
                jnp.cumprod(jnp.full((self.fit_order,), x)),
            ]
        )

    def tree_flatten(self):
        """Flatten into traced children and static auxiliary data.

        The PSFEx data (basis, polynomial zero points and scales) and the
        ``wcs`` are traced children, so that models read from different files
        can be stacked and evaluated in one ``jit``/``vmap`` call. Only
        ``fit_order``/``fit_size`` are auxiliary, since they fix the number of
        polynomial terms and hence the shapes of the traced arrays.
        ``file_name`` is not part of the tree, so that models from different
        files share a tree structure.
        """
        children = (
            self.wcs,
            self.basis,
            self.x_zero,
            self.y_zero,
            self.x_scale,
            self.y_scale,
            self.sample_scale,
        )
        aux_data = {"fit_order": self.fit_order, "fit_size": self.fit_size}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Rebuild an instance without re-reading the file.

        ``__init__`` opens the PSFEx file, so (following ``CelestialCoord`` /
        ``Image``) we construct via ``object.__new__`` and restore attributes
        directly from the flattened representation. ``file_name`` is not
        carried through the tree and is set to ``None``.
        """
        obj = object.__new__(cls)
        (
            obj.wcs,
            obj.basis,
            obj.x_zero,
            obj.y_zero,
            obj.x_scale,
            obj.y_scale,
            obj.sample_scale,
        ) = children
        obj.fit_order = aux_data["fit_order"]
        obj.fit_size = aux_data["fit_size"]
        obj.file_name = None
        return obj
