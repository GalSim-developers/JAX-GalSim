import galsim as _galsim
import numpy as np

from jax_galsim.bounds import BoundsI
from jax_galsim.core.utils import implements
from jax_galsim.errors import (
    GalSimHSMError,
    GalSimIncompatibleValuesError,
    GalSimValueError,
)
from jax_galsim.image import Image, ImageD, ImageF, ImageI
from jax_galsim.position import PositionD
from jax_galsim.shear import Shear

HSM_LAX_DOCS = """\
Contrary to most other classes and objects in jax-galsim, the HSM
functionality is not implemented using JAX primitives.

All HSM-related methods directly rely on the original GalSim
implementation and therefore:
    - do not run on GPU or TPU
    - are not JIT-compilable
    - do not benefit from JAX transformations (vmap, grad, etc.)

As a result, all computations are performed on the CPU using classical
GalSim code, and HSM should be considered outside the JAX execution model.
"""


@implements(_galsim.hsm.ShapeData, lax_description=HSM_LAX_DOCS)
class ShapeData:
    def __init__(
        self,
        image_bounds=BoundsI(),
        moments_status=-1,
        observed_shape=Shear(),
        moments_sigma=-1.0,
        moments_amp=-1.0,
        moments_centroid=PositionD(),
        moments_rho4=-1.0,
        moments_n_iter=0,
        correction_status=-1,
        corrected_e1=-10.0,
        corrected_e2=-10.0,
        corrected_g1=-10.0,
        corrected_g2=-10.0,
        meas_type="None",
        corrected_shape_err=-1.0,
        correction_method="None",
        resolution_factor=-1.0,
        psf_sigma=-1.0,
        psf_shape=Shear(),
        error_message="",
    ):
        # Avoid empty string, which can caus problems in C++ layer.
        if error_message == "":
            error_message = "None"

        if not isinstance(image_bounds, BoundsI):
            raise TypeError("image_bounds must be a BoundsI instance")

        # The others will raise an appropriate TypeError from the call to _galsim.ShapeData
        # when converting to int, float, etc.
        self._data = _galsim._galsim.ShapeData(
            image_bounds._b,
            int(moments_status),
            float(observed_shape.e1),
            float(observed_shape.e2),
            float(moments_sigma),
            float(moments_amp),
            moments_centroid._p,
            float(moments_rho4),
            int(moments_n_iter),
            int(correction_status),
            float(corrected_e1),
            float(corrected_e2),
            float(corrected_g1),
            float(corrected_g2),
            str(meas_type),
            float(corrected_shape_err),
            str(correction_method),
            float(resolution_factor),
            float(psf_sigma),
            float(psf_shape.e1),
            float(psf_shape.e2),
            str(error_message),
        )

    @property
    def image_bounds(self):
        return BoundsI(self._data.image_bounds)

    @property
    def moments_status(self):
        return self._data.moments_status

    @property
    def observed_e1(self):
        return self._data.observed_e1

    @property
    def observed_e2(self):
        return self._data.observed_e2

    @property
    def observed_shape(self):
        return Shear(e1=self.observed_e1, e2=self.observed_e2)

    @property
    def moments_sigma(self):
        return self._data.moments_sigma

    @property
    def moments_amp(self):
        return self._data.moments_amp

    @property
    def moments_centroid(self):
        return PositionD(self._data.moments_centroid)

    @property
    def moments_rho4(self):
        return self._data.moments_rho4

    @property
    def moments_n_iter(self):
        return self._data.moments_n_iter

    @property
    def correction_status(self):
        return self._data.correction_status

    @property
    def corrected_e1(self):
        return self._data.corrected_e1

    @property
    def corrected_e2(self):
        return self._data.corrected_e2

    @property
    def corrected_g1(self):
        return self._data.corrected_g1

    @property
    def corrected_g2(self):
        return self._data.corrected_g2

    @property
    def meas_type(self):
        return self._data.meas_type

    @property
    def corrected_shape_err(self):
        return self._data.corrected_shape_err

    @property
    def correction_method(self):
        return self._data.correction_method

    @property
    def resolution_factor(self):
        return self._data.resolution_factor

    @property
    def psf_sigma(self):
        return self._data.psf_sigma

    @property
    def psf_shape(self):
        return Shear(e1=self._data.psf_e1, e2=self._data.psf_e2)

    @property
    def error_message(self):
        # We use "None" in C++ ShapeData to indicate no error messages to avoid problems on
        # (some) Macs using zero-length strings.  Here, we revert that back to "".
        if self._data.error_message == "None":
            return ""
        else:
            return self._data.error_message

    def __repr__(self):
        s = "galsim.hsm.ShapeData("
        if self.image_bounds.isDefined():
            s += "image_bounds=%r, " % self.image_bounds
        if self.moments_status != -1:
            s += "moments_status=%r, " % self.moments_status
        # Always include this one:
        s += "observed_shape=%r" % self.observed_shape
        if self.moments_sigma != -1:
            s += ", moments_sigma=%r" % self.moments_sigma
        if self.moments_amp != -1:
            s += ", moments_amp=%r" % self.moments_amp
        if self.moments_centroid != PositionD():
            s += ", moments_centroid=%r" % self.moments_centroid
        if self.moments_rho4 != -1:
            s += ", moments_rho4=%r" % self.moments_rho4
        if self.moments_n_iter != 0:
            s += ", moments_n_iter=%r" % self.moments_n_iter
        if self.correction_status != -1:
            s += ", correction_status=%r" % self.correction_status
        if self.corrected_e1 != -10.0:
            s += ", corrected_e1=%r" % self.corrected_e1
        if self.corrected_e2 != -10.0:
            s += ", corrected_e2=%r" % self.corrected_e2
        if self.corrected_g1 != -10.0:
            s += ", corrected_g1=%r" % self.corrected_g1
        if self.corrected_g2 != -10.0:
            s += ", corrected_g2=%r" % self.corrected_g2
        if self.meas_type != "None":
            s += ", meas_type=%r" % self.meas_type
        if self.corrected_shape_err != -1.0:
            s += ", corrected_shape_err=%r" % self.corrected_shape_err
        if self.correction_method != "None":
            s += ", correction_method=%r" % self.correction_method
        if self.resolution_factor != -1.0:
            s += ", resolution_factor=%r" % self.resolution_factor
        if self.psf_sigma != -1.0:
            s += ", psf_sigma=%r" % self.psf_sigma
        if self.psf_shape != Shear():
            s += ", psf_shape=%r" % self.psf_shape
        if self.error_message != "":
            s += ", error_message=%r" % self.error_message
        s += ")"
        return s

    def __eq__(self, other):
        return self is other or (
            isinstance(other, ShapeData) and self._getinitargs() == other._getinitargs()
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(("galsim.hsm.ShapeData", self._getinitargs()))

    def _getinitargs(self):
        return (
            self.image_bounds,
            self.moments_status,
            self.observed_shape,
            self.moments_sigma,
            self.moments_amp,
            self.moments_centroid,
            self.moments_rho4,
            self.moments_n_iter,
            self.correction_status,
            self.corrected_e1,
            self.corrected_e2,
            self.corrected_g1,
            self.corrected_g2,
            self.meas_type,
            self.corrected_shape_err,
            self.correction_method,
            self.resolution_factor,
            self.psf_sigma,
            self.psf_shape,
            self.error_message,
        )

    def __getstate__(self):
        return self._getinitargs()

    def __setstate__(self, state):
        self.__init__(*state)

    @implements(_galsim.hsm.ShapeData.applyWCS)
    def applyWCS(self, wcs, image_pos):
        jac = wcs.jacobian(image_pos=image_pos)
        scale, shear, theta, flip = jac.getDecomposition()

        # Fix moments_sigma
        moments_sigma = self.moments_sigma * scale

        # Fix observed_shape
        shape = self.observed_shape
        # First the flip, if any.
        if flip:
            shape = Shear(g1=-shape.g1, g2=shape.g2)
        # Next the rotation
        shape = Shear(g=shape.g, beta=shape.beta + theta)
        # Finally the shear
        observed_shape = shear + shape

        # Fix moments_centroid
        moments_centroid = jac.toWorld(self.moments_centroid) - jac.toWorld(image_pos)

        return ShapeData(
            image_bounds=self.image_bounds,
            moments_status=self.moments_status,
            observed_shape=observed_shape,
            moments_sigma=moments_sigma,
            moments_amp=self.moments_amp,
            moments_centroid=moments_centroid,
            moments_rho4=self.moments_rho4,
            moments_n_iter=self.moments_n_iter,
            error_message=self.error_message,
        )
        # The other values are reset to the defaults, since they are
        # results from EstimateShear.


@implements(_galsim.hsm.HSMParams, lax_description=HSM_LAX_DOCS)
class HSMParams:
    def __init__(
        self,
        nsig_rg=3.0,
        nsig_rg2=3.6,
        max_moment_nsig2=0,
        regauss_too_small=1,
        adapt_order=2,
        convergence_threshold=1.0e-6,
        max_mom2_iter=400,
        num_iter_default=-1,
        bound_correct_wt=0.25,
        max_amoment=8000.0,
        max_ashift=15.0,
        ksb_moments_max=4,
        ksb_sig_weight=0.0,
        ksb_sig_factor=1.0,
        failed_moments=-1000.0,
    ):
        if max_moment_nsig2 != 0:
            from .deprecated import depr

            depr("max_moment_nsig2", 2.4, "", "This parameter is no longer used.")

        self._nsig_rg = float(nsig_rg)
        self._nsig_rg2 = float(nsig_rg2)
        self._regauss_too_small = int(regauss_too_small)
        self._adapt_order = int(adapt_order)
        self._convergence_threshold = float(convergence_threshold)
        self._max_mom2_iter = int(max_mom2_iter)
        self._num_iter_default = int(num_iter_default)
        self._bound_correct_wt = float(bound_correct_wt)
        self._max_amoment = float(max_amoment)
        self._max_ashift = float(max_ashift)
        self._ksb_moments_max = int(ksb_moments_max)
        self._ksb_sig_weight = float(ksb_sig_weight)
        self._ksb_sig_factor = float(ksb_sig_factor)
        self._failed_moments = float(failed_moments)
        self._make_hsmp()

    def _make_hsmp(self):
        self._hsmp = _galsim._galsim.HSMParams(*self._getinitargs())

    def _getinitargs(self):
        # TODO: For now, leave 3rd param as unused max_moment_nsig2.
        #       Remove it at version 3.0 to avoid changing C++ API yet.
        return (
            self.nsig_rg,
            self.nsig_rg2,
            0.0,
            self.regauss_too_small,
            self.adapt_order,
            self.convergence_threshold,
            self.max_mom2_iter,
            self.num_iter_default,
            self.bound_correct_wt,
            self.max_amoment,
            self.max_ashift,
            self.ksb_moments_max,
            self.ksb_sig_weight,
            self.ksb_sig_factor,
            self.failed_moments,
        )

    @property
    def nsig_rg(self):
        return self._nsig_rg

    @property
    def nsig_rg2(self):
        return self._nsig_rg2

    @property
    def max_moment_nsig2(self):
        return 0.0

    @property
    def regauss_too_small(self):
        return self._regauss_too_small

    @property
    def adapt_order(self):
        return self._adapt_order

    @property
    def convergence_threshold(self):
        return self._convergence_threshold

    @property
    def max_mom2_iter(self):
        return self._max_mom2_iter

    @property
    def num_iter_default(self):
        return self._num_iter_default

    @property
    def bound_correct_wt(self):
        return self._bound_correct_wt

    @property
    def max_amoment(self):
        return self._max_amoment

    @property
    def max_ashift(self):
        return self._max_ashift

    @property
    def ksb_moments_max(self):
        return self._ksb_moments_max

    @property
    def ksb_sig_weight(self):
        return self._ksb_sig_weight

    @property
    def ksb_sig_factor(self):
        return self._ksb_sig_factor

    @property
    def failed_moments(self):
        return self._failed_moments

    @staticmethod
    def check(hsmparams, default=None):
        """Checks that hsmparams is either a valid HSMParams instance or None.

        In the former case, it returns hsmparams, in the latter it returns default
        (HSMParams.default if no other default specified).
        """
        if hsmparams is None:
            return default if default is not None else HSMParams.default
        elif not isinstance(hsmparams, HSMParams):
            raise TypeError("Invalid HSMParams: %s" % hsmparams)
        else:
            return hsmparams

    def __repr__(self):
        return ("galsim.hsm.HSMParams(" + 14 * "%r," + "%r)") % self._getinitargs()

    def __eq__(self, other):
        return self is other or (
            isinstance(other, HSMParams) and self._getinitargs() == other._getinitargs()
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(("galsim.hsm.HSMParams", self._getinitargs()))

    def __getstate__(self):
        d = self.__dict__.copy()
        del d["_hsmp"]
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._make_hsmp()


# We use the default a lot, so make it a class attribute.
HSMParams.default = HSMParams()


# A helper function that checks if the weight and the badpix bounds are
# consistent with that of the image, and that the weight is non-negative.
def _checkWeightAndBadpix(image, weight=None, badpix=None):
    # Check that the weight and badpix, if given, are sensible and compatible
    # with the image.
    if weight is not None:
        if weight.bounds != image.bounds:
            raise GalSimIncompatibleValuesError(
                "Weight image does not have same bounds as the input Image.",
                weight=weight,
                image=image,
            )
            # also make sure there are no negative values

        if np.any(weight.array < 0):
            raise GalSimValueError(
                "Weight image cannot contain negative values.", weight
            )

    if badpix is not None and badpix.bounds != image.bounds:
        raise GalSimIncompatibleValuesError(
            "Badpix image does not have the same bounds as the input Image.",
            badpix=badpix,
            image=image,
        )


# A helper function for taking input weight and badpix Images, and returning a weight Image in the
# format that the C++ functions want
def _convertMask(image, weight=None, badpix=None):
    # Convert from input weight and badpix images to a single mask image needed by C++ functions.
    # This is used by EstimateShear() and FindAdaptiveMom().

    # if no weight image was supplied, make an int array (same size as gal image) filled with 1's
    if weight is None:
        mask = ImageI(bounds=image.bounds, init_value=1)
    else:
        # if weight is an ImageI, then we can use it as the mask image:
        if weight.dtype == np.int32:
            if not badpix:
                mask = weight
            else:
                # If we need to mask bad pixels, we'll need a copy anyway.
                mask = ImageI(weight)

        # otherwise, we need to convert it to the right type
        else:
            mask = ImageI(bounds=image.bounds, init_value=0)
            mask.array[weight.array > 0.0] = 1

    # if badpix image was supplied, identify the nonzero (bad) pixels and set them to zero in weight
    # image; also check bounds
    if badpix is not None:
        mask.array[badpix.array != 0] = 0

    # if no pixels are used, raise an exception
    if not np.any(mask.array):
        raise GalSimHSMError("No pixels are being used!")

    # finally, return the Image for the weight map
    return mask


# A simpler helper function to force images to be of type ImageF or ImageD
def _convertImage(image):
    # Convert the given image to the correct format needed to pass to the C++ layer.
    # This is used by EstimateShear() and FindAdaptiveMom().

    # if weight is not of type float/double, convert to float/double
    if image.dtype == np.int16 or image.dtype == np.uint16:
        image = ImageF(image)
    elif image.dtype == np.int32 or image.dtype == np.uint32:
        image = ImageD(image)

    return image


@implements(_galsim.hsm.EstimateShear, lax_description=HSM_LAX_DOCS)
def EstimateShear(
    gal_image,
    PSF_image,
    weight=None,
    badpix=None,
    sky_var=0.0,
    shear_est="REGAUSS",
    recompute_flux="FIT",
    guess_sig_gal=5.0,
    guess_sig_PSF=3.0,
    precision=1.0e-6,
    guess_centroid=None,
    strict=True,
    check=True,
    hsmparams=None,
):
    # prepare inputs to C++ routines: ImageF or ImageD for galaxy, PSF, and ImageI for weight map
    gal_image = _convertImage(gal_image)
    PSF_image = _convertImage(PSF_image)
    hsmparams = HSMParams.check(hsmparams)
    if check:
        _checkWeightAndBadpix(gal_image, weight=weight, badpix=badpix)
    weight = _convertMask(gal_image, weight=weight, badpix=badpix)

    if guess_centroid is None:
        guess_centroid = gal_image.true_center
    try:
        result = ShapeData()
        _galsim._galsim.EstimateShearView(
            result._data,
            gal_image._image,
            PSF_image._image,
            weight._image,
            float(sky_var),
            shear_est.upper(),
            recompute_flux.upper(),
            float(guess_sig_gal),
            float(guess_sig_PSF),
            float(precision),
            guess_centroid._p,
            hsmparams._hsmp,
        )
        return result
    except RuntimeError as err:
        if strict:
            raise GalSimHSMError(str(err)) from None
        else:
            return ShapeData(error_message=str(err))


@implements(_galsim.hsm.FindAdaptiveMom, lax_description=HSM_LAX_DOCS)
def FindAdaptiveMom(
    object_image,
    weight=None,
    badpix=None,
    guess_sig=5.0,
    precision=1.0e-6,
    guess_centroid=None,
    strict=True,
    check=True,
    round_moments=False,
    hsmparams=None,
    use_sky_coords=False,
):
    # prepare inputs to C++ routines: ImageF or ImageD for galaxy, PSF, and ImageI for weight map
    object_image = _convertImage(object_image)
    hsmparams = HSMParams.check(hsmparams)
    if check:
        _checkWeightAndBadpix(object_image, weight=weight, badpix=badpix)

    weight = _convertMask(object_image, weight=weight, badpix=badpix)

    if guess_centroid is None:
        guess_centroid = object_image.true_center

    try:
        result = ShapeData()
        _galsim._galsim.FindAdaptiveMomView(
            result._data,
            object_image._image,
            weight._image,
            float(guess_sig),
            float(precision),
            guess_centroid._p,
            bool(round_moments),
            hsmparams._hsmp,
        )

        if use_sky_coords:
            result = result.applyWCS(
                object_image.wcs, image_pos=object_image.true_center
            )
        return result
    except RuntimeError as err:
        if strict:
            raise GalSimHSMError(str(err)) from None
        else:
            return ShapeData(error_message=str(err))


# make FindAdaptiveMom a method of Image class
Image.FindAdaptiveMom = FindAdaptiveMom
