# Copyright (c) 2012-2023 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

from dataclasses import dataclass

import jax.numpy as jnp

import galsim as _galsim
from jax_galsim.core.utils import implements
from jax_galsim.position import PositionD
from jax_galsim.bounds import BoundsI
from jax_galsim.shear import Shear
from jax_galsim.image import Image, ImageI, ImageF, ImageD
from jax_galsim.errors import GalSimValueError, GalSimHSMError, GalSimIncompatibleValuesError
from jax_galsim.core.utils import cast_to_float, cast_to_int

@implements(_galsim.hsm.ShapeData)
class ShapeData:
    def __init__(self, image_bounds=BoundsI(), moments_status=-1,
                 observed_shape=Shear(), moments_sigma=-1.0, moments_amp=-1.0,
                 moments_centroid=PositionD(), moments_rho4=-1.0, moments_n_iter=0,
                 correction_status=-1, corrected_e1=-10., corrected_e2=-10.,
                 corrected_g1=-10., corrected_g2=-10., meas_type="None",
                 corrected_shape_err=-1.0, correction_method="None",
                 resolution_factor=-1.0, psf_sigma=-1.0,
                 psf_shape=Shear(), error_message=""):

        # from https://github.com/GalSim-developers/GalSim/blob/releases/2.5/include/galsim/hsm/PSFCorr.h#L281
        #  This representation of an object shape contains information about observed shapes and shape
        #  estimators after PSF correction.  It also contains information about what PSF correction was
        #  used; if no PSF correction was carried out and only the observed moments were measured, the
        #  PSF correction method will be 'None'.  Note that observed shapes are bounded to lie in the
        #  range |e| < 1 or |g| < 1, so they can be represented using a Shear object.  In contrast,
        #  the PSF-corrected distortions and shears are not bounded at a maximum of 1 since they are
        #  shear estimators, and placing such a bound would bias the mean.  Thus, the corrected results
        #  are not represented using Shear objects, since it may not be possible to make a meaningful
        #  per-object conversion from distortion to shear (e.g., if |e|>1).
        
        # Avoid empty string, which can caus problems in C++ layer.
        if error_message == "": error_message = "None"

        if not isinstance(image_bounds, BoundsI):
            raise TypeError("image_bounds must be a BoundsI instance")

        # The others will raise an appropriate TypeError from the call to _galsim.ShapeData
        # when converting to int, float, etc.
        # self._data = _galsim.ShapeData(
        #     image_bounds._b, int(moments_status), observed_shape.e1, observed_shape.e2,
        #     float(moments_sigma), float(moments_amp), moments_centroid._p,
        #     float(moments_rho4), int(moments_n_iter), int(correction_status),
        #     float(corrected_e1), float(corrected_e2), float(corrected_g1), float(corrected_g2),
        #     str(meas_type), float(corrected_shape_err), str(correction_method),
        #     float(resolution_factor), float(psf_sigma), psf_shape.e1, psf_shape.e2,
        #     str(error_message))

        self._image_bounds = image_bounds
        self._moments_status = cast_to_int(moments_status)
        self._observed_e1 = observed_shape.e1
        self._observed_e2 = observed_shape.e2
        self._moments_sigma = cast_to_float(moments_sigma)
        self._moments_amp = cast_to_float(moments_amp)
        self._moments_centroid = moments_centroid
        self._moments_rho4 = cast_to_float(moments_rho4)
        self._moments_n_iter = cast_to_int(moments_n_iter)
        self._correction_status = cast_to_int(correction_status)
        self._corrected_e1 = cast_to_float(corrected_e1)
        self._corrected_e2 = cast_to_float(corrected_e2)
        self._corrected_g1 = cast_to_float(corrected_g1)
        self._corrected_g2 = cast_to_float(corrected_g2)
        self._meas_type = meas_type
        self._corrected_shape_err = cast_to_float(corrected_shape_err)
        self._correction_method = correction_method
        self._resolution_factor = cast_to_float(resolution_factor)
        self._psf_sigma = cast_to_float(psf_sigma)
        self._psf_e1 = psf_shape.e1
        self._psf_e2 = psf_shape.e2
        self._error_message = error_message

    @property
    def image_bounds(self): return BoundsI(self._image_bounds)
    @property
    def moments_status(self): return self._moments_status

    @property
    def observed_e1(self):
        return self._observed_e1

    @property
    def observed_e2(self):
        return self._observed_e2

    @property
    def observed_shape(self):
        return Shear(e1=self.observed_e1, e2=self.observed_e2)

    @property
    def moments_sigma(self): return self._moments_sigma
    @property
    def moments_amp(self): return self._moments_amp
    @property
    def moments_centroid(self): return PositionD(self._moments_centroid)
    @property
    def moments_rho4(self): return self._moments_rho4
    @property
    def moments_n_iter(self): return self._moments_n_iter
    @property
    def correction_status(self): return self._correction_status
    @property
    def corrected_e1(self): return self._corrected_e1
    @property
    def corrected_e2(self): return self._corrected_e2
    @property
    def corrected_g1(self): return self._corrected_g1
    @property
    def corrected_g2(self): return self._corrected_g2
    @property
    def meas_type(self): return self._meas_type
    @property
    def corrected_shape_err(self): return self._corrected_shape_err
    @property
    def correction_method(self): return self._correction_method
    @property
    def resolution_factor(self): return self._resolution_factor
    @property
    def psf_sigma(self): return self._psf_sigma

    @property
    def psf_shape(self):
        return Shear(e1=self._psf_e1, e2=self._psf_e2)

    @property
    def error_message(self):
        # We use "None" in C++ ShapeData to indicate no error messages to avoid problems on
        # (some) Macs using zero-length strings.  Here, we revert that back to "".
        if self._error_message == "None":
            return ""
        else:
            return self._error_message

    def __repr__(self):
        s = 'galsim.hsm.ShapeData('
        if self.image_bounds.isDefined(): s += 'image_bounds=%r, '%self.image_bounds
        if self.moments_status != -1: s += 'moments_status=%r, '%self.moments_status
        # Always include this one:
        s += 'observed_shape=%r'%self.observed_shape
        if self.moments_sigma != -1: s += ', moments_sigma=%r'%self.moments_sigma
        if self.moments_amp != -1: s += ', moments_amp=%r'%self.moments_amp
        if self.moments_centroid != PositionD():
            s += ', moments_centroid=%r'%self.moments_centroid
        if self.moments_rho4 != -1: s += ', moments_rho4=%r'%self.moments_rho4
        if self.moments_n_iter != 0: s += ', moments_n_iter=%r'%self.moments_n_iter
        if self.correction_status != -1: s += ', correction_status=%r'%self.correction_status
        if self.corrected_e1 != -10.: s += ', corrected_e1=%r'%self.corrected_e1
        if self.corrected_e2 != -10.: s += ', corrected_e2=%r'%self.corrected_e2
        if self.corrected_g1 != -10.: s += ', corrected_g1=%r'%self.corrected_g1
        if self.corrected_g2 != -10.: s += ', corrected_g2=%r'%self.corrected_g2
        if self.meas_type != 'None': s += ', meas_type=%r'%self.meas_type
        if self.corrected_shape_err != -1.:
            s += ', corrected_shape_err=%r'%self.corrected_shape_err
        if self.correction_method != 'None': s += ', correction_method=%r'%self.correction_method
        if self.resolution_factor != -1.: s += ', resolution_factor=%r'%self.resolution_factor
        if self.psf_sigma != -1.: s += ', psf_sigma=%r'%self.psf_sigma
        if self.psf_shape != Shear(): s += ', psf_shape=%r'%self.psf_shape
        if self.error_message != "": s += ', error_message=%r'%self.error_message
        s += ')'
        return s

    def __eq__(self, other):
        return (self is other or
                (isinstance(other,ShapeData) and self._getinitargs() == other._getinitargs()))
    def __ne__(self, other): return not self.__eq__(other)
    def __hash__(self): return hash(("galsim.hsm.ShapeData", self._getinitargs()))

    def _getinitargs(self):
        return (self.image_bounds, self.moments_status, self.observed_shape,
                self.moments_sigma, self.moments_amp, self.moments_centroid, self.moments_rho4,
                self.moments_n_iter, self.correction_status, self.corrected_e1, self.corrected_e2,
                self.corrected_g1, self.corrected_g2, self.meas_type, self.corrected_shape_err,
                self.correction_method, self.resolution_factor, self.psf_sigma,
                self.psf_shape, self.error_message)

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
            shape = Shear(g1 = -shape.g1, g2 = shape.g2)
        # Next the rotation
        shape = Shear(g = shape.g, beta = shape.beta + theta)
        # Finally the shear
        observed_shape = shear + shape

        # Fix moments_centroid
        moments_centroid = jac.toWorld(self.moments_centroid) - jac.toWorld(image_pos)

        return ShapeData(image_bounds=self.image_bounds,
                         moments_status=self.moments_status,
                         observed_shape=observed_shape,
                         moments_sigma=moments_sigma,
                         moments_amp=self.moments_amp,
                         moments_centroid=moments_centroid,
                         moments_rho4=self.moments_rho4,
                         moments_n_iter=self.moments_n_iter,
                         error_message=self.error_message)
                         # The other values are reset to the defaults, since they are
                         # results from EstimateShear.


@implements(_galsim.hsm.HSMParams)
# @dataclass(repr=False)
class HSMParams:
    nsig_rg: float = 3.0 
    nsig_rg2: float = 3.6  
    regauss_too_small: int = 1 
    adapt_order: int = 2
    convergence_threshold: float = 1.e-6 
    max_mom2_iter: int = 400
    num_iter_default: int = -1
    bound_correct_wt: float = 0.25
    max_amoment: float = 8000.
    max_ashift: float = 15.
    ksb_moments_max: int = 4
    ksb_sig_weight: float = 0.0
    ksb_sig_factor: float = 1.0
    failed_moments: float = -1000

    def _getinitargs(self):
        # TODO: For now, leave 3rd param as unused max_moment_nsig2.
        #       Remove it at version 3.0 to avoid changing C++ API yet.
        return (self.nsig_rg, self.nsig_rg2, 0., self.regauss_too_small,
                self.adapt_order, self.convergence_threshold, self.max_mom2_iter,
                self.num_iter_default, self.bound_correct_wt, self.max_amoment, self.max_ashift,
                self.ksb_moments_max, self.ksb_sig_weight, self.ksb_sig_factor,
                self.failed_moments)

    @property
    def nsig_rg(self): return self._nsig_rg
    @property
    def nsig_rg2(self): return self._nsig_rg2
    @property
    def regauss_too_small(self): return self._regauss_too_small
    @property
    def adapt_order(self): return self._adapt_order
    @property
    def convergence_threshold(self): return self._convergence_threshold
    @property
    def max_mom2_iter(self): return self._max_mom2_iter
    @property
    def num_iter_default(self): return self._num_iter_default
    @property
    def bound_correct_wt(self): return self._bound_correct_wt
    @property
    def max_amoment(self): return self._max_amoment
    @property
    def max_ashift(self): return self._max_ashift
    @property
    def ksb_moments_max(self): return self._ksb_moments_max
    @property
    def ksb_sig_weight(self): return self._ksb_sig_weight
    @property
    def ksb_sig_factor(self): return self._ksb_sig_factor
    @property
    def failed_moments(self): return self._failed_moments

    @staticmethod
    def check(hsmparams, default=None):
        """Checks that hsmparams is either a valid HSMParams instance or None.

        In the former case, it returns hsmparams, in the latter it returns default
        (HSMParams.default if no other default specified).
        """
        if hsmparams is None:
            return default if default is not None else HSMParams.default
        elif not isinstance(hsmparams, HSMParams):
            raise TypeError("Invalid HSMParams: %s"%hsmparams)
        else:
            return hsmparams

    def __repr__(self):
        return ('galsim.hsm.HSMParams(' + 14*'%r,' + '%r)')%self._getinitargs()

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, HSMParams) and self._getinitargs() == other._getinitargs()))
    def __ne__(self, other):
        return not self.__eq__(other)
    def __hash__(self):
        return hash(('galsim.hsm.HSMParams', self._getinitargs()))

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['_hsmp']
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
                weight=weight, image=image)
                # also make sure there are no negative values

        if jnp.any(weight.array < 0):
            raise GalSimValueError("Weight image cannot contain negative values.", weight)

    if badpix is not None and badpix.bounds != image.bounds:
        raise GalSimIncompatibleValuesError(
            "Badpix image does not have the same bounds as the input Image.",
            badpix=badpix, image=image)


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
        if weight.dtype == jnp.int32:
            if not badpix:
                mask = weight
            else:
                # If we need to mask bad pixels, we'll need a copy anyway.
                mask = ImageI(weight)

        # otherwise, we need to convert it to the right type
        else:
            mask = ImageI(bounds=image.bounds, init_value=0)
            mask.array[weight.array > 0.] = 1

    # if badpix image was supplied, identify the nonzero (bad) pixels and set them to zero in weight
    # image; also check bounds
    if badpix is not None:
        mask.array[badpix.array != 0] = 0

    # if no pixels are used, raise an exception
    if not jnp.any(mask.array):
        raise GalSimHSMError("No pixels are being used!")

    # finally, return the Image for the weight map
    return mask


# A simpler helper function to force images to be of type ImageF or ImageD
def _convertImage(image):
    # Convert the given image to the correct format needed to pass to the C++ layer.
    # This is used by EstimateShear() and FindAdaptiveMom().

    # if weight is not of type float/double, convert to float/double
    if (image.dtype == jnp.int16 or image.dtype == jnp.uint16):
        image = ImageF(image)
    elif (image.dtype == jnp.int32 or image.dtype == jnp.uint32):
        image = ImageD(image)

    return image

@implements(_galsim.hsm.EstimateShear)
def EstimateShear(gal_image, PSF_image, weight=None, badpix=None, sky_var=0.0,
                  shear_est="REGAUSS", recompute_flux="FIT", guess_sig_gal=5.0,
                  guess_sig_PSF=3.0, precision=1.0e-6, guess_centroid=None,
                  strict=True, check=True, hsmparams=None):
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
        EstimateShearView(result._data,
                                  gal_image._image, PSF_image._image, weight._image,
                                  float(sky_var), shear_est.upper(), recompute_flux.upper(),
                                  float(guess_sig_gal), float(guess_sig_PSF), float(precision),
                                  guess_centroid._p, hsmparams._hsmp)
        return result
    except RuntimeError as err:
        if (strict == True):
            raise GalSimHSMError(str(err)) from None
        else:
            return ShapeData(error_message = str(err))

@implements(_galsim.hsm.FindAdaptiveMom)
def FindAdaptiveMom(object_image, weight=None, badpix=None, guess_sig=5.0, precision=1.0e-6,
                    guess_centroid=None, strict=True, check=True, round_moments=False, hsmparams=None,
                    use_sky_coords=False):
    """Measure adaptive moments of an object.

    This method estimates the best-fit elliptical Gaussian to the object (see Hirata & Seljak 2003
    for more discussion of adaptive moments).  This elliptical Gaussian is computed iteratively
    by initially guessing a circular Gaussian that is used as a weight function, computing the
    weighted moments, recomputing the moments using the result of the previous step as the weight
    function, and so on until the moments that are measured are the same as those used for the
    weight function.  `FindAdaptiveMom` can be used either as a free function, or as a method of the
    `Image` class.

    By default, this routine computes moments in pixel coordinates, which generally use (x,y)
    for the coordinate variables, so the underlying second moments are Ixx, Iyy, and Ixy.
    If the WCS is (at least approximately) just a `PixelScale`, then this scale can be applied to
    convert the moments' units from pixels to arcsec.  The derived shapes are unaffected by
    the pixel scale.

    However, there is also an option to apply a non-trivial WCS, which may potentially rotate
    and/or shear the (x,y) moments to the local sky coordinates, which generally use (u,v)
    for the coordinate variables. These coordinates are measured in arcsec and are oriented
    such that +v is towards North and +u is towards West. In this case, the returned values are
    all in arcsec, and are based instead on Iuu, Ivv, and Iuv.  To enable this feature, use
    ``use_sky_coords=True``.  See also the method `ShapeData.applyWCS` for more details.

    .. note::

        The application of the WCS implicitly assumes that the WCS is locally uniform across the
        size of the object being measured.  This is normally a very good approximation for most
        applications of interest.

    Like `EstimateShear`, `FindAdaptiveMom` works on `Image` inputs, and fails if the object is
    small compared to the pixel scale.  For more details, see `EstimateShear`.

    Example::

        >>> my_gaussian = galsim.Gaussian(flux=1.0, sigma=1.0)
        >>> my_gaussian_image = my_gaussian.drawImage(scale=0.2, method='no_pixel')
        >>> my_moments = galsim.hsm.FindAdaptiveMom(my_gaussian_image)

    or::

        >>> my_moments = my_gaussian_image.FindAdaptiveMom()

    Assuming a successful measurement, the most relevant pieces of information are
    ``my_moments.moments_sigma``, which is ``|det(M)|^(1/4)`` (= ``sigma`` for a circular Gaussian)
    and ``my_moments.observed_shape``, which is a `Shear`.  In this case,
    ``my_moments.moments_sigma`` is precisely 5.0 (in units of pixels), and
    ``my_moments.observed_shape`` is consistent with zero.

    Methods of the `Shear` class can be used to get the distortion ``e``, the shear ``g``, the
    conformal shear ``eta``, and so on.

    As an example of how to use the optional ``hsmparams`` argument, consider cases where the input
    images have unusual properties, such as being very large.  This could occur when measuring the
    properties of a very over-sampled image such as that generated using::

        >>> my_gaussian = galsim.Gaussian(sigma=5.0)
        >>> my_gaussian_image = my_gaussian.drawImage(scale=0.01, method='no_pixel')

    If the user attempts to measure the moments of this very large image using the standard syntax,
    ::

        >>> my_moments = my_gaussian_image.FindAdaptiveMom()

    then the result will be a ``GalSimHSMError`` due to moment measurement failing because the
    object is so large.  While the list of all possible settings that can be changed is accessible
    in the docstring of the `HSMParams` class, in this case we need to modify ``max_amoment`` which
    is the maximum value of the moments in units of pixel^2.  The following measurement, using the
    default values for every parameter except for ``max_amoment``, will be
    successful::

        >>> new_params = galsim.hsm.HSMParams(max_amoment=5.0e5)
        >>> my_moments = my_gaussian_image.FindAdaptiveMom(hsmparams=new_params)

    Parameters:
        object_image:       The `Image` for the object being measured.
        weight:             The optional weight image for the object being measured.  Can be an int
                            or a float array.  Currently, GalSim does not account for the variation
                            in non-zero weights, i.e., a weight map is converted to an image with 0
                            and 1 for pixels that are not and are used.  Full use of spatial
                            variation in non-zero weights will be included in a future version of
                            the code. [default: None]
        badpix:             The optional bad pixel mask for the image being used.  Zero should be
                            used for pixels that are good, and any nonzero value indicates a bad
                            pixel. [default: None]
        guess_sig:          Optional argument with an initial guess for the Gaussian sigma of the
                            object (in pixels). [default: 5.0]
        precision:          The convergence criterion for the moments. [default: 1e-6]
        guess_centroid:     An initial guess for the object centroid (useful in case it is not
                            located at the center, which is used if this keyword is not set).  The
                            convention for centroids is such that the center of the lower-left pixel
                            is (image.xmin, image.ymin).
                            [default: object_image.true_center]
        strict:             Whether to require success. If ``strict=True``, then there will be a
                            ``GalSimHSMError`` exception if shear estimation fails.  If set to
                            ``False``, then information about failures will be silently stored in
                            the output ShapeData object. [default: True]
        check:              Check if the object_image, weight and badpix are in the correct format and valid.
                            [default: True]
        round_moments:      Use a circular weight function instead of elliptical.
                            [default: False]
        hsmparams:          The hsmparams keyword can be used to change the settings used by
                            FindAdaptiveMom when estimating moments; see `HSMParams` documentation
                            for more information. [default: None]
        use_sky_coords:     Whether to convert the measured moments to sky_coordinates.
                            Setting this to true is equivalent to running
                            ``applyWCS(object_image.wcs, image_pos=object_image.true_center)``
                            on the result.  [default: False]

    Returns:
        a `ShapeData` object containing the results of moment measurement.
    """
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
        FindAdaptiveMomView(result._data,
                            object_image._image, weight._image,
                            float(guess_sig), float(precision), guess_centroid._p,
                            bool(round_moments), hsmparams._hsmp)

        if use_sky_coords:
            result = result.applyWCS(object_image.wcs, image_pos=object_image.true_center)
        return result
    except RuntimeError as err:
        if (strict == True):
            raise GalSimHSMError(str(err)) from None
        else:
            return ShapeData(error_message = str(err))

# make FindAdaptiveMom a method of Image class
Image.FindAdaptiveMom = FindAdaptiveMom

def nonZeroBounds(image):
    pass

def MakeMaskedImage(image, mask):
    b1 = image.nonZeroBounds()
    b2 = mask.noneZeroBounds()
    b = b1 & b2
    
    masked_image = image[b] * mask[b]

    # return masked_image
    return image

def FindAdaptativeMomView(results, 
                          object_image, 
                          object_mask_image, 
                          guess_sig, 
                          precision, 
                          guess_centroid, 
                          round_moments, 
                          hsmparams
                          ):
    
    tc = object_image.getBounds().trueCenter()
    results.moments_centroid = jnp.where(guess_centroid!=-1000.0,
                                         guess_centroid,
                                         tc)
    
    m_xx = guess_sig*guess_sig
    m_yy = m_xx
    m_xy = 0.

    # Apply the mask
    masked_object_image = MakeMaskedImage(object_image, object_mask_image)

    results.image_bounds = object_image.bounds

    # TODO: find_ellipmom_2
    # TODO: find_ellipmom_1

    # def find_ellipmom_1(data, x0, y0, Mxx, Mxy, Myy, A, Bx, By, Cxx, Cxy, Cyy, rho4w, hsmparams):
    #     xmin = data.XMin()
    #     xmax = data.XMax()
    #     ymin = data.YMin()
    #     ymax = data.YMax()
