# Change log

## JAX-GalSim 0.0.1 (Unreleased)

* Changes
  * Added simplified versions of basic objects, with no support of Chromatic objects yet:
    * `GSObjects` and `gsparams`
    * `Sum`
    * `Position`, `PositionD` and `PositionI`
    * `Transformation`
    * `Shear`
    * `Convolve`
  * Added implementation of fundamental operations:
    * `drawImage`
    * `drawReal`
    * `drawFFT`
    * `drawKImage`
  * Added implementation of simple light profiles:
    * `Gaussian`, `Exponential`, `Pixel`, `Box`, `Moffat`
  * Added implementation of simple WCS:
    * `PixelScale`, `OffsetWCS`, `JacobianWCS`, `AffineTransform`, `ShearWCS`, `OffsetShearWCS`
  * Added automated suite of tests against reference GalSim
  * Added support for the `galsim.fits` module
  * Added a `from_galsim` method to convert from GalSim objects to JAX-GalSim objects

* Caveats
  * Currently the FFT convolution does not perform kwrapping,
  so it will lead to erroneous results on underesolved images.
  * Real space convolution and photon shooting methods are not
  yet implemented in drawImage.
