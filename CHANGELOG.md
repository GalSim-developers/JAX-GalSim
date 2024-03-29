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
    * `InterpolatedImage` and `Interpolant`
    * `PhotonArray`
    * `Sensor`
    * `AngleUnit`, `Angle`, and `CelestialCoord`
    * `BaseDeviate` and child classes
    * `BaseNoise` and child classes
  * Added implementation of fundamental operations:
    * `drawImage`
    * `drawReal`
    * `drawFFT`
    * `drawKImage`
    * `makePhot`
    * `drawPhot`
  * Added implementation of simple light profiles:
    * `Gaussian`, `Exponential`, `Pixel`, `Box`, `Moffat`, `Spergel`, `DeltaFunction`
  * Added implementation of simple WCS:
    * `PixelScale`, `OffsetWCS`, `JacobianWCS`, `AffineTransform`, `ShearWCS`, `OffsetShearWCS`, `GSFitsWCS`, `FitsWCS`, `TanWCS`
  * Added automated suite of tests using the reference GalSim and LSSTDESC-Coord test suites
  * Added support for the `galsim.fits` module
  * Added a `from_galsim` method to convert from GalSim objects to JAX-GalSim objects

* Caveats
  * Real space convolution are not yet implemented in `drawImage``.
