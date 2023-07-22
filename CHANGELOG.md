# Change log

## JAX-GalSim 0.0.1 (Unreleased)

* Changes
  * Added simplified versions of basic objects, with no support of Chromatic objects yet:
    * `GSObjects` and `gsparams`
    * `Sum`
    * `Position`, `PositionD` and `PositionI`
    * `Transformation`
    * `Shear`
  * Added implementation of simple light profiles:
    * `Gaussian`, `Exponential`, `Pixel`, `Box` 
  * Added implementation of simple WCS:
    * `PixelScale`, `OffsetWCS`, `JacobianWCS`, `AffineTransform`, `ShearWCS`, `OffsetShearWCS`
  * Added automated suite of tests against reference GalSim
