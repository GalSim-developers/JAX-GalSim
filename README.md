# JAX-GalSim

**JAX port of GalSim, for parallelized, GPU accelerated, and differentiable galaxy image simulations.**

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md) [![Python package](https://github.com/GalSim-developers/JAX-GalSim/actions/workflows/python_package.yaml/badge.svg)](https://github.com/GalSim-developers/JAX-GalSim/actions/workflows/python_package.yaml) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/GalSim-developers/JAX-GalSim/main.svg)](https://results.pre-commit.ci/latest/github/GalSim-developers/JAX-GalSim/main) [![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/GalSim-developers/JAX-GalSim)

**Disclaimer**: This project is still in an early development phase, **please use the [reference GalSim implementation](https://github.com/GalSim-developers/GalSim) for any scientific applications.**

## Objective and Design

The goal of this library is to reimplement GalSim functionalities in pure JAX to allow for automatic differentiation, GPU acceleration, and batched computations.

### Guiding Principles**

- Strive to be a drop-in replacement for GalSim, i.e. provide a close match to the GalSim API.
- Each function/feature will be tested against the reference GalSim implementation.
- This package will aim to be a **subset** of GalSim (i.e. only contains functions with a reference GalSim implementation).
- Implementations should be easy to read and understand.
- Code should be pip-installable on any machine, no compilation required.
- Any notable differences between the JAX and reference implementations will be clearly documented.

### Notable Differences

- JAX arrays are immutable. Thus, in-place operations on images are not possible and a new image is
  returned instead. Also, the RNG classes cannot fill arrays and instead return new arrays.
- JAX arrays do not support all the kinds of views that numpy arrays support. This means that some
  operations that work in GalSim may not work in JAX-GalSim (e.g., real views of complex images).
- JAX-GalSim uses a different random number generator than GalSim. This leads to different results in terms of both the
  generated random numbers and in terms of which RNGs have stable discarding.

## Contributing

Everyone can contribute to this project, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) document for details.

In short, to interact with the project you can:

- Ask or Answer questions on the [Discussions Q&A](https://github.com/GalSim-developers/JAX-GalSim/discussions/categories/q-a) page
- Report a bug by opening a [GitHub issue](https://github.com/GalSim-developers/JAX-GalSim/issues)
- Open a [GitHub issue](https://github.com/GalSim-developers/JAX-GalSim/issues) or [Discussions](https://github.com/GalSim-developers/JAX-GalSim/discussions) to ask for feedback on a planned contribution.
- Submit a Pull Request to contribute to the code.

Issues marked with _contributions welcome_ or _good first issue_ are particularly good places to start. These are great ways to learn more
about the inner workings of GalSim and how to code in JAX.

## Current GalSim API Coverage

<!-- start-api-coverage -->
JAX-GalSim has implemented 16.7% of the GalSim API. See the list below for the supported APIs.

<details>

- galsim.Add
- galsim.AffineTransform
- galsim.Angle
- galsim.AngleUnit
- galsim.BaseDeviate
- galsim.BaseNoise
- galsim.BaseWCS
- galsim.BinomialDeviate
- galsim.Bounds
- galsim.BoundsD
- galsim.BoundsI
- galsim.Box
- galsim.CCDNoise
- galsim.CelestialCoord
- galsim.Chi2Deviate
- galsim.Convolution
- galsim.Convolve
- galsim.Cubic
- galsim.Deconvolution
- galsim.Deconvolve
- galsim.Delta
- galsim.DeltaFunction
- galsim.DeviateNoise
- galsim.Exponential
- galsim.FitsHeader
- galsim.FitsWCS
- galsim.GSFitsWCS
- galsim.GSObject
- galsim.GSParams
- galsim.GalSimBoundsError
- galsim.GalSimConfigError
- galsim.GalSimConfigValueError
- galsim.GalSimDeprecationWarning
- galsim.GalSimError
- galsim.GalSimFFTSizeError
- galsim.GalSimHSMError
- galsim.GalSimImmutableError
- galsim.GalSimIncompatibleValuesError
- galsim.GalSimIndexError
- galsim.GalSimKeyError
- galsim.GalSimNotImplementedError
- galsim.GalSimRangeError
- galsim.GalSimSEDError
- galsim.GalSimUndefinedBoundsError
- galsim.GalSimValueError
- galsim.GalSimWarning
- galsim.GammaDeviate
- galsim.Gaussian
- galsim.GaussianDeviate
- galsim.GaussianNoise
- galsim.Image
- galsim.ImageCD
- galsim.ImageCF
- galsim.ImageD
- galsim.ImageF
- galsim.ImageI
- galsim.ImageS
- galsim.ImageUI
- galsim.ImageUS
- galsim.Interpolant
- galsim.InterpolatedImage
- galsim.JacobianWCS
- galsim.Lanczos
- galsim.Linear
- galsim.Moffat
- galsim.Nearest
- galsim.OffsetShearWCS
- galsim.OffsetWCS
- galsim.PhotonArray
- galsim.Pixel
- galsim.PixelScale
- galsim.PoissonDeviate
- galsim.PoissonNoise
- galsim.Position
- galsim.PositionD
- galsim.PositionI
- galsim.Quintic
- galsim.Sensor
- galsim.Shear
- galsim.ShearWCS
- galsim.SincInterpolant
- galsim.Spergel
- galsim.Sum
- galsim.TanWCS
- galsim.Transform
- galsim.Transformation
- galsim.UniformDeviate
- galsim.VariableGaussianNoise
- galsim.WeibullDeviate
- galsim.angle
- galsim.angle.Angle
- galsim.angle.AngleUnit
- galsim.bessel
- galsim.bounds
- galsim.bounds.Bounds
- galsim.bounds.BoundsD
- galsim.bounds.BoundsI
- galsim.bounds.Position
- galsim.bounds.PositionD
- galsim.bounds.PositionI
- galsim.box
- galsim.box.Box
- galsim.box.GSObject
- galsim.box.Pixel
- galsim.celestial
- galsim.celestial.CelestialCoord
- galsim.convolve
- galsim.convolve.Convolution
- galsim.convolve.Convolve
- galsim.convolve.Deconvolution
- galsim.convolve.Deconvolve
- galsim.convolve.GSObject
- galsim.convolve.GSParams
- galsim.convolve.PhotonArray
- galsim.convolve.galsim_warn
- galsim.deltafunction
- galsim.deltafunction.DeltaFunction
- galsim.deltafunction.GSObject
- galsim.errors
- galsim.errors.GalSimBoundsError
- galsim.errors.GalSimConfigError
- galsim.errors.GalSimConfigValueError
- galsim.errors.GalSimDeprecationWarning
- galsim.errors.GalSimError
- galsim.errors.GalSimFFTSizeError
- galsim.errors.GalSimHSMError
- galsim.errors.GalSimImmutableError
- galsim.errors.GalSimIncompatibleValuesError
- galsim.errors.GalSimIndexError
- galsim.errors.GalSimKeyError
- galsim.errors.GalSimNotImplementedError
- galsim.errors.GalSimRangeError
- galsim.errors.GalSimSEDError
- galsim.errors.GalSimUndefinedBoundsError
- galsim.errors.GalSimValueError
- galsim.errors.GalSimWarning
- galsim.errors.galsim_warn
- galsim.exponential
- galsim.exponential.Exponential
- galsim.exponential.GSObject
- galsim.exponential.lazy_property
- galsim.fits
- galsim.fits.FitsHeader
- galsim.fits.Image
- galsim.fits.closeHDUList
- galsim.fits.galsim_warn
- galsim.fits.read
- galsim.fits.readCube
- galsim.fits.readFile
- galsim.fits.readMulti
- galsim.fits.write
- galsim.fits.writeCube
- galsim.fits.writeFile
- galsim.fits.writeMulti
- galsim.fitswcs
- galsim.fitswcs.AffineTransform
- galsim.fitswcs.AngleUnit
- galsim.fitswcs.CelestialCoord
- galsim.fitswcs.CelestialWCS
- galsim.fitswcs.FitsWCS
- galsim.fitswcs.GSFitsWCS
- galsim.fitswcs.GalSimError
- galsim.fitswcs.GalSimIncompatibleValuesError
- galsim.fitswcs.GalSimNotImplementedError
- galsim.fitswcs.GalSimValueError
- galsim.fitswcs.JacobianWCS
- galsim.fitswcs.OffsetWCS
- galsim.fitswcs.PixelScale
- galsim.fitswcs.PositionD
- galsim.fitswcs.TanWCS
- galsim.fitswcs.fits
- galsim.fitswcs.galsim_warn
- galsim.fitswcs.horner2d
- galsim.gaussian
- galsim.gaussian.GSObject
- galsim.gaussian.Gaussian
- galsim.gsobject
- galsim.gsobject.BaseDeviate
- galsim.gsobject.GSObject
- galsim.gsobject.GSParams
- galsim.gsobject.GalSimError
- galsim.gsobject.GalSimIncompatibleValuesError
- galsim.gsobject.GalSimNotImplementedError
- galsim.gsobject.GalSimValueError
- galsim.gsobject.Position
- galsim.gsobject.Sensor
- galsim.gsobject.galsim_warn
- galsim.gsobject.pa
- galsim.gsobject.parse_pos_args
- galsim.gsparams
- galsim.gsparams.GSParams
- galsim.image
- galsim.image.BaseWCS
- galsim.image.BoundsD
- galsim.image.BoundsI
- galsim.image.GalSimImmutableError
- galsim.image.Image
- galsim.image.ImageCD
- galsim.image.ImageCF
- galsim.image.ImageD
- galsim.image.ImageF
- galsim.image.ImageI
- galsim.image.ImageS
- galsim.image.ImageUI
- galsim.image.ImageUS
- galsim.image.PixelScale
- galsim.image.PositionI
- galsim.image.parse_pos_args
- galsim.interpolant
- galsim.interpolant.Cubic
- galsim.interpolant.Delta
- galsim.interpolant.GSParams
- galsim.interpolant.GalSimValueError
- galsim.interpolant.Interpolant
- galsim.interpolant.Lanczos
- galsim.interpolant.Linear
- galsim.interpolant.Nearest
- galsim.interpolant.Quintic
- galsim.interpolant.SincInterpolant
- galsim.interpolant.lazy_property
- galsim.interpolatedimage
- galsim.interpolatedimage.BaseWCS
- galsim.interpolatedimage.GSObject
- galsim.interpolatedimage.GSParams
- galsim.interpolatedimage.GalSimIncompatibleValuesError
- galsim.interpolatedimage.GalSimRangeError
- galsim.interpolatedimage.GalSimUndefinedBoundsError
- galsim.interpolatedimage.GalSimValueError
- galsim.interpolatedimage.Image
- galsim.interpolatedimage.InterpolatedImage
- galsim.interpolatedimage.PixelScale
- galsim.interpolatedimage.PositionD
- galsim.interpolatedimage.Quintic
- galsim.interpolatedimage.convert_interpolant
- galsim.interpolatedimage.doc_inherit
- galsim.interpolatedimage.fits
- galsim.interpolatedimage.lazy_property
- galsim.moffat
- galsim.moffat.GSObject
- galsim.moffat.Moffat
- galsim.noise
- galsim.noise.BaseDeviate
- galsim.noise.BaseNoise
- galsim.noise.CCDNoise
- galsim.noise.DeviateNoise
- galsim.noise.GalSimError
- galsim.noise.GalSimIncompatibleValuesError
- galsim.noise.GaussianDeviate
- galsim.noise.GaussianNoise
- galsim.noise.Image
- galsim.noise.ImageD
- galsim.noise.PoissonDeviate
- galsim.noise.PoissonNoise
- galsim.noise.VariableGaussianNoise
- galsim.noise.addNoise
- galsim.noise.addNoiseSNR
- galsim.photon_array
- galsim.photon_array.BaseDeviate
- galsim.photon_array.GalSimIncompatibleValuesError
- galsim.photon_array.GalSimRangeError
- galsim.photon_array.GalSimUndefinedBoundsError
- galsim.photon_array.GalSimValueError
- galsim.photon_array.PhotonArray
- galsim.position
- galsim.position.Position
- galsim.position.PositionD
- galsim.position.PositionI
- galsim.random
- galsim.random.BaseDeviate
- galsim.random.BinomialDeviate
- galsim.random.Chi2Deviate
- galsim.random.GammaDeviate
- galsim.random.GaussianDeviate
- galsim.random.PoissonDeviate
- galsim.random.UniformDeviate
- galsim.random.WeibullDeviate
- galsim.random.permute
- galsim.sensor
- galsim.sensor.GalSimUndefinedBoundsError
- galsim.sensor.PositionI
- galsim.sensor.Sensor
- galsim.shear
- galsim.shear.Angle
- galsim.shear.GalSimIncompatibleValuesError
- galsim.shear.Shear
- galsim.spergel
- galsim.spergel.GSObject
- galsim.spergel.Spergel
- galsim.spergel.lazy_property
- galsim.sum
- galsim.sum.Add
- galsim.sum.GSObject
- galsim.sum.GSParams
- galsim.sum.Sum
- galsim.transform
- galsim.transform.GSObject
- galsim.transform.GSParams
- galsim.transform.PositionD
- galsim.transform.Transform
- galsim.transform.Transformation
- galsim.utilities
- galsim.utilities.GalSimIncompatibleValuesError
- galsim.utilities.GalSimValueError
- galsim.utilities.PositionD
- galsim.utilities.PositionI
- galsim.utilities.convert_interpolant
- galsim.utilities.g1g2_to_e1e2
- galsim.utilities.horner
- galsim.utilities.horner2d
- galsim.utilities.lazy_property
- galsim.utilities.parse_pos_args
- galsim.utilities.printoptions
- galsim.utilities.unweighted_moments
- galsim.utilities.unweighted_shape
- galsim.wcs
- galsim.wcs.AffineTransform
- galsim.wcs.AngleUnit
- galsim.wcs.BaseWCS
- galsim.wcs.CelestialCoord
- galsim.wcs.CelestialWCS
- galsim.wcs.EuclideanWCS
- galsim.wcs.GalSimValueError
- galsim.wcs.JacobianWCS
- galsim.wcs.LocalWCS
- galsim.wcs.OffsetShearWCS
- galsim.wcs.OffsetWCS
- galsim.wcs.PixelScale
- galsim.wcs.Position
- galsim.wcs.PositionD
- galsim.wcs.Shear
- galsim.wcs.ShearWCS
- galsim.wcs.UniformWCS
- galsim.wcs.compatible
- galsim.wcs.readFromFitsHeader

</details>
<!-- end-api-coverage -->

_**Note**: The coverage list is generated automatically by the `scripts/update_api_coverage.py` script. To update it, run `python scripts/update_api_coverage.py` from the root of the repository._
