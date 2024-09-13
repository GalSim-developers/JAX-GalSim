# JAX-GalSim

**JAX port of GalSim, for parallelized, GPU accelerated, and differentiable galaxy image simulations.**

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md) [![Python package](https://github.com/GalSim-developers/JAX-GalSim/actions/workflows/python_package.yaml/badge.svg)](https://github.com/GalSim-developers/JAX-GalSim/actions/workflows/python_package.yaml) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/GalSim-developers/JAX-GalSim/main.svg)](https://results.pre-commit.ci/latest/github/GalSim-developers/JAX-GalSim/main) [![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/GalSim-developers/JAX-GalSim)

**Disclaimer**: This project is still in an early development phase, **please use the [reference GalSim implementation](https://github.com/GalSim-developers/GalSim) for any scientific applications.**

## Objective and Design

The goal of this library is to reimplement GalSim functionalities in pure JAX to allow for automatic differentiation, GPU acceleration, and batched computations.

### Guiding Principles

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
JAX-GalSim has implemented 22.5% of the GalSim API. See the list below for the supported APIs.

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
- galsim.bessel.j0
- galsim.bessel.kv
- galsim.bessel.si
- galsim.fits.closeHDUList
- galsim.fits.readCube
- galsim.fits.readFile
- galsim.fits.readMulti
- galsim.fits.write
- galsim.fits.writeFile
- galsim.fitswcs.CelestialWCS
- galsim.integ.int1d
- galsim.noise.addNoise
- galsim.noise.addNoiseSNR
- galsim.random.permute
- galsim.utilities.g1g2_to_e1e2
- galsim.utilities.horner
- galsim.utilities.printoptions
- galsim.utilities.unweighted_moments
- galsim.utilities.unweighted_shape
- galsim.wcs.EuclideanWCS
- galsim.wcs.LocalWCS
- galsim.wcs.UniformWCS

</details>
<!-- end-api-coverage -->

_**Note**: The coverage list is generated automatically by the `scripts/update_api_coverage.py` script. To update it, run `python scripts/update_api_coverage.py` from the root of the repository._
