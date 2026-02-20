# API Coverage

JAX-GalSim has implemented **22.5%** of the GalSim API. The project focuses on
the most commonly used profiles and operations, with coverage expanding over time.

## Supported APIs

??? note "Click to expand the full list of implemented APIs"

    - `galsim.Add`
    - `galsim.AffineTransform`
    - `galsim.Angle`
    - `galsim.AngleUnit`
    - `galsim.BaseDeviate`
    - `galsim.BaseNoise`
    - `galsim.BaseWCS`
    - `galsim.BinomialDeviate`
    - `galsim.Bounds`
    - `galsim.BoundsD`
    - `galsim.BoundsI`
    - `galsim.Box`
    - `galsim.CCDNoise`
    - `galsim.CelestialCoord`
    - `galsim.Chi2Deviate`
    - `galsim.Convolution`
    - `galsim.Convolve`
    - `galsim.Cubic`
    - `galsim.Deconvolution`
    - `galsim.Deconvolve`
    - `galsim.Delta`
    - `galsim.DeltaFunction`
    - `galsim.DeviateNoise`
    - `galsim.Exponential`
    - `galsim.FitsHeader`
    - `galsim.FitsWCS`
    - `galsim.GSFitsWCS`
    - `galsim.GSObject`
    - `galsim.GSParams`
    - `galsim.GalSimBoundsError`
    - `galsim.GalSimConfigError`
    - `galsim.GalSimConfigValueError`
    - `galsim.GalSimDeprecationWarning`
    - `galsim.GalSimError`
    - `galsim.GalSimFFTSizeError`
    - `galsim.GalSimHSMError`
    - `galsim.GalSimImmutableError`
    - `galsim.GalSimIncompatibleValuesError`
    - `galsim.GalSimIndexError`
    - `galsim.GalSimKeyError`
    - `galsim.GalSimNotImplementedError`
    - `galsim.GalSimRangeError`
    - `galsim.GalSimSEDError`
    - `galsim.GalSimUndefinedBoundsError`
    - `galsim.GalSimValueError`
    - `galsim.GalSimWarning`
    - `galsim.GammaDeviate`
    - `galsim.Gaussian`
    - `galsim.GaussianDeviate`
    - `galsim.GaussianNoise`
    - `galsim.Image`
    - `galsim.ImageCD`
    - `galsim.ImageCF`
    - `galsim.ImageD`
    - `galsim.ImageF`
    - `galsim.ImageI`
    - `galsim.ImageS`
    - `galsim.ImageUI`
    - `galsim.ImageUS`
    - `galsim.Interpolant`
    - `galsim.InterpolatedImage`
    - `galsim.JacobianWCS`
    - `galsim.Lanczos`
    - `galsim.Linear`
    - `galsim.Moffat`
    - `galsim.Nearest`
    - `galsim.OffsetShearWCS`
    - `galsim.OffsetWCS`
    - `galsim.PhotonArray`
    - `galsim.Pixel`
    - `galsim.PixelScale`
    - `galsim.PoissonDeviate`
    - `galsim.PoissonNoise`
    - `galsim.Position`
    - `galsim.PositionD`
    - `galsim.PositionI`
    - `galsim.Quintic`
    - `galsim.Sensor`
    - `galsim.Shear`
    - `galsim.ShearWCS`
    - `galsim.SincInterpolant`
    - `galsim.Spergel`
    - `galsim.Sum`
    - `galsim.TanWCS`
    - `galsim.Transform`
    - `galsim.Transformation`
    - `galsim.UniformDeviate`
    - `galsim.VariableGaussianNoise`
    - `galsim.WeibullDeviate`
    - `galsim.bessel.j0`
    - `galsim.bessel.kv`
    - `galsim.bessel.si`
    - `galsim.fits.closeHDUList`
    - `galsim.fits.readCube`
    - `galsim.fits.readFile`
    - `galsim.fits.readMulti`
    - `galsim.fits.write`
    - `galsim.fits.writeFile`
    - `galsim.fitswcs.CelestialWCS`
    - `galsim.integ.int1d`
    - `galsim.noise.addNoise`
    - `galsim.noise.addNoiseSNR`
    - `galsim.random.permute`
    - `galsim.utilities.g1g2_to_e1e2`
    - `galsim.utilities.horner`
    - `galsim.utilities.printoptions`
    - `galsim.utilities.unweighted_moments`
    - `galsim.utilities.unweighted_shape`
    - `galsim.wcs.EuclideanWCS`
    - `galsim.wcs.LocalWCS`
    - `galsim.wcs.UniformWCS`

## Updating Coverage

```bash
python scripts/update_api_coverage.py
```

Compares GalSim's public API against `jax_galsim`'s implementations and updates the coverage percentage and list in `README.md`.
