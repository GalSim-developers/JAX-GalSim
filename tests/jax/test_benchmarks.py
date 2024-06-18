import jax

import jax_galsim as jgs


def test_benchmarks_interpolated_image_jit_compile(benchmark):
    gal = jgs.Gaussian(fwhm=1.2)
    im_gal = gal.drawImage(nx=32, ny=32, scale=0.2)
    igal = jgs.InterpolatedImage(
        im_gal, gsparams=jgs.GSParams(minimum_fft_size=128, maximum_fft_size=128)
    )

    def f():
        return igal.drawImage(nx=32, ny=32, scale=0.2)

    benchmark(lambda: jax.jit(f)().array.block_until_ready())


def test_benchmarks_interpolated_image_jit_run(benchmark):
    gal = jgs.Gaussian(fwhm=1.2)
    im_gal = gal.drawImage(nx=32, ny=32, scale=0.2)
    igal = jgs.InterpolatedImage(
        im_gal, gsparams=jgs.GSParams(minimum_fft_size=128, maximum_fft_size=128)
    )

    def f():
        return igal.drawImage(nx=32, ny=32, scale=0.2)

    jitf = jax.jit(f)

    # run once to compile
    jitf().array.block_until_ready()

    # now benchmark
    benchmark(lambda: jitf().array.block_until_ready())
