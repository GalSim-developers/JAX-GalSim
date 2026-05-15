import os

os.environ["JAX_ENABLE_X64"] = "True"


from functools import partial

import galsim
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random, vmap
from jax.typing import ArrayLike
from surveycodex.utilities import mag2counts, mean_sky_level

import jax_galsim as jgs

PARAM_NAMES = [
    "flux_b",
    "flux_d",
    "hlr_b",
    "hlr_d",
    "q_d",
    "q_b",
    "beta",
]

DUMMY = 0.1  # not too small,

_DUMMY_PARAMS = {
    "flux_b": 0.0,
    "flux_d": 0.0,
    "hlr_b": DUMMY,
    "hlr_d": DUMMY,
    "q_d": 1.0,
    "q_b": 1.0,
    "beta": 0.0,
}


# TODO: consider adding padding to images


def get_default_lsst_background() -> float:
    return mean_sky_level("LSST", "i").to_value("electron").astype(np.float32).item()


def add_noise_galsim(rng_key, *, bg: float, galsim_image):
    # add noise
    seed = random.randint(rng_key, shape=(), minval=0, maxval=2**32 - 1).item()
    rng = galsim.BaseDeviate(seed)
    noise = galsim.GaussianNoise(rng=rng, sigma=np.sqrt(bg))
    galsim_image.addNoise(noise)  # background subtracted
    return galsim_image.array


def add_noise(rng_key, *, x: ArrayLike, bg: float, n: int = 1):
    """Produce `n` independent Gaussian noise realizations of a given image `x`.

    NOTE: This function assumes image is background-subtracted and dominated.
    """
    assert isinstance(bg, float) or bg.shape == ()
    x = x.reshape(1, *x.shape)
    x = x.repeat(n, axis=0)
    noise = random.normal(rng_key, shape=x.shape) * jnp.sqrt(bg)
    out = x + noise
    return out.squeeze(0)


def _format_column_to_dict(row):
    # ignore AGN component
    a_b = row["a_b"].item()
    a_d = row["a_d"].item()
    b_b = row["b_b"].item()
    b_d = row["b_d"].item()
    i_ab = row["i_ab"].item()

    fluxnorm_bulge = row["fluxnorm_bulge"].item()
    fluxnorm_disk = row["fluxnorm_disk"].item()
    total_fluxnorm = fluxnorm_disk + fluxnorm_bulge

    assert fluxnorm_bulge > 0 or fluxnorm_disk > 0

    pa_bulge = row["pa_bulge"].item()
    pa_disk = row["pa_disk"].item()

    theta = pa_bulge if fluxnorm_bulge > 0 else pa_disk  # degrees

    # sanity
    if fluxnorm_bulge > 0 and fluxnorm_disk > 0:
        assert pa_bulge == pa_disk

    # get flux
    flux_tot = mag2counts(i_ab, survey="LSST", filter="i").to_value("electron")
    flux_b = flux_tot * fluxnorm_bulge / total_fluxnorm
    flux_d = flux_tot * fluxnorm_disk / total_fluxnorm

    # TODO: do flux 0 gsobjects change anything in the final image?
    # dummy values to play nicely with vmap
    return {
        "flux_b": flux_b,
        "flux_d": flux_d,
        "hlr_b": np.sqrt(a_b * b_b) if fluxnorm_bulge > 0 else DUMMY,
        "hlr_d": np.sqrt(a_d * b_d) if fluxnorm_disk > 0 else DUMMY,
        "q_d": b_d / a_d if fluxnorm_disk > 0 else DUMMY,
        "q_b": b_b / a_b if fluxnorm_bulge > 0 else DUMMY,  # dummy values
        "beta": theta,
    }


# use jax random here, which I think will make reproducibility easier
def sample_cat(key, *, n_sources: int, max_n_sources: int, cat):
    indices = random.choice(key, jnp.arange(len(cat)), shape=(n_sources,), replace=True)
    indices_np = np.array(indices)
    rows = cat[indices_np]

    sample_params = []
    for row in rows:
        sample_params.append(_format_column_to_dict(row))

    all_params = {}
    for p in PARAM_NAMES:
        out = []
        for n in range(n_sources):
            out.append(sample_params[n][p])

        all_params[p] = np.array(out)

    # add dummy values so that shape of parameter arrays is always the same
    for p in PARAM_NAMES:
        for n in range(n_sources, max_n_sources):
            all_params[p] = np.append(all_params[p], _DUMMY_PARAMS[p])

    return all_params


def get_one_full_sample(
    key, *, cat, ilen: int, max_n_gals: int
) -> dict[str, np.ndarray]:
    density = len(cat) / (60 * 60) ** 2  # arcsec^2
    mean_sources = density * (ilen * 0.2) ** 2

    k, k1 = random.split(key)
    n_sources = random.poisson(k1, lam=mean_sources, shape=())

    k, k2 = random.split(k)
    k, k3 = random.split(k)
    x = np.array(random.uniform(k2, minval=0, maxval=ilen, shape=(max_n_gals,)))
    y = np.array(random.uniform(k3, minval=0, maxval=ilen, shape=(max_n_gals,)))

    x[n_sources:] = 0.0
    y[n_sources:] = 0.0

    # get galaxy properties
    _, k4 = random.split(k)
    galaxy_props = sample_cat(
        k4, n_sources=n_sources, max_n_sources=max_n_gals, cat=cat
    )
    all_props = {**galaxy_props, "x": x, "y": y}

    assert all_props["x"].shape[0] == max_n_gals
    assert all_props["flux_b"].shape[0] == max_n_gals

    return all_props, n_sources.item()


# drawing in vanilla GalSim first a la wl-shear-sims
def get_bd_galsim(
    flux_d, flux_b, hlr_b, hlr_d, q_b, q_d, beta, *, psf: galsim.GSObject
) -> galsim.GSObject:
    assert flux_d > 0 or flux_b > 0, "This object should not be in the catalog."

    components = []

    # disk
    if flux_d > 0.0:
        disk = galsim.Exponential(flux=flux_d, half_light_radius=hlr_d).shear(
            q=q_d, beta=beta * galsim.degrees
        )
        components.append(disk)

    # bulge
    if flux_b > 0.0:
        bulge = galsim.Spergel(nu=-0.6, flux=flux_b, half_light_radius=hlr_b).shear(
            q=q_b, beta=beta * galsim.degrees
        )
        components.append(bulge)

    galaxy = galsim.Add(components)
    gal_conv = galsim.Convolve([galaxy, psf])
    return gal_conv


def draw_galsim(
    galaxy_params: dict,
    n_sources: int,
    *,
    ilen: int,
    psf: galsim.GSObject,
    max_n_gals: int,
    slen: int | None = None,
    fft_size: int | None = None,
    max_slen: int | None = None,
):

    # create big image
    image = galsim.Image(ncol=ilen, nrow=ilen, scale=0.2, dtype=np.float64)
    wcs = image.wcs

    assert n_sources <= max_n_gals, (
        "Number of sources in sample exceeds maximum number of sources."
    )

    for n in range(n_sources):
        _gal_params = {k: v[n].item() for k, v in galaxy_params.items()}
        x = _gal_params.pop("x")
        y = _gal_params.pop("y")
        image_pos = galsim.PositionD(x=x, y=y)
        local_wcs = wcs.local(image_pos=image_pos)
        gal = get_bd_galsim(**_gal_params, psf=psf)
        if fft_size is not None:
            gal = gal.withGSParams(minimum_fft_size=fft_size, maximum_fft_size=fft_size)
        stamp = gal.drawImage(
            center=image_pos, wcs=local_wcs, dtype=image.dtype, nx=slen, ny=slen
        )

        if max_slen:
            assert max(stamp.array.shape) <= max_slen, (
                f"Stamp size {stamp.array.shape} exceeds maximum stamp size. Consider increasing max_slen."
            )

        b = stamp.bounds & image.bounds
        if b.isDefined():
            image[b] += stamp[b]

    return image.array


def _get_bd_jgs(
    flux_d,
    flux_b,
    hlr_b,
    hlr_d,
    q_b,
    q_d,
    beta,
    *,
    psf: jgs.GSObject,
):

    disk = jgs.Exponential(flux=flux_d, half_light_radius=hlr_d).shear(
        q=q_d, beta=beta * jgs.degrees
    )
    bulge = jgs.Spergel(nu=-0.6, flux=flux_b, half_light_radius=hlr_b).shear(
        q=q_b, beta=beta * jgs.degrees
    )
    galaxy = disk + bulge
    gal_conv = jgs.Convolve([galaxy, psf])
    return gal_conv


@partial(jit, static_argnames=("psf", "slen", "fft_size"))
def _draw_stamp_jgs(
    galaxy_params: dict,
    image_pos: jgs.PositionD,
    local_wcs: jgs.PixelScale,
    psf: jgs.GSObject,
    slen: int,
    fft_size: int,
) -> jax.Array:
    gsparams = jgs.GSParams(minimum_fft_size=fft_size, maximum_fft_size=fft_size)
    convolved_object = _get_bd_jgs(**galaxy_params, psf=psf).withGSParams(gsparams)
    stamp = convolved_object.drawImage(
        nx=slen, ny=slen, center=image_pos, wcs=local_wcs, dtype=jnp.float64
    )
    return stamp


@partial(jax.jit, static_argnames=("psf", "fft_size", "slen"))
def _draw_stamp_and_add_to_image(carry, x, psf, fft_size, slen):
    # scan already jits so a bit overkill
    image = carry[0]
    gparams, image_pos, lwcs = x
    stamp = _draw_stamp_jgs(
        galaxy_params=gparams,
        image_pos=image_pos,
        local_wcs=lwcs,
        psf=psf,
        slen=slen,
        fft_size=fft_size,
    )
    image[stamp.bounds] += stamp
    return (image,), None


@partial(jit, static_argnames=("psf", "ilen", "slen", "fft_size", "max_n_gals"))
def draw_jgs_scan_stamps(
    galaxy_params: dict,
    psf: jgs.GSObject,
    ilen: int,
    slen: int,
    fft_size: int,
    max_n_gals: int,
):
    # I think this version will be better in CPU than vmap

    # create big image
    image = jgs.ImageD(ncol=ilen, nrow=ilen, scale=0.2)
    wcs = image.wcs

    x = galaxy_params.pop("x")
    y = galaxy_params.pop("y")

    image_positions = vmap(lambda x, y: jgs.PositionD(x=x, y=y))(x, y)
    local_wcss = vmap(lambda x: wcs.local(image_pos=x))(image_positions)

    pad_image = jgs.ImageD(
        jnp.pad(image.array, slen), wcs=image.wcs, bounds=image.bounds.withBorder(slen)
    )

    final_pad_image = jax.lax.scan(
        partial(_draw_stamp_and_add_to_image, psf=psf, fft_size=fft_size, slen=slen),
        (pad_image,),
        xs=(galaxy_params, image_positions, local_wcss),
        length=max_n_gals,
    )[0][0]

    return final_pad_image.array[slen:-slen, slen:-slen]


@partial(jit)
def _add_to_image(carry, x):
    image = carry[0]
    stamp = x
    image[stamp.bounds] += stamp
    return (image,), None


@partial(jit, static_argnames=("psf", "ilen", "slen", "fft_size", "max_n_gals"))
def draw_jgs_vmap_stamps(
    galaxy_params: dict,
    psf: jgs.GSObject,
    ilen: int,
    slen: int,
    fft_size: int,
    max_n_gals: int,  # TODO: need to force this to always be maximum parameters used.
):

    # create big image
    image = jgs.ImageD(ncol=ilen, nrow=ilen, scale=0.2)
    wcs = image.wcs

    x = galaxy_params.pop("x")
    y = galaxy_params.pop("y")

    image_positions = jax.vmap(lambda x, y: jgs.PositionD(x=x, y=y))(x, y)
    local_wcss = jax.vmap(lambda x: wcs.local(image_pos=x))(image_positions)

    _draw_stamps_vmapped = vmap(
        partial(_draw_stamp_jgs, psf=psf, slen=slen, fft_size=fft_size)
    )
    stamps = _draw_stamps_vmapped(galaxy_params, image_positions, local_wcss)
    assert stamps.array.shape[0] == max_n_gals

    pad_image = jgs.ImageD(
        jnp.pad(image.array, slen), wcs=image.wcs, bounds=image.bounds.withBorder(slen)
    )

    final_pad_image = jax.lax.scan(
        _add_to_image,
        (pad_image,),
        xs=stamps,
        length=max_n_gals,
    )[0][0]

    return final_pad_image.array[slen:-slen, slen:-slen]
