import os

import galsim.errors

os.environ["JAX_ENABLE_X64"] = "True"


import math
from functools import partial
from pathlib import Path

import galsim
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from jax import random, vmap
from jax.lax import fori_loop
from jax.typing import ArrayLike
from surveycodex.utilities import mag2counts, mean_sky_level
from tqdm import tqdm

import jax_galsim as jgs

# be super careful with ffts
galsim.errors.raise_fft_size_error = True

PARAM_NAMES = [
    "flux_b",
    "flux_d",
    "hlr_b",
    "hlr_d",
    "q_d",
    "q_b",
    "beta",
]

DUMMY_PARAMS = {
    "flux_b": 0.0,
    "flux_d": 0.0,
    "hlr_b": 0.2,
    "hlr_d": 0.2,
    "q_d": 1.0,  # circle
    "q_b": 1.0,  # circle
    "beta": 0.0,
    "good_size": 1,
    "x": 0.0,
    "y": 0.0,
}


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


def format_column_to_dict(row):
    # ignore AGN component
    a_b = row["a_b"].item()
    b_b = row["b_b"].item()
    a_d = row["a_d"].item()
    b_d = row["b_d"].item()
    i_ab = row["i_ab"].item()

    fluxnorm_bulge = row["fluxnorm_bulge"].item()
    fluxnorm_disk = row["fluxnorm_disk"].item()
    total_fluxnorm = fluxnorm_disk + fluxnorm_bulge

    assert fluxnorm_bulge > 0 or fluxnorm_disk > 0, (
        "This should never happen in the catalog."
    )

    pa_bulge = row["pa_bulge"].item()
    pa_disk = row["pa_disk"].item()

    theta = pa_bulge if fluxnorm_bulge > 0 else pa_disk  # degrees

    # sanity
    if fluxnorm_bulge > 0 and fluxnorm_disk > 0:
        assert np.isclose(pa_bulge, pa_disk)

    # get flux
    flux_tot = mag2counts(i_ab, survey="LSST", filter="i").to_value("electron")
    flux_b = flux_tot * fluxnorm_bulge / total_fluxnorm
    flux_d = flux_tot * fluxnorm_disk / total_fluxnorm

    # dummy values to play nicely with vmap
    return {
        "flux_b": flux_b,
        "flux_d": flux_d,
        "hlr_b": np.sqrt(a_b * b_b) if fluxnorm_bulge > 0 else 0.2,
        "hlr_d": np.sqrt(a_d * b_d) if fluxnorm_disk > 0 else 0.2,
        "q_d": b_d / a_d if fluxnorm_disk > 0 else 1.0,  # circle
        "q_b": b_b / a_b if fluxnorm_bulge > 0 else 1.0,  # circle
        "beta": theta,
    }


def format_column_to_dict_extra(row):
    out = format_column_to_dict(row)
    out["good_size"] = row["good_size"]
    return out


# use jax random here, which I think will make reproducibility easier
def sample_cat(key, *, n_sources: int, cat):

    indices = random.choice(key, jnp.arange(len(cat)), shape=(n_sources,), replace=True)
    indices_np = np.array(indices)
    rows = cat[indices_np]

    sample_params = []
    for row in rows:
        sample_params.append(format_column_to_dict_extra(row))

    all_params = {}
    for p in PARAM_NAMES + ["good_size"]:
        out = []
        for n in range(n_sources):
            out.append(sample_params[n][p])
        all_params[p] = np.array(out)

    assert all_params["flux_b"].shape[0] == n_sources
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
    x = np.array(random.uniform(k2, minval=0, maxval=ilen, shape=(n_sources,)))
    y = np.array(random.uniform(k3, minval=0, maxval=ilen, shape=(n_sources,)))

    # get galaxy properties
    _, k4 = random.split(k)
    galaxy_props = sample_cat(k4, n_sources=n_sources, cat=cat)
    good_sizes = galaxy_props.pop("good_size")
    all_props = {**galaxy_props, "x": x, "y": y}

    assert n_sources <= max_n_gals, (
        "Number of sources in sample {} exceeds maximum number of sources {}.".format(
            n_sources, max_n_gals
        )
    )
    assert all_props["x"].shape == (n_sources,)
    assert all_props["flux_b"].shape == (n_sources,)
    assert good_sizes.shape == (n_sources,)
    assert np.all(good_sizes > 1), "Unphysical good size found"  # unphysical?

    return all_props, n_sources.item(), good_sizes


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
    slen: int | None = None,
    fft_size: int | None = None,
    check_stamp_sizes: bool = False,
    max_slen: int | None = None,
    good_sizes=None,
    buffer: int = None,
    size_bins=None,
):

    # create big image
    image = galsim.Image(ncol=ilen, nrow=ilen, scale=0.2, dtype=np.float64)
    wcs = image.wcs

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

        if check_stamp_sizes:
            assert max_slen is not None
            assert good_sizes is not None
            assert size_bins is not None
            assert buffer is not None

            # check no galaxy stamp size exceeds maximum one (we will miss one)
            assert max(stamp.array.shape) <= max_slen, (
                f"Stamp size {stamp.array.shape} exceeds maximum stamp size. "
                "Consider increasing largest stamp size bin."
            )

            # first determine what bin is actually used for this galaxy
            gsize = good_sizes[n]
            ss_bin_idx = np.searchsorted(size_bins, gsize + buffer)
            jgs_ss = size_bins[ss_bin_idx]

            # the check is whether that bin is larger than what galsim uses
            assert jgs_ss >= max(stamp.array.shape), (
                f"The stamp size used by JAX GalSim is smaller than what GalSim uses for this galaxy: {jgs_ss} vs {max(stamp.array.shape)}."
            )

        b = stamp.bounds & image.bounds
        if b.isDefined():
            image[b] += stamp[b]

    return image.array


def get_bd_jgs(
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


def _draw_stamp_jgs(
    galaxy_params: dict,
    image_pos: jgs.PositionD,
    local_wcs: jgs.PixelScale,
    *,
    psf: jgs.GSObject,
    slen: int,
    fft_size: int,
) -> jax.Array:
    gsparams = jgs.GSParams(minimum_fft_size=fft_size, maximum_fft_size=fft_size)
    convolved_object = get_bd_jgs(**galaxy_params, psf=psf).withGSParams(gsparams)
    stamp = convolved_object.drawImage(
        nx=slen, ny=slen, center=image_pos, wcs=local_wcs, dtype=jnp.float64
    )
    return stamp


def _always_draw_and_add(image, gparams, image_pos, local_wcs, psf, slen, fft_size):
    stamp = _draw_stamp_jgs(
        galaxy_params=gparams,
        image_pos=image_pos,
        local_wcs=local_wcs,
        psf=psf,
        slen=slen,
        fft_size=fft_size,
    )
    image[stamp.bounds] += stamp
    return image


def _draw_stamp_and_add_to_image(carry, x, *, psf, fft_size, slen):
    image = carry[0]
    gparams, image_pos, lwcs = x
    total_flux = gparams["flux_d"] + gparams["flux_b"]
    _body_fnc = partial(
        _always_draw_and_add,
        image=image,
        gparams=gparams,
        image_pos=image_pos,
        local_wcs=lwcs,
        psf=psf,
        slen=slen,
        fft_size=fft_size,
    )
    # skips all computation if flux is 0.0, but only with scan not vmap
    image = jax.lax.cond(total_flux == 0.0, lambda: image, _body_fnc)

    return (image,), None


def draw_jgs_scan_stamps(
    galaxy_params: dict,
    psf: jgs.GSObject,
    *,
    ilen: int,
    slen: int,
    fft_size: int,
    max_n_gals: int,
):
    # I think this version will be better in CPU than vmap

    # create big image
    image = jgs.ImageD(ncol=ilen, nrow=ilen, scale=0.2)
    wcs = image.wcs
    gparams = {**galaxy_params}  # copy dict to avoid changing original
    assert gparams["flux_d"].shape[0] == max_n_gals

    x = gparams.pop("x")
    y = gparams.pop("y")

    image_positions = vmap(lambda x, y: jgs.PositionD(x=x, y=y))(x, y)
    local_wcss = vmap(lambda x: wcs.local(image_pos=x))(image_positions)

    pad_image = jgs.ImageD(
        jnp.pad(image.array, slen), wcs=image.wcs, bounds=image.bounds.withBorder(slen)
    )

    _fnc_to_scan = partial(
        _draw_stamp_and_add_to_image, psf=psf, fft_size=fft_size, slen=slen
    )
    final_pad_image = jax.lax.scan(
        _fnc_to_scan,
        (pad_image,),
        xs=(gparams, image_positions, local_wcss),
        length=max_n_gals,
    )[0][0]

    return final_pad_image.array[slen:-slen, slen:-slen]


def _add_to_image(carry, x):
    image = carry[0]
    stamp = x
    image[stamp.bounds] += stamp
    return (image,), None


def draw_jgs_vmap_stamps(
    galaxy_params: dict,
    psf: jgs.GSObject,
    *,
    ilen: int,
    slen: int,
    fft_size: int,
    max_n_gals: int,
):

    # create big image
    image = jgs.ImageD(ncol=ilen, nrow=ilen, scale=0.2)
    wcs = image.wcs
    gparams = {**galaxy_params}
    assert gparams["flux_d"].shape[0] == max_n_gals

    x = gparams.pop("x")
    y = gparams.pop("y")

    image_positions = jax.vmap(lambda x, y: jgs.PositionD(x=x, y=y))(x, y)
    local_wcss = jax.vmap(lambda x: wcs.local(image_pos=x))(image_positions)

    _draw_stamps_vmapped = vmap(
        partial(_draw_stamp_jgs, psf=psf, slen=slen, fft_size=fft_size)
    )
    stamps = _draw_stamps_vmapped(gparams, image_positions, local_wcss)
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


################################################
# Catalog and other utilities below


def prepare_catalog(
    catsim_file: str,
    min_hlr=0.0,
    max_mag: float = 27.0,
):
    cat = Table.read(catsim_file, format="fits")

    # avoid objects that are too bright, too dim, or too big
    hlr_b = np.sqrt(cat["a_b"] * cat["b_b"])
    hlr_d = np.sqrt(cat["a_d"] * cat["b_d"])
    _mask1 = (hlr_b > min_hlr) | (hlr_d > min_hlr)
    _mask2 = cat["r_ab"] < max_mag
    mask = _mask1 & _mask2
    fcat = cat[mask]
    return fcat


def get_good_sizes_galsim(
    *, cat, psf, suffix: str, out_path: Path, overwrite: bool = False
):
    cache_fpath = out_path / f"good_sizes-{suffix}.npz"
    if Path(cache_fpath).exists() and not overwrite:
        print(f"INFO: Loading good sizes from file: {cache_fpath}")
        dt = np.load(cache_fpath)
        _good_sizes = dt["good_sizes"]
        _good_fft_sizes = dt["good_fft_sizes"]
    else:
        print("INFO: Computing good sizes for catalog")
        _good_sizes = []
        _good_fft_sizes = []
        for ii in tqdm(range(len(cat)), desc="Getting good sizes for cut..."):
            gal = get_bd_galsim(**format_column_to_dict(cat[ii]), psf=psf)
            _good_size = gal.getGoodImageSize(0.2)
            _good_sizes.append(_good_size)

            _, _good_fft_size = calculate_fft_size(gal, pixel_scale=0.2)
            _good_fft_sizes.append(_good_fft_size)

        _good_sizes = np.array(_good_sizes)
        _good_fft_sizes = np.array(_good_fft_sizes)
        np.savez(cache_fpath, good_sizes=_good_sizes, good_fft_sizes=_good_fft_sizes)

    return _good_sizes, _good_fft_sizes


def add_results_to_pdf(ii, pdf, *, gs_arr, jgs_np_arr, t_galsim, t_jgalsim):

    vmin = min(gs_arr.min(), jgs_np_arr.min())
    vmax = max(gs_arr.max(), jgs_np_arr.max())

    # residual with symmetric colormap
    res = gs_arr - jgs_np_arr
    res_vmax = max(abs(res.min()), abs(res.max()))
    res_vmin = -res_vmax
    # mask = gs_arr > 0  # galsim flux can only be positive or 0
    # res = np.zeros_like(residual)
    # res[mask] = residual[mask] / gs_arr[mask]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Sample {ii}  |  GalSim: {t_galsim:.4f}s  |  JAX-GalSim: {t_jgalsim:.4f}s",
        fontsize=13,
    )

    im0 = axes[0].imshow(gs_arr, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("GalSim")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        jgs_np_arr, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[1].set_title("JAX-GalSim")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        res, origin="lower", cmap="RdBu_r", vmin=res_vmin, vmax=res_vmax
    )
    axes[2].set_title("Residual (GalSim - JAX-GalSim)")
    fig.colorbar(im2, ax=axes[2])

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def calculate_fft_size(obj, pixel_scale, nx=None, ny=None):
    """Calculate the FFT size(s) GalSim would use to draw ``obj`` via ``drawImage(method='fft')``,
    following the same logic as ``galsim.GSObject.drawFFT_makeKImage``.

    Parameters:
        obj:            The profile (a `GSObject`) that would be drawn in Fourier space.
        pixel_scale:    The pixel scale of the image the profile would be drawn onto.
        nx:             The x-direction size (in pixels) of the target image, if already known.
                        [default: None]
        ny:             The y-direction size (in pixels) of the target image, if already known.
                        [default: None]

    Returns:
        A tuple ``(N, Nk)`` where ``N`` is the size of the real-space image used for the final
        (possibly wrapped) inverse FFT, and ``Nk`` is the size of the k-space image over which
        ``obj``'s ``kValue`` gets evaluated. ``Nk >= N``, with equality unless the k-space image
        would need to be larger to avoid aliasing, in which case it gets wrapped down to ``N``
        before the inverse FFT.
    """
    from galsim.errors import galsim_warn_fft

    from jax_galsim.image import Image

    # Start with what this profile thinks a good size would be given the pixel scale.
    N = int(obj.getGoodImageSize(pixel_scale))

    # We must make something big enough to cover the target image size, if given.
    if nx is not None and ny is not None:
        N = max(N, nx, ny)
    elif nx is not None or ny is not None:
        raise ValueError("Must provide both nx and ny, or neither.")

    # Round up to a good size for making FFTs:
    N = Image.good_fft_size(N)

    # Make sure we hit the minimum size specified in the gsparams.
    N = max(N, obj.gsparams.minimum_fft_size)

    dk = 2.0 * math.pi / (N * pixel_scale)

    maxk = float(obj.maxk)
    if N * dk / 2 > maxk:
        Nk = N
    else:
        # There will be aliasing.  Make a larger image and then wrap it.
        Nk = int(math.ceil(maxk / dk)) * 2

    if Nk > obj.gsparams.maximum_fft_size:
        galsim_warn_fft("drawFFT requires a very large FFT.", Nk)

    return N, Nk


def draw_all_bins_jgs(
    samples_per_bin_jax,
    n_iters_per_bin,
    psf: jgs.GSObject,
    *,
    ilen: int,
    draw_fncs: tuple,
    n_bins: int,
    device,
):
    param_names = tuple(samples_per_bin_jax[0].keys())
    jgs_arr = jnp.zeros((ilen, ilen), device=device, dtype=jnp.float64)

    for jj in range(n_bins):  # unrolled at traced time, could vmap?
        draw_fnc_jj = draw_fncs[jj]
        n_iters_jj = n_iters_per_bin[jj]
        samples_per_bin_jj = samples_per_bin_jax[jj]

        def _body_fnc(kk, arr):
            _batch = {p: samples_per_bin_jj[p][kk] for p in param_names}
            return arr + draw_fnc_jj(_batch, psf)

        jgs_arr = fori_loop(0, n_iters_jj, _body_fnc, jgs_arr)

    return jgs_arr


def prepare_per_bin_samples(
    sample: np.ndarray,
    gsizes: np.ndarray,
    *,
    stamp_slen_bins: tuple,
    max_n_gals_bins: tuple,
    max_n_iters: int,
    buffer: int,
):

    samples_per_bin = []
    n_iters_per_bin = []
    n_bins = len(stamp_slen_bins)

    # keep track of already assigned galaxies
    # ignore ones that are dummies in sample
    _already_assigned = np.zeros_like(gsizes).astype(bool)
    for jj in range(n_bins):
        stamp_slen_jj = stamp_slen_bins[jj]
        max_n_gals_jj = max_n_gals_bins[jj]

        _mask1 = ~_already_assigned
        _mask2 = np.less_equal(gsizes + buffer, stamp_slen_jj)
        _mask = _mask1 & _mask2
        sample_jj = {k: v[_mask] for k, v in sample.items()}

        n_gals = _mask.sum().item()
        n_iters_jj = math.ceil(n_gals / max_n_gals_jj)
        assert n_iters_jj <= max_n_iters, (
            f"Number of iterations in size bin index {jj} is {n_iters_jj} which is larger than max_n_iters:{max_n_iters}"
        )

        # here we want static shapes (small memory overheard with parameters)
        # but in the drawing function will explicitly skip in while loop these extra ones
        # based on n_iters_per_bin ==> especially useful for vmap
        n_pad = max_n_iters * max_n_gals_jj - n_gals
        for p in sample_jj:
            _padding = np.full(n_pad, fill_value=DUMMY_PARAMS[p])
            sample_jj[p] = np.concatenate([sample_jj[p], _padding])
            sample_jj[p] = sample_jj[p].reshape(max_n_iters, max_n_gals_jj)

        samples_per_bin.append(sample_jj)
        n_iters_per_bin.append(n_iters_jj)
        _already_assigned[_mask] = True

    assert np.all(_already_assigned), (
        "Not all galaxies in sampled were assigned. "
        "Probably some galaxy needs too large of a stamp size."
    )
    return samples_per_bin, np.array(n_iters_per_bin)
