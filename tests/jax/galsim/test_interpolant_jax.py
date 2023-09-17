"""These are pure tests of the interpolants to use while
InterpolatedImage is not yet implemented.

Much of the code is copied out of the galsim test suite.
"""
import jax
import time
import galsim as ref_galsim
import numpy as np
import jax_galsim as galsim
from galsim_test_helpers import timer, assert_raises
from scipy.special import sici
import pickle
import pytest


def do_pickle(obj1):
    return pickle.loads(pickle.dumps(obj1))


def test_interpolant_jax_eq():
    interps1 = (
        [
            galsim.Delta(),
            galsim.Nearest(),
            galsim.SincInterpolant(),
            galsim.Linear(),
            galsim.Cubic(),
            galsim.Quintic(),
        ]
        + [galsim.Lanczos(i, conserve_dc=False) for i in range(1, 31)]
        + [galsim.Lanczos(i, conserve_dc=True) for i in range(1, 31)]
    )
    interps2 = (
        [
            galsim.Delta(),
            galsim.Nearest(),
            galsim.SincInterpolant(),
            galsim.Linear(),
            galsim.Cubic(),
            galsim.Quintic(),
        ]
        + [galsim.Lanczos(i, conserve_dc=False) for i in range(1, 31)]
        + [galsim.Lanczos(i, conserve_dc=True) for i in range(1, 31)]
    )

    def _compare():
        for i, intrp1 in enumerate(interps1):
            for j, intrp2 in enumerate(interps1):
                if i == j:
                    assert intrp1 == intrp2
                    assert not (intrp1 != intrp2)
                    assert intrp1 is intrp2
                else:
                    assert intrp1 != intrp2
                    assert not (intrp1 == intrp2)
                    assert intrp1 is not intrp2

            for j, intrp2 in enumerate(interps2):
                if i == j:
                    assert intrp1 == intrp2
                    assert not (intrp1 != intrp2)
                else:
                    assert intrp1 != intrp2
                    assert not (intrp1 == intrp2)
                assert intrp1 is not intrp2

    _compare()
    # make sure that any flattening and unflatting doesn't change anything
    for interp in interps1:
        f = jax.jit(interp.xval)
        f(0.1)
    for interp in interps2:
        f = jax.jit(interp.xval)
        f(0.1)
    _compare()


@pytest.mark.parametrize("cdc", [True, False])
def test_interpolant_jax_lanczos_perf(cdc):
    t0 = time.time()
    jgs = galsim.Lanczos(5, conserve_dc=cdc)
    t0 = time.time() - t0
    print("\njax_galsim init: %0.4e" % t0, flush=True)

    t0 = time.time()
    jgs = galsim.Lanczos(5, conserve_dc=cdc)
    t0 = time.time() - t0
    print("jax_galsim init: %0.4e" % t0, flush=True)

    t0 = time.time()
    gs = ref_galsim.Lanczos(5, conserve_dc=cdc)
    t0 = time.time() - t0
    print("galsim init    : %0.4e" % t0, flush=True)

    def _timeit(lz, ntest=10, jit=False, dox=False):
        k = np.array([0.3, 0.4, 0.5])
        if dox:
            if isinstance(lz, galsim.Lanczos) and jit:
                f = jax.jit(lz.xval)
            else:
                f = lz.xval
        else:
            if isinstance(lz, galsim.Lanczos) and jit:
                f = jax.jit(lz.kval)
            else:
                f = lz.kval
        f(k)
        t0 = time.time()
        for i in range(ntest):
            f(k + i / ntest)
        return (time.time() - t0) / ntest / 1e-6

    print("\nkval timing:", flush=True)
    print("jax_galsim:        %0.4e [ns]" % _timeit(jgs), flush=True)
    print("jax_galsim w/ JIT: %0.4e [ns]" % _timeit(jgs, jit=True), flush=True)
    print("galsim:            %0.4e [ns]" % _timeit(gs), flush=True)

    print("\nxval timing:", flush=True)
    print("jax_galsim:        %0.4e [ns]" % _timeit(jgs, dox=True), flush=True)
    print(
        "jax_galsim w/ JIT: %0.4e [ns]" % _timeit(jgs, jit=True, dox=True), flush=True
    )
    print("galsim:            %0.4e [ns]" % _timeit(gs, dox=True), flush=True)


@timer
def test_interpolant_jax_same_as_galsim():
    x = np.linspace(-10, 10, 141)
    k = np.linspace(-0.2, 0.2, 5)
    if np.any(~np.isfinite(k)):
        k = k[np.isfinite(k)]
    interps = (
        [
            galsim.Delta(),
            galsim.Nearest(),
            galsim.SincInterpolant(),
            galsim.Linear(),
            galsim.Cubic(),
            galsim.Quintic(),
        ]
        + [galsim.Lanczos(i, conserve_dc=False) for i in range(1, 31)]
        + [galsim.Lanczos(i, conserve_dc=True) for i in range(1, 31)]
    )
    for interp in interps:
        print(str(interp))
        gs = getattr(ref_galsim, interp.__class__.__name__)
        if isinstance(interp, galsim.Lanczos):
            gs = gs(interp.n, conserve_dc=interp.conserve_dc)
        else:
            gs = gs()
        jgs = interp
        np.testing.assert_allclose(
            gs.positive_flux, jgs.positive_flux, atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(
            gs.negative_flux, jgs.negative_flux, atol=1e-5, rtol=1e-5
        )
        if not isinstance(jgs, galsim.Lanczos):
            np.testing.assert_allclose(gs.krange, jgs.krange)
        np.testing.assert_allclose(gs.xrange, jgs.xrange)
        np.testing.assert_allclose(gs._i.urange(), jgs.urange())
        np.testing.assert_allclose(gs.ixrange, jgs.ixrange)
        np.testing.assert_allclose(gs.xval(x), jgs.xval(x), rtol=0, atol=1e-10)
        np.testing.assert_allclose(gs.kval(k), jgs.kval(k), rtol=0, atol=5e-5)


def test_interpolant_jax_sinc_integrals():
    tol = 1e-5
    gs = ref_galsim.SincInterpolant(gsparams=ref_galsim.GSParams(kvalue_accuracy=tol))
    jgs = galsim.SincInterpolant(gsparams=galsim.GSParams(kvalue_accuracy=tol))
    np.testing.assert_allclose(gs.unit_integrals(), jgs.unit_integrals())
    np.testing.assert_allclose(gs.xrange, jgs.xrange)
    np.testing.assert_allclose(gs.ixrange, jgs.ixrange)
    np.testing.assert_allclose(jgs.negative_flux, jgs.positive_flux - 1.0)
    np.testing.assert_allclose(
        gs.positive_flux, jgs.positive_flux, atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        gs.negative_flux, jgs.negative_flux, atol=1e-5, rtol=1e-5
    )


@timer
def test_interpolant_jax_si_pade():
    x = np.linspace(-10, 10, 141)
    np.testing.assert_allclose(
        [float(galsim.bessel.si(_x)) for _x in x],
        [sici(_x)[0] for _x in x],
    )


@timer
def test_interpolant_jax_smoke():
    """Test the interpolants directly."""
    x = np.linspace(-10, 10, 141)

    # Delta
    d = galsim.Delta()
    print(repr(d.gsparams))
    print(repr(galsim.GSParams()))
    assert d.gsparams == galsim.GSParams()
    assert d.xrange == 0
    assert d.ixrange == 0
    assert np.isclose(d.krange, 2.0 * np.pi / d.gsparams.kvalue_accuracy)
    assert np.isclose(d.krange, 2.0 * np.pi * d._i.urange())
    assert d.positive_flux == 1
    assert d.negative_flux == 0
    print(repr(d))
    do_pickle(galsim.Delta())
    do_pickle(galsim.Interpolant.from_name("delta"))

    true_xval = np.zeros_like(x)
    true_xval[np.abs(x) < d.gsparams.kvalue_accuracy / 2] = (
        1.0 / d.gsparams.kvalue_accuracy
    )
    np.testing.assert_allclose(d.xval(x), true_xval)
    np.testing.assert_allclose(d.kval(x), 1.0)
    assert np.isclose(d.xval(x[12]), true_xval[12])
    assert np.isclose(d.kval(x[12]), 1.0)

    # Nearest
    n = galsim.Nearest()
    assert n.gsparams == galsim.GSParams()
    assert n.xrange == 0.5
    assert n.ixrange == 1
    assert np.isclose(n.krange, 2.0 / n.gsparams.kvalue_accuracy)
    assert n.positive_flux == 1
    assert n.negative_flux == 0
    do_pickle(galsim.Nearest())
    do_pickle(galsim.Interpolant.from_name("nearest"))

    true_xval = np.zeros_like(x)
    true_xval[np.abs(x) < 0.5] = 1
    np.testing.assert_allclose(n.xval(x), true_xval)
    true_kval = np.sinc(x / 2 / np.pi)
    np.testing.assert_allclose(n.kval(x), true_kval)
    assert np.isclose(n.xval(x[12]), true_xval[12])
    assert np.isclose(n.kval(x[12]), true_kval[12])

    # Conserves dc flux:
    # Most interpolants (not Delta above) conserve a constant (DC) flux.
    # This means input points separated by 1 pixel with any subpixel phase
    # will sum to 1.  The input x array has 7 phases, so the total sum is 7.
    print("Nearest sum = ", np.sum(n.xval(x)))
    assert np.isclose(np.sum(n.xval(x)), 7.0)

    # SincInterpolant
    s = galsim.SincInterpolant()
    assert s.gsparams == galsim.GSParams()
    assert np.isclose(s.xrange, 1.0 / (np.pi * s.gsparams.kvalue_accuracy))
    assert s.ixrange == 2 * np.ceil(s.xrange)
    assert np.isclose(s.krange, np.pi)
    assert np.isclose(s.krange, 2.0 * np.pi * s._i.urange())
    assert np.isclose(
        s.positive_flux, 3.18726437
    )  # Empirical -- this is a regression test
    assert np.isclose(s.negative_flux, s.positive_flux - 1.0, rtol=1.0e-4)
    do_pickle(galsim.SincInterpolant())
    do_pickle(galsim.Interpolant.from_name("sinc"))

    true_xval = np.sinc(x)
    np.testing.assert_allclose(s.xval(x), true_xval)
    true_kval = np.zeros_like(x)
    true_kval[np.abs(x) < np.pi] = 1.0
    np.testing.assert_allclose(s.kval(x), true_kval)
    assert np.isclose(s.xval(x[12]), true_xval[12])
    assert np.isclose(s.kval(x[12]), true_kval[12])

    # Conserves dc flux:
    # This one would conserve dc flux, but we don't go out far enough.
    # At +- 10 pixels, it's only about 6.86
    print("Sinc sum = ", np.sum(s.xval(x)))
    assert np.isclose(np.sum(s.xval(x)), 7.0, rtol=0.02)

    # Linear
    ln = galsim.Linear()
    assert ln.gsparams == galsim.GSParams()
    assert ln.xrange == 1.0
    assert ln.ixrange == 2
    assert np.isclose(ln.krange, 2.0 / ln.gsparams.kvalue_accuracy**0.5)
    assert np.isclose(ln.krange, 2.0 * np.pi * ln._i.urange())
    assert ln.positive_flux == 1
    assert ln.negative_flux == 0
    do_pickle(galsim.Linear())
    do_pickle(galsim.Interpolant.from_name("linear"))

    true_xval = np.zeros_like(x)
    true_xval[np.abs(x) < 1] = 1.0 - np.abs(x[np.abs(x) < 1])
    np.testing.assert_allclose(ln.xval(x), true_xval)
    true_kval = np.sinc(x / 2 / np.pi) ** 2
    np.testing.assert_allclose(ln.kval(x), true_kval)
    assert np.isclose(ln.xval(x[12]), true_xval[12])
    assert np.isclose(ln.kval(x[12]), true_kval[12])

    # Conserves dc flux:
    print("Linear sum = ", np.sum(ln.xval(x)))
    assert np.isclose(np.sum(ln.xval(x)), 7.0)

    # Cubic
    c = galsim.Cubic()
    assert c.gsparams == galsim.GSParams()
    assert c.xrange == 2.0
    assert c.ixrange == 4
    assert np.isclose(
        c.krange, 2.0 * (3**1.5 / 8 / c.gsparams.kvalue_accuracy) ** (1.0 / 3.0)
    )
    assert np.isclose(c.krange, 2.0 * np.pi * c._i.urange())
    assert np.isclose(c.positive_flux, 13.0 / 12.0)
    assert np.isclose(c.negative_flux, 1.0 / 12.0)
    do_pickle(galsim.Cubic())
    do_pickle(galsim.Interpolant.from_name("cubic"))

    true_xval = np.zeros_like(x)
    ax = np.abs(x)
    m = ax < 1
    true_xval[m] = 1.0 + ax[m] ** 2 * (1.5 * ax[m] - 2.5)
    m = (1 <= ax) & (ax < 2)
    true_xval[m] = -0.5 * (ax[m] - 1) * (2.0 - ax[m]) ** 2
    np.testing.assert_allclose(c.xval(x), true_xval)
    sx = np.sinc(x / 2 / np.pi)
    cx = np.cos(x / 2)
    true_kval = sx**3 * (3 * sx - 2 * cx)
    np.testing.assert_allclose(c.kval(x), true_kval)
    assert np.isclose(c.xval(x[12]), true_xval[12])
    assert np.isclose(c.kval(x[12]), true_kval[12])

    # Conserves dc flux:
    print("Cubic sum = ", np.sum(c.xval(x)))
    assert np.isclose(np.sum(c.xval(x)), 7.0)

    # Quintic
    q = galsim.Quintic()
    assert q.gsparams == galsim.GSParams()
    assert q.xrange == 3.0
    assert q.ixrange == 6
    assert np.isclose(
        q.krange, 2.0 * (5**2.5 / 108 / q.gsparams.kvalue_accuracy) ** (1.0 / 3.0)
    )
    assert np.isclose(q.krange, 2.0 * np.pi * q._i.urange())
    assert np.isclose(
        q.positive_flux, (13018561.0 / 11595672.0) + (17267.0 / 14494590.0) * 31**0.5
    )
    assert np.isclose(q.negative_flux, q.positive_flux - 1.0)
    do_pickle(galsim.Quintic())
    do_pickle(galsim.Interpolant.from_name("quintic"))

    true_xval = np.zeros_like(x)
    ax = np.abs(x)
    m = ax < 1.0
    true_xval[m] = 1.0 + ax[m] ** 3 * (
        -95.0 / 12.0 + 23.0 / 2.0 * ax[m] - 55.0 / 12.0 * ax[m] ** 2
    )
    m = (1 <= ax) & (ax < 2)
    true_xval[m] = (
        (ax[m] - 1)
        * (2.0 - ax[m])
        * (
            23.0 / 4.0
            - 29.0 / 2.0 * ax[m]
            + 83.0 / 8.0 * ax[m] ** 2
            - 55.0 / 24.0 * ax[m] ** 3
        )
    )
    m = (2 <= ax) & (ax < 3)
    true_xval[m] = (
        (ax[m] - 2)
        * (3.0 - ax[m]) ** 2
        * (-9.0 / 4.0 + 25.0 / 12.0 * ax[m] - 11.0 / 24.0 * ax[m] ** 2)
    )
    np.testing.assert_allclose(q.xval(x), true_xval)
    sx = np.sinc(x / 2 / np.pi)
    cx = np.cos(x / 2)
    true_kval = sx**5 * (
        sx * (55.0 - 19.0 / 4.0 * x**2) + cx * (x**2 / 2.0 - 54.0)
    )
    np.testing.assert_allclose(q.kval(x), true_kval)
    assert np.isclose(q.xval(x[12]), true_xval[12])
    assert np.isclose(q.kval(x[12]), true_kval[12])

    # Conserves dc flux:
    print("Quintic sum = ", np.sum(q.xval(x)))
    assert np.isclose(np.sum(q.xval(x)), 7.0)

    # Lanczos
    l3 = galsim.Lanczos(3)
    assert l3.gsparams == galsim.GSParams()
    assert l3.conserve_dc is True
    assert l3.n == 3
    assert l3.xrange == l3.n
    assert l3.ixrange == 2 * l3.n
    assert np.isclose(
        l3.krange, 2.0 * np.pi * l3._i.urange()
    )  # No analytic version for this one.
    print(l3.positive_flux, l3.negative_flux)
    assert np.isclose(
        l3.positive_flux, 1.1793639
    )  # Empirical -- this is a regression test
    assert np.isclose(l3.negative_flux, l3.positive_flux - 1.0, rtol=1.0e-4)
    do_pickle(galsim.Lanczos(n=7, conserve_dc=False))
    do_pickle(galsim.Lanczos(3))
    do_pickle(galsim.Interpolant.from_name("lanczos7"))
    do_pickle(galsim.Interpolant.from_name("lanczos9F"))
    do_pickle(galsim.Interpolant.from_name("lanczos8T"))
    assert_raises(ValueError, galsim.Interpolant.from_name, "lanczos3A")
    assert_raises(ValueError, galsim.Interpolant.from_name, "lanczosF")
    assert_raises(ValueError, galsim.Interpolant.from_name, "lanzos")

    # Note: 1-7 all have special case code, so check them. 8 uses the generic code.
    for n in [1, 2, 3, 4, 5, 6, 7, 8]:
        ln = galsim.Lanczos(n, conserve_dc=False)
        assert ln.conserve_dc is False
        assert ln.n == n
        true_xval = np.zeros_like(x)
        true_xval[np.abs(x) < n] = np.sinc(x[np.abs(x) < n]) * np.sinc(
            x[np.abs(x) < n] / n
        )
        np.testing.assert_allclose(ln.xval(x), true_xval, rtol=1.0e-5, atol=1.0e-10)
        assert np.isclose(ln.xval(x[12]), true_xval[12])

        # Lanczos notably does not conserve dc flux
        print("Lanczos(%s,conserve_dc=False) sum = " % n, np.sum(ln.xval(x)))

        # With conserve_dc=True, it does a bit better, but still only to 1.e-4 accuracy.
        lndc = galsim.Lanczos(n, conserve_dc=True)
        np.testing.assert_allclose(lndc.xval(x), true_xval, rtol=0.3, atol=1.0e-10)
        print("Lanczos(%s,conserve_dc=True) sum = " % n, np.sum(lndc.xval(x)))
        assert np.isclose(np.sum(lndc.xval(x)), 7.0, rtol=1.0e-4)

        # The math for kval (at least when conserve_dc=False) is complicated, but tractable.
        # It ends up using the Si function, which is in scipy as scipy.special.sici
        vp = n * (x / np.pi + 1)
        vm = n * (x / np.pi - 1)
        true_kval = (
            (vm - 1) * sici(np.pi * (vm - 1))[0]
            - (vm + 1) * sici(np.pi * (vm + 1))[0]
            - (vp - 1) * sici(np.pi * (vp - 1))[0]
            + (vp + 1) * sici(np.pi * (vp + 1))[0]
        ) / (2 * np.pi)
        np.testing.assert_allclose(ln.kval(x), true_kval, rtol=1.0e-4, atol=1.0e-8)
        assert np.isclose(ln.kval(x[12]), true_kval[12])

    # Base class is invalid.
    assert_raises(NotImplementedError, galsim.Interpolant)

    # 2d arrays are invalid.
    x2d = np.ones((5, 5))
    with assert_raises(galsim.GalSimValueError):
        s.xval(x2d)
    with assert_raises(galsim.GalSimValueError):
        s.kval(x2d)


@timer
def test_interpolant_jax_unit_integrals():
    interps = (
        [
            galsim.Delta(),
            galsim.Nearest(),
            galsim.SincInterpolant(),
            galsim.Linear(),
            galsim.Cubic(),
            galsim.Quintic(),
        ]
        + [galsim.Lanczos(i, conserve_dc=False) for i in range(1, 31)]
        + [galsim.Lanczos(i, conserve_dc=True) for i in range(1, 31)]
    )
    for interp in interps:
        print(str(interp))
        # Compute directly with int1d
        n = interp.ixrange // 2 + 1
        direct_integrals = np.zeros(n)
        if isinstance(interp, galsim.Delta):
            # int1d doesn't handle this well.
            direct_integrals[0] = 1
        else:
            for k in range(n):
                direct_integrals[k] = ref_galsim.integ.int1d(
                    interp.xval, k - 0.5, k + 0.5
                )
        print("direct: ", direct_integrals)

        # Get from unit_integrals method (sometimes using analytic formulas)
        integrals = interp.unit_integrals()
        print("integrals: ", len(integrals), integrals)

        assert len(integrals) == n
        np.testing.assert_allclose(integrals, direct_integrals, atol=1.0e-12)

        if n > 10:
            print("n>10 for ", repr(interp))
            integrals2 = interp.unit_integrals(max_len=10)
            assert len(integrals2) == 10
            np.testing.assert_allclose(integrals2, integrals[:10], atol=0, rtol=0)

    # Test making shorter versions before longer ones
    interp = galsim.Lanczos(11)
    short = interp.unit_integrals(max_len=5)
    long = interp.unit_integrals(max_len=10)
    med = interp.unit_integrals(max_len=8)
    full = interp.unit_integrals()

    assert len(full) > 10
    np.testing.assert_array_equal(short, full[:5])
    np.testing.assert_array_equal(med, full[:8])
    np.testing.assert_array_equal(long, full[:10])
