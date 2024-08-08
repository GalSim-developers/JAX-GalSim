import galsim as _galsim
import numpy as np
import pytest

import jax_galsim


@pytest.mark.parametrize("projection", [None, "gnomonic", "stereographic", "lambert", "postel"])
def test_celestial_jax_comp_to_galsim(projection):
    gcc = _galsim.CelestialCoord(0.234 * _galsim.radians, 0.342 * _galsim.radians)
    gcA = _galsim.CelestialCoord(-0.193 * _galsim.radians, 0.882 * _galsim.radians)
    gcB = _galsim.CelestialCoord(
        (-0.193 + 1.7e-8) * _galsim.radians,
        (0.882 + 1.2e-8) * _galsim.radians,
    )
    gcC = _galsim.CelestialCoord(
        (-0.193 - 2.4e-8) * _galsim.radians,
        (0.882 + 3.1e-8) * _galsim.radians,
    )

    cc = jax_galsim.CelestialCoord.from_galsim(gcc)
    cA = jax_galsim.CelestialCoord.from_galsim(gcA)
    cB = jax_galsim.CelestialCoord.from_galsim(gcB)
    cC = jax_galsim.CelestialCoord.from_galsim(gcC)

    np.testing.assert_allclose(
        cB.distanceTo(cC).rad,
        gcB.distanceTo(gcC).rad,
    )
    np.testing.assert_allclose(
        cC.distanceTo(cA).rad,
        gcC.distanceTo(gcA).rad,
    )
    np.testing.assert_allclose(
        cA.distanceTo(cB).rad,
        gcA.distanceTo(gcB).rad,
    )
    np.testing.assert_allclose(
        cA.angleBetween(cB, cC).rad,
        gcA.angleBetween(gcB, gcC).rad,
    )
    np.testing.assert_allclose(
        cB.angleBetween(cC, cA).rad,
        gcB.angleBetween(gcC, gcA).rad,
    )
    np.testing.assert_allclose(
        cC.angleBetween(cA, cB).rad,
        gcC.angleBetween(gcA, gcB).rad,
    )

    np.testing.assert_allclose(
        cA.area(cB, cC),
        gcA.area(gcB, gcC),
    )

    np.testing.assert_allclose(cc.ra.rad, gcc.ra.rad)
    np.testing.assert_allclose(cc.dec.rad, gcc.dec.rad)
    np.testing.assert_allclose(cA.ra.rad, gcA.ra.rad)
    np.testing.assert_allclose(cA.dec.rad, gcA.dec.rad)
    np.testing.assert_allclose(cB.ra.rad, gcB.ra.rad)
    np.testing.assert_allclose(cB.dec.rad, gcB.dec.rad)

    np.testing.assert_allclose(cc.get_xyz(), gcc.get_xyz())
    np.testing.assert_allclose(cA.get_xyz(), gcA.get_xyz())
    np.testing.assert_allclose(cB.get_xyz(), gcB.get_xyz())

    np.testing.assert_allclose(cc.distanceTo(cA).rad, gcc.distanceTo(gcA).rad)
    np.testing.assert_allclose(cc.area(cA, cB), gcc.area(gcA, gcB))

    np.testing.assert_allclose(
        cc.greatCirclePoint(cA, 0.1 * jax_galsim.radians).rad,
        gcc.greatCirclePoint(gcA, 0.1 * _galsim.radians).rad,
    )

    np.testing.assert_allclose(
        cc.angleBetween(cA, cB).rad,
        gcc.angleBetween(gcA, gcB).rad,
    )

    np.testing.assert_allclose(
        cc.project(cA, projection=projection)[0] / jax_galsim.radians,
        gcc.project(gcA, projection=projection)[0] / _galsim.radians,
    )

    np.testing.assert_allclose(
        cc.project(cA, projection=projection)[1] / jax_galsim.radians,
        gcc.project(gcA, projection=projection)[1] / _galsim.radians,
    )

    u2, v2 = cc.project(cA, projection=projection)
    gu2, gv2 = gcc.project(gcA, projection=projection)

    np.testing.assert_allclose(
        cc.deproject(u2, v2, projection=projection).ra / jax_galsim.radians,
        gcc.deproject(gu2, gv2, projection=projection).ra / _galsim.radians,
    )

    np.testing.assert_allclose(
        cc.deproject(u2, v2, projection=projection).dec / jax_galsim.radians,
        gcc.deproject(gu2, gv2, projection=projection).dec / _galsim.radians,
    )

    np.testing.assert_allclose(
        cc.jac_deproject(u2, v2, projection=projection),
        gcc.jac_deproject(gu2, gv2, projection=projection),
    )


def test_celestial_jax_ecliptic_obliquity():
    from coord.util import ecliptic_obliquity

    from jax_galsim.celestial import _ecliptic_obliquity

    for epoch in range(1900, 2200):
        np.testing.assert_allclose(
            ecliptic_obliquity(epoch).rad,
            _ecliptic_obliquity(epoch).rad,
        )
