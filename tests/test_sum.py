import jax_galsim as galsim


def test_gsparams():
    """Test withGSParams with some non-default gsparams
    """
    obj1 = galsim.Exponential(half_light_radius=1.7)
    obj2 = galsim.Gaussian(sigma=0.2)
    gsp = galsim.GSParams(folding_threshold=1.e-4,
                          maxk_threshold=1.e-4, maximum_fft_size=1.e4)
    gsp2 = galsim.GSParams(folding_threshold=1.e-2, maxk_threshold=1.e-2)

    sum = galsim.Sum(obj1, obj2)
    sum1 = sum.withGSParams(gsp)
    assert sum.gsparams == galsim.GSParams()
    assert sum1.gsparams == gsp
    assert sum1.obj_list[0].gsparams == gsp
    assert sum1.obj_list[1].gsparams == gsp

    sum2 = galsim.Sum(obj1.withGSParams(gsp), obj2.withGSParams(gsp))
    sum3 = galsim.Sum(galsim.Exponential(half_light_radius=1.7, gsparams=gsp),
                      galsim.Gaussian(sigma=0.2))
    sum4 = galsim.Add(obj1, obj2, gsparams=gsp)
    assert sum != sum1
    assert sum1 == sum2
    assert sum1 == sum3
    assert sum1 == sum4
    print('stepk = ', sum.stepk, sum1.stepk)
    assert sum1.stepk < sum.stepk
    print('maxk = ', sum.maxk, sum1.maxk)
    assert sum1.maxk > sum.maxk

    sum5 = galsim.Add(obj1, obj2, gsparams=gsp, propagate_gsparams=False)
    assert sum5 != sum4
    assert sum5.gsparams == gsp
    assert sum5.obj_list[0].gsparams == galsim.GSParams()
    assert sum5.obj_list[1].gsparams == galsim.GSParams()

    sum6 = sum5.withGSParams(gsp2)
    assert sum6 != sum5
    assert sum6.gsparams == gsp2
    assert sum6.obj_list[0].gsparams == galsim.GSParams()
    assert sum6.obj_list[1].gsparams == galsim.GSParams()
