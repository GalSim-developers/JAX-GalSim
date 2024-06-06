from jax_galsim.core.utils import implements, _parse_galsimdoc
from galsim import Gaussian as _Gaussian


def test_implements_parse_galsimdoc():
    docstring = _Gaussian.__doc__
    p = _parse_galsimdoc(docstring)

    assert p.signature == ""
    assert p.summary == "A class describing a 2D Gaussian surface brightness profile."
    assert "is characterized by two propertie" in p.front_matter
    assert "Parameters:" in p.front_matter
    assert p.sections == {}


class TestImplements:
    """The summary is
    cool.

    This is front matter.

    Parameters:
        blah:    here we go again
    """
    pass

@implements(TestImplements, lax_description="This is a lax description")
class LAXTestImplements:
    pass


def test_implements():
    docstring = TestImplements.__doc__
    p = _parse_galsimdoc(docstring)

    assert p.signature == ""
    assert p.summary == "The summary is\n    cool."
    assert "This is front matter." in p.front_matter
    assert "LAX" not in p.front_matter
    assert p.sections == {}

    docstring = LAXTestImplements.__doc__
    p = _parse_galsimdoc(docstring)
    assert p.signature == ""
    assert p.summary == "The summary is\n    cool."
    assert "This is front matter." in p.front_matter
    assert "LAX" in p.front_matter
    assert p.sections == {}
