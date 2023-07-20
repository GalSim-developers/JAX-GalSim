import galsim as _galsim
from jax._src.numpy.util import _wraps

from jax_galsim.position import PositionD, PositionI


@_wraps(_galsim.utilities.parse_pos_args)
def parse_pos_args(args, kwargs, name1, name2, integer=False, others=[]):
    def canindex(arg):
        try:
            arg[0], arg[1]
        except (TypeError, IndexError):
            return False
        else:
            return True

    other_vals = []
    if len(args) == 0:
        # Then name1,name2 need to be kwargs
        try:
            x = kwargs.pop(name1)
            y = kwargs.pop(name2)
        except KeyError:
            raise TypeError("Expecting kwargs %s, %s.  Got %s" % (name1, name2, kwargs.keys()))
    elif (
        isinstance(args[0], PositionI) or (not integer and isinstance(args[0], PositionD))
    ) and len(args) <= 1 + len(others):
        x = args[0].x
        y = args[0].y
        for arg in args[1:]:
            other_vals.append(arg)
            others.pop(0)
    elif canindex(args[0]) and len(args) <= 1 + len(others):
        x = args[0][0]
        y = args[0][1]
        for arg in args[1:]:
            other_vals.append(arg)
            others.pop(0)
    elif len(args) == 1:
        if integer:
            raise TypeError("Cannot parse argument %s as a PositionI" % (args[0]))
        else:
            raise TypeError("Cannot parse argument %s as a PositionD" % (args[0]))
    elif len(args) <= 2 + len(others):
        x = args[0]
        y = args[1]
        for arg in args[2:]:
            other_vals.append(arg)
            others.pop(0)
    else:
        raise TypeError("Too many arguments supplied")
    # Read any remaining other kwargs
    if others:
        for name in others:
            val = kwargs.pop(name)
            other_vals.append(val)
    if kwargs:
        raise TypeError("Received unexpected keyword arguments: %s", kwargs)

    if integer:
        pos = PositionI(x, y)
    else:
        pos = PositionD(x, y)
    if other_vals:
        return (pos,) + tuple(other_vals)
    else:
        return pos
