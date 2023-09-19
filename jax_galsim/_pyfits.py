# copied out of galsim/_pyfits.py
# I would just use the galsim version, but it is a private submodule
# and I don't want to deal with an API break - MRB 2023-09-17

# We used to support legacy pyfits in addition to astropy.io.fits.  We still call
# astropy.io.fits pyfits in the code, but we have removed the legacy compatibility hacks.

import sys
import astropy.io.fits as pyfits

if "PyPy" in sys.version:  # pragma: no cover
    # As of astropy version 4.2.1, the memmap stuff didn't work with PyPy, since it
    # needed getrefcount, which PyPy doesn't have.
    pyfits.Conf.use_memmap = False
