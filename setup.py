from setuptools import find_packages, setup

setup(
    name="JAX-GalSim",
    version="0.0.1rc1",
    url="https://github.com/GalSim-developers/JAX-GalSim",
    author="GalSim Developers",
    description="The modular galaxy image simulation toolkit, but in JAX",
    packages=find_packages(),
    license="BSD License",
    install_requires=[
        "numpy >= 1.18.0",
        "galsim >= 2.3.0",
        "jax",
        "jaxlib",
        "astropy >= 2.0",
        "tensorflow-probability>=0.21.0",
    ],
    tests_require=["pytest"],
)
