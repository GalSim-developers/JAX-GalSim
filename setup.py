from setuptools import find_packages, setup

setup(
    name="Galaxim",
    version="0.0.1rc1",
    url="https://github.com/GalSim-developers/Galaxim",
    author="GalSim Developers",
    description="The modular galaxy image simulation toolkit, but in JAX",
    packages=find_packages(),
    license="BSD License",
    install_requires=["numpy >= 1.18.0", "galsim >= 2.3.0", "jax", "jaxlib"],
    tests_require=["pytest"],
)
