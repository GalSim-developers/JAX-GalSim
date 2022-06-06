from setuptools import setup, find_packages

setup(
    name="JAX-GalSim",
    version="0.0.1rc1",
    url="https://github.com/GalSim-developers/JAX-GalSim",
    author="GalSim Developers",
    description="The modular galaxy image simulation toolkit, but in JAX",
    packages=find_packages(),
    license="BSD License",
    install_requires=["numpy >= 1.18.0", "jax >= 0.3.0", "galsim"],
)
