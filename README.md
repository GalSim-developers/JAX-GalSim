# JAX-GalSim

**JAX port of GalSim, for parallelized, GPU accelerated, and differentiable galaxy image simulations.**

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md) [![Python package](https://github.com/GalSim-developers/JAX-GalSim/actions/workflows/python_package.yaml/badge.svg)](https://github.com/GalSim-developers/JAX-GalSim/actions/workflows/python_package.yaml)

**Disclaimer**: This project is still in an early development phase, **please use the [reference GalSim implementation](https://github.com/GalSim-developers/GalSim) for any scientific applications.**

In fact, we are still thinking about how to name this project, checkout [this poll](https://github.com/GalSim-developers/JAX-GalSim/discussions/2).

## Objective and design

See [design document](https://docs.google.com/document/d/1NalCc_5dc3Z8F4q37y-RsJS_mr9gzvfyANb2PYUpsb4/edit?usp=sharing).

The goal of this library is to reimplement GalSim functionalities in pure JAX to allow for automatic differentiation, GPU acceleration, and batched computations.

**Guiding Principles**:

- Strive to be a drop-in replacement for GalSim, i.e. provide a close match to the GalSim API.
- Each function/feature will be tested against the reference GalSim implementation.
- This package will aim to be a **subset** of GalSim (i.e. only contains functions with a reference GalSim implementation).
- Implementations should be easy to read and understand.
- Code should be pip installable on any machine, no compilation required.
- Any notable differences between the JAX and reference implementations will be clearly documented.

## Contributing

Everyone can contribute to this project, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) document for details.

In short, to interact with the project you can:

- Ask or Answer questions on the [Discussions Q&A](https://github.com/GalSim-developers/JAX-GalSim/discussions/categories/q-a) page
- Report a bug by opening a [GitHub issue](https://github.com/GalSim-developers/JAX-GalSim/issues)
- Open a [GitHub issue](https://github.com/GalSim-developers/JAX-GalSim/issues) or [Discussions](https://github.com/GalSim-developers/JAX-GalSim/discussions) to ask for feedback on a planned contribution.
- Submit a Pull Request to contribute to the code.

Issues marked with _contributions welcome_ or _good first issue_ are particularly good places to start. These are great ways to learn more
about the inner workings of GalSim and how to code in JAX.

## Current GalSim capabilities coverage

0%
