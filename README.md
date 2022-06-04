# JAX-GalSim
JAX port of GalSim, for parallelized, GPU accelerated, and differentiable galaxy image simulations.

**Disclaimer**: This project is still in an early development phase, **please use the [reference GalSim implementation](https://github.com/GalSim-developers/GalSim) for any scientific applications.**

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

Everyone can contribute to this project, please refer to the CONTRIBUTING.md document for details. 

In short, to interact with the project you can:
- Ask or Answer questions on the Discussions page
- Report a bug by opening a GitHub issue
- Open a GitHub issue or Discussion to ask for feedback on a planned contribution.
- Submit a Pull Request to contribute to the code.

Issues marked with *contributions welcome* or *good first issue* are particularly good places to start. These are great ways to learn more 
about the inner workings of GalSim and how to code in JAX. 

## Current GalSim capabilities coverage

0%
