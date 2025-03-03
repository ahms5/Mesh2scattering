# Mesh2scattering

[![PyPI version](https://badge.fury.io/py/mesh2scattering.svg)](https://badge.fury.io/py/mesh2scattering)
[![Documentation Status](https://readthedocs.org/projects/mesh2scattering/badge/?version=latest)](https://mesh2scattering.readthedocs.io/en/latest/?badge=latest)
[![CircleCI](https://circleci.com/gh/ahms5/mesh2scattering.svg?style=shield)](https://circleci.com/gh/ahms5/mesh2scattering)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pyfar/gallery/main?labpath=docs/gallery/interactive/pyfar_introduction.ipynb)

Mesh2scattering is based on [Mesh2HRTF](https://github.com/Any2HRTF/Mesh2HRTF) and is an open-source project aiming an easy-to-use software package for the numerical calculation of scattering pattern and scattering and diffusion coefficients of any surface. In a nutshell, Mesh2scattering consists of three parts:

- input: prepares geometrical data and acoustic parameters for the simulation,
- numcalc: based on the input from ``input``, it calculates the corresponding sound field
- output: processes the output from NumCalc to scattering pattern.
- process: processes the output to scattering and/or diffusion coefficients.
- utils: helping functions.

Please notice that this project does not support HRTF post processing, use [Mesh2HRTF](https://github.com/Any2HRTF/Mesh2HRTF) instead.

## Getting Started

Check out the examples folder for a tour of the most important mesh2scattering
functionality and [read the docs](https://mesh2scattering.readthedocs.io/en/latest) for the complete documentation.

## Installation

Use pip to install mesh2scattering

```bash
pip install mesh2scattering
```

(Requires Python 3.9 or higher)

For Windows the exe is downloaded automatically.
For Linux and MacOS NumCalc is build automatically, note that this requires
the ``build-essential`` on Linux and ``xcode`` on mac.
In the [contributing guidelines](https://mesh2scattering.readthedocs.io/en/stable/contributing.html), you will find a complete Instructions for
manual building.

## Contributing

Check out the [contributing guidelines](https://mesh2scattering.readthedocs.io/en/stable/contributing.html).
