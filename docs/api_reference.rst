.. _api_reference:

API Reference
=============

In a nutshell, Mesh2scattering consists of five modules:

- ``input``: prepares geometrical data and acoustic parameters for the simulation,
- ``numcalc``: based on the input from ``input``, it calculates the corresponding sound field
- ``output``: processes the output from NumCalc to scattering pattern.
- ``process``: processes the output to scattering and/or diffusion coefficients.
- ``utils``: helping functions.

Modules
-------

.. toctree::
   :maxdepth: 1

   modules/mesh2scattering.input
   modules/mesh2scattering.numcalc
   modules/mesh2scattering.output
   modules/mesh2scattering.process
   modules/mesh2scattering.utils
