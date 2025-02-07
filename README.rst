===============
Mesh2scattering
===============
.. image:: https://badge.fury.io/py/mesh2scattering.svg
    :target: https://badge.fury.io/py/mesh2scattering
.. image:: https://readthedocs.org/projects/mesh2scattering/badge/?version=latest
    :target: https://mesh2scattering.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://circleci.com/gh/ahms5/Mesh2scattering.svg?style=shield
    :target: https://circleci.com/gh/ahms5/Mesh2scattering

Mesh2scattering is based on `Mesh2HRTF`_ and is an open-source project aiming an easy-to-use software package for the numerical calculation of scattering pattern and scattering and diffusion coefficients of any surface. In a nutshell, Mesh2scattering consists of three parts:

* input: prepares geometrical data and acoustic parameters for the simulation,
* numcalc: based on the input from ``input``, it calculates the corresponding sound field
* output: processes the output from NumCalc to scattering pattern.
* process: processes the output to scattering and/or diffusion coefficients.
* utils: helping functions.

Please notice that this project does not support HRTF post processing, use `Mesh2HRTF`_ instead.


Getting Started
===============

Check out the examples folder for a tour of the most important mesh2scattering
functionality and `read the docs`_ for the complete documentation. 

Installation
============

Use pip to install mesh2scattering

.. code-block:: console

    $ pip install mesh2scattering
    $ pip install git+https://github.com/pyfar/imkar.git@scattering-freefield

(Requires Python 3.8 or higher)

Note that NumCalc need to be build on Linux and MacOS. For Windows it can be downloaded.

for Linux:
~~~~~~~~~~

* Install the C++ build essentials by running 

.. code-block:: console

    $ sudo apt-get install build-essential

* Go into the NumCalc directory by running

.. code-block:: console

    $ cd path/to/your/Mesh2scattering/mesh2scattering/numcalc/src

* Compile NumCalc by running make. It is now located in the folder ``mesh2scattering/numcalc/bin``

.. code-block:: console

    $ make

* Copy NumCalc to a folder in your program path: in the same directory run

.. code-block:: console

    $ sudo cp NumCalc /usr/local/bin/

* Now NumCalc can be used by running NumCalc (don't do this yet).

for MacOS:
~~~~~~~~~~

* Install the C++ build essentials by installing ``xcode``
* Go into the ``numcalc/src`` directory by running

.. code-block:: console

    $ cd path/to/your/Mesh2scattering/mesh2scattering/numcalc/src

* Compile NumCalc by running ``make``. It is now located in the folder ``mesh2scattering/numcalc/bin``

.. code-block:: console

    $ make

* Now NumCalc can be used by running ``path/to/mesh2scattering/numcalc/bin/NumCalc`` (don't do this yet)

.. code-block:: console

    $ path/to/mesh2scattering/numcalc/bin/NumCalc


for Windows:
~~~~~~~~~~~~

download the executable from the release.


Contributing
============

Refer to the `contribution guidelines`_ for more information.


.. _contribution guidelines: https://github.com/ahms5/mesh2scattering/blob/develop/CONTRIBUTING.rst
.. _Mesh2HRTF: https://github.com/Any2HRTF/Mesh2HRTF
.. _read the docs: https://mesh2scattering.readthedocs.io/en/latest
