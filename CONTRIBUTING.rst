.. highlight:: shell

------------
Contributing
------------

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs and Submit Feedback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The best way to report bugs of send feedback is to open an issue at 
https://github.com/ahms5/Mesh2scattering/pulls.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Fix Bugs or Implement Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" or
"enhancement" is open to whoever wants to implement it. It might be good to
contact us first, to see if anyone is already working on it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

Mesh2scattering could always use more documentation, whether as part of the
official Mesh2scattering docs, in docstrings, or even on the web in blog posts,
articles, and such.

Start Contributing
------------------

Ready to contribute? Here's how to set up `Mesh2scattering` for local development.
Work on the numerical core `NumCalc` and the Python API requires a local copy of Mesh2scattering to 
install the API for local development

1. Fork the `Mesh2scattering` repo on GitHub.
2. Clone your fork locally and cd into the Mesh2scattering directory::

    $ git clone https://github.com/ahms5/Mesh2scattering.git
    $ cd Mesh2scattering

3. Install your local copy into a virtualenv. Assuming you have Anaconda or Miniconda installed, this is how you set up your fork for local development::

    $ conda create --name mesh2scattering python
    $ conda activate mesh2scattering
    $ conda install pip
    $ pip install -e .
    $ pip install -r requirements_dev.txt

4. Create a branch for local development. Indicate the intention of your branch in its respective name (i.e. `feature/branch-name` or `bugfix/branch-name`)::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the
   tests::

    $ flake8 mesh2scattering tests
    $ pytest

   flake8 test must pass without any warnings for `./mesh2scattering` and `./tests` using the default or a stricter configuration. Flake8 ignores `E123/E133, E226` and `E241/E242` by default. If necessary adjust the your flake8 and linting configuration in your IDE accordingly.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request on the develop branch through the GitHub website.


Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring.
3. If checks to not pass, have a look at https://travis-ci.com/pyfar/pyfar/pull_requests for more information.


Testing Guidelines
-----------------------
mesh2scattering uses test-driven development based on `three steps <https://martinfowler.com/bliki/TestDrivenDevelopment.html>`_ and `continuous integration <https://en.wikipedia.org/wiki/Continuous_integration>`_ to test and monitor the code.
In the following, you'll find a guideline. Note: these instructions are not generally applicable outside of mesh2scattering.

- The main tool used for testing is `pytest <https://docs.pytest.org/en/stable/index.html>`_.
- All tests are located in the *tests/* folder.
- Make sure that all important parts of pyfar are covered by the tests. This can be checked using *coverage* (see below).
- In case of mesh2scattering, mainly **state verification** is applied in the tests. This means that the outcome of a function is compared to a desired value (``assert ...``). For more information, it is refered to `Martin Fowler's article <https://martinfowler.com/articles/mocksArentStubs.html.>`_.

Tips
~~~~~~~~~~~
Pytest provides several, sophisticated functionalities which could reduce the effort of implementing tests.

- Similar tests executing the same code with different variables can be `parametrized <https://docs.pytest.org/en/stable/example/parametrize.html>`_. An example is ``test___eq___differInPoints`` in *test_coordinates.py*.

- Run a single test with

    $ pytest tests/test_plot.py::test_line_plots

- Exclude tests (for example the time consuming test of plot) with

    $ pytest -k 'not numcalc'

- Create an html report on the test `coverage <https://coverage.readthedocs.io/en/coverage-5.5/>`_ with

    $ pytest --cov=. --cov-report=html

- Feel free to add more recommendations on useful pytest functionalities here. Consider, that a trade-off between easy implemention and good readability of the tests needs to be found.

Fixtures
~~~~~~~~
"Software test fixtures initialize test functions. They provide a fixed baseline so that tests execute reliably and produce consistent, repeatable, results. Initialization may setup services, state, or other operating environments. These are accessed by test functions through arguments; for each fixture used by a test function there is typically a parameter (named after the fixture) in the test function’s definition." (from https://docs.pytest.org/en/stable/fixture.html)

- All fixtures are implemented in *conftest.py*, which makes them automatically available to all tests. This prevents from implementing redundant, unreliable code in several test files.
- Typical fixtures are mesh2scattering objects with varying properties, stubs as well as functions need for initiliazing tests.
- Define the variables used in the tests only once, either in the test itself or in the definition of the fixture. This assures consistency and prevents from failing tests due to the definition of variables with the same purpose at different positions or in different files.

Have a look at already implemented fixtures in *confest.py*.

**Dummies**

If the objects used in the tests have arbitrary properties, tests are usually better to read, when these objects are initialized within the tests. If the initialization requires several operations or the object has non-arbitrary properties, this is a hint to use a fixture.
Good examples illustrating these two cases are the initializations in *test_signal.py* vs. the sine and impulse signal fixtures in *conftest.py*.


Writing the Documentation
-------------------------

Pyfar follows the `numpy style guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_ for the docstring. A docstring has to consist at least of

- A short and/or extended summary,
- the Parameters section, and
- the Returns section

Optional fields that are often used are

- References,
- Examples, and
- Notes

Here are a few tips to make things run smoothly

- Use the tags ``:py:func:``, ``:py:mod:``, and ``:py:class:`` to reference mesh2scattering functions, modules, and classes: For example ``:py:func:`~mesh2scattering.input.write_scattering_project``` for a link that displays only the function name. For links with custom text use ``:py:mod:`input functions <mesh2scattering.input>```.
- Code snippets and values as well as external modules, classes, functions are marked by double ticks \`\` to appear in mono spaced font, e.g., ``x=3`` or ``pyfar.Signal``.
- Parameters, returns, and attributes are marked by single ticks \` to appear as emphasized text, e.g., *unit*.
- Use ``[#]_`` and ``.. [#]`` to get automatically numbered footnotes.
- Do not use footnotes in the short summary. Only use footnotes in the extended summary if there is a short summary. Otherwise, it messes with the auto-footnotes.
- If a method or class takes or returns pyfar objects for example write ``parameter_name : Signal``. This will create a link to the ``pyfar.Signal`` class.
- Plots can be included in by using the prefix ``.. plot::`` followed by an empty line and an indented block containing the code for the plot. See `pyfar.plot.line.time.py` for examples.

See the `Sphinx homepage <https://www.sphinx-doc.org>`_ for more information.

Building the Documentation
--------------------------

You can build the documentation of your branch using Sphinx by executing the make script inside the docs folder.

.. code-block:: console

    $ cd docs/
    $ make html

After Sphinx finishes you can open the generated html using any browser

.. code-block:: console

    $ docs/_build/index.html

Note that some warnings are only shown the first time you build the
documentation. To show the warnings again use

.. code-block:: console

    $ make clean

before building the documentation.


Deploying
~~~~~~~~~

A reminder for the maintainers on how to deploy.

- Commit all changes to develop
- Update HISTORY.rst in develop
- Check if examples/create_project.ipynb needs to be updated
- Merge develop into master

Switch to main and run::

$ bumpversion patch # possible: major / minor / patch
$ git push --follow-tags

Travis will then deploy to PyPI if tests pass.

- merge main back into develop
