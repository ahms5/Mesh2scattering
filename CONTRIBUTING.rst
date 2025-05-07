.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given. The following helps you to start
contributing specifically to mesh2scattering. Please also consider the
`general contributing guidelines`_ for example regarding the style
of code and documentation and some helpful hints.

Types of Contributions
----------------------

Report Bugs or Suggest Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The best place for this is https://github.com/ahms5/mesh2scattering/issues.

Fix Bugs or Implement Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Look through https://github.com/ahms5/mesh2scattering/issues for bugs or feature request
and contact us or comment if you are interested in implementing.

Write Documentation
~~~~~~~~~~~~~~~~~~~

mesh2scattering could always use more documentation, whether as part of the
official mesh2scattering docs, in docstrings, or even on the web in blog posts,
articles, and such.

Get Started!
------------

Ready to contribute? Here's how to set up `mesh2scattering` for local development using the command-line interface. Note that several alternative user interfaces exist, e.g., the Git GUI, `GitHub Desktop <https://desktop.github.com/>`_, extensions in `Visual Studio Code <https://code.visualstudio.com/>`_ ...

1. `Fork <https://docs.github.com/en/get-started/quickstart/fork-a-repo/>`_ the `mesh2scattering` repo on GitHub.
2. Clone your fork locally and cd into the mesh2scattering directory::

    $ git clone https://github.com/YOUR_USERNAME/mesh2scattering.git
    $ cd mesh2scattering

3. Install your local copy into a virtualenv. Assuming you have Anaconda or Miniconda installed, this is how you set up your fork for local development::

    $ conda create --name mesh2scattering python
    $ conda activate mesh2scattering
    $ pip install -e ".[dev]"

4. Create a branch for local development. Indicate the intention of your branch in its respective name (i.e. `feature/branch-name` or `bugfix/branch-name`)::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass ruff and the
   tests::

    $ ruff check
    $ pytest

   ruff must pass without any warnings for `./mesh2scattering` and `./tests` using the default or a stricter configuration. Ruff ignores a couple of PEP Errors (see `./pyproject.toml`). If necessary, adjust your linting configuration in your IDE accordingly.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request on the develop branch through the GitHub website.


Manual build
------------

for Linux
~~~~~~~~~

Install the C++ build essentials by running::

    $ sudo apt-get install build-essential

Go into the NumCalc directory by running::

    $ cd path/to/your/Mesh2scattering/mesh2scattering/numcalc/src

Compile NumCalc by running make. It is now located in the folder ``mesh2scattering/numcalc/bin``::

    $ make

Copy NumCalc to a folder in your program path: in the same directory run::

    $ sudo cp NumCalc /usr/local/bin/

Now NumCalc can be used by running NumCalc (don't do this yet).

for MacOS
~~~~~~~~~

Install the C++ build essentials by installing ``xcode``
Go into the ``numcalc/src`` directory by running::

    $ cd path/to/your/Mesh2scattering/mesh2scattering/numcalc/src

Compile NumCalc by running ``make``. It is now located in the folder ``mesh2scattering/numcalc/bin``::

    $ make

Now NumCalc can be used by running ``path/to/mesh2scattering/numcalc/bin/NumCalc`` (don't do this yet)::
    
    $ path/to/mesh2scattering/numcalc/bin/NumCalc

for Windows
~~~~~~~~~~~

This guide is taken from the `Mesh2HRTF documentation`_.

If you need to re-compile NumCalc.exe (compiled binaries are obsolete), then it can be compiled by any MinGW 64bit compiler. This method uses MSYS2 version of MinGW64 compiler, but there are also other valid approaches to use MinGW compiler variants on Windows. 

Go to `https://www.msys2.org <https://www.msys2.org>`_ and get the MSYS2 by following the excellent Installation instructions on the front-page.
First run the Windows Installer using simple GUI.
Then open MSYS2 MSYS.exe and update everything by pasting-in the following commands (answer “Y” or just hit Enter key on all prompts). Tip: re-run all update commands more than once until it says that there is nothing more to update.::

    $ pacman -Syu (Update the package database and base packages.)
    $ pacman -Su (Update the rest of the base packages)

Install the MinGW with GCC needed for NumCalc compiling::

    $ pacman -S --needed base-devel mingw-w64-x86_64-toolchain

Close “MSYS2 MSYS.exe”

Open MSYS2 MinGW 64-bit.exe that you just installed and proceed to compile NumCalc.exe:

Use the following commands to change to the “Source” folder from which the compilation needs to be started. (To be on the safe side, do not use paths with spaces and non-standard characters)::

    $ cd C:
    $ cd path/to/your/mesh2hrtf-git/mesh2hrtf/NumCalc/src

Compile (using the normal Linux make command)::
    
    $ make

You should now have “NumCalc.exe” ready in the "bin" folder (path/to/your/mesh2hrtf-git/mesh2hrtf/NumCalc/bin), but there are a few steps left before running the simulation. Close “MSYS MinGW 64-bit.exe”

Fixing the DLLs to make your NumCalc.exe work. (Note, it is somehow possible to include all the necessary DLLs into .exe file during compilation, but this method works too):
Copy your “NumCalc.exe” out to another folder before running it (recommended).
Try running “NumCalc.exe”. Most likely you will get “___.dll not found” error which means that it needs some .dll files in the same folder as NumCalc.exe to run.
You can find all the missing .dll files in the C:\msys64\mingw64\bin folder. Just copy the files with the right name next to the “NumCalc.exe”. Most likely the needed DLLs are: “libgcc_s_seh-1.dll”, “libstdc++-6.dll”, “libwinpthread-1.dll”.
If you run “NumCalc.exe” and right after opening it closes without errors and you see 2 new folders and a new “NC.out” file then it means it is working and ready to be used.

Fully compiled and ready to run NumCalc instance for running on Windows includes see example here::

    "NumCalc.exe"

(most likely) 3 different necessary .dll files

Optional, just for testing - "- run_NumCalc_instance.bat"

.. _general contributing guidelines: https://pyfar-gallery.readthedocs.io/en/latest/contribute/index.html

.. _Mesh2HRTF documentation: https://github.com/Any2HRTF/Mesh2HRTF/wiki/Installation_2#compiling-numcalc-on-windows-using-msys2