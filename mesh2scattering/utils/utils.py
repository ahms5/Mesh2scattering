"""
Contains utility functions for the mesh2scattering package.
"""

import os


def program_root():
    """Return the root directory of the repository as absolute path.

    This function
    relies on the correct setting of the environment variable `REPOSITORY_ROOT`
    which is set during the setup of the utils module.

    Returns
    -------
    root : str
        String containing the root directory
    """
    environ = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(environ, os.pardir))
    return root


def repository_root():
    """Return the root directory of the repository as absolute path.

    This function relies on the correct setting of the environment variable
    `REPOSITORY_ROOT` which is set during the setup of the utils module.

    Returns
    -------
    root : str
        String containing the root directory
    """
    environ = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(environ, os.pardir, os.pardir))
    return root
