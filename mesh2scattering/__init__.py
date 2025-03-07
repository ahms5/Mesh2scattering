# -*- coding: utf-8 -*-

"""Top-level package for mesh2scattering."""

__author__ = """The Mesh2hrtf and mesh2scattering developers"""
__email__ = ''
__version__ = '1.0.0'

from . import input  # noqa: A004
from . import numcalc
from . import output
from . import process
from . import utils

__all__ = [
    'input',
    'numcalc',
    'output',
    'process',
    'utils',
    ]
