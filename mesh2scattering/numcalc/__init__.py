"""
Contains functions for managing numerical calculations.
"""
from .numcalc import (
    build_or_fetch_numcalc,
    manage_numcalc,
    remove_outputs,
    read_ram_estimates,
    calc_and_read_ram,
    )


__all__ = [
    'build_or_fetch_numcalc',
    'manage_numcalc',
    'remove_outputs',
    'read_ram_estimates',
    'calc_and_read_ram',
    ]
