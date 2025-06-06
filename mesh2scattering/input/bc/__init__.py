"""
By default, all boundary conditions in mesh2scattering are assumed to be
sound hard unless specified otherwise.

This module provides classes and types to define custom boundary conditions
for individual mesh elements.

Supported boundary condition types are described in
:py:class:`BoundaryConditionType`.
Use the :py:class:`BoundaryCondition` class to specify the complex values of
boundary conditions.
The :py:class:`BoundaryConditionMapping` class maps boundary conditions to
mesh elements as defined in the Mesh, with element indices starting at 0.

Examples
--------
The following example demonstrates how to define a sound soft boundary
condition for a mesh with 100 elements:

.. plot::

    >>> import mesh2scattering.input as m2si
    >>> import pyfar as pf
    >>> bc = m2si.bc.BoundaryCondition(
    ...     kind=m2si.bc.BoundaryConditionType.pressure,
    ...     values=0,
    ... )
    >>> mapping = m2si.bc.BoundaryConditionMapping(100)
    >>> mapping.add_boundary_condition(bc, 0, 99)  # apply to all elements

"""


from .bc import (
    BoundaryConditionType,
    BoundaryCondition,
    BoundaryConditionMapping,
)

__all__ = [
    "BoundaryConditionType",
    "BoundaryCondition",
    "BoundaryConditionMapping",
]
