"""
In mesh2scattering all boundary conditions are assumed as sound
hard if not specified.

This module provides the boundary condition classes and types to define
custom boundary conditions for each mesh element.

Different types of boundary conditions are supported, see
:py:class:`BoundaryConditionType` for details.
The :py:class:`BoundaryCondition` class is used to define the complex vales
of the boundary conditions.
The :py:class:`BoundaryConditionMapping` class is used to map the boundary
conditions to the mesh elements as defined in the Mesh. The element counting
starts at 0, so the first element has index 0.


Examples
--------
here is a short example of a sound soft boundary condition with a mesh with
100 elements:

.. plot::

    >>> import mesh2scattering.input as m2si
    >>> import pyfar as pf
    >>> bc = m2si.bc.BoundaryCondition(
    >>>     kind=m2si.BoundaryConditionType.pressure,
    >>>     values=pf.FrequencyData(0, 0),
    >>> )
    >>> mapping = m2si.bc.BoundaryConditionMapping(100)
    >>> mapping.add_material(bc, 0, 99)  # apply to all elements
    >>> mesh = m2si.Mesh(elements=100, boundary_conditions=mapping)

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
