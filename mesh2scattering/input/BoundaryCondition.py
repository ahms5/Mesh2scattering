"""Defines a boundary condition for a mesh, e.g material property.

Frequency dependent boundary conditions can only be specified for
ADMI and IMPE but not for PRES and VELO.
"""
import numpy as np
import pyfar as pf
from enum import Enum


class BoundaryConditionType(Enum):
    """Defines the type of the Boundary condition.
    """

    VELO = "VELO"
    """velocity boundary condition."""

    PRES = "PRES"
    """pressure boundary condition."""

    ADMI = "ADMI"
    """admittance boundary condition."""

    IMPE = "IMPE"
    """impedance  boundary condition."""


class BoundaryCondition:
    """Defines a boundary condition for a mesh, e.g material property.

    NOTE: Frequency dependent boundary conditions can only be specified for
    ADMI and IMPE but not for PRES and VELO

    Parameters
    ----------
    values : pf.FrequencyData
        Defines the boundary condition values with its frequencies.
    kind : str
        the kind of boundary condition.
    comment : str
        A comment that is written to the beginning of the material file.
    """

    _values: pf.FrequencyData = None
    _kind: BoundaryConditionType = None
    _comment: str = None

    def __init__(
            self,
            values: pf.FrequencyData,
            kind: BoundaryConditionType,
            comment: str=""):
        """Initialize the Material object.

        Parameters
        ----------
        values : pf.FrequencyData, number
            Defines the boundary condition values with its frequencies.
        kind : BoundaryConditionType
            the kind of boundary condition.
        comment : str
            A comment that is written to the beginning of the material file.
        """
        self.kind = kind
        if kind in (BoundaryConditionType.PRES, BoundaryConditionType.VELO):
            if not (np.isclose(
                    values.frequencies, 0).all() or values.n_bins==1):
                raise ValueError(
                    "Frequency dependent boundary conditions can only be "
                    "specified for ADMI and IMPE but not for PRES and VELO.")
        self.values = values
        self.comment = comment

    @property
    def values(self):
        """Defines the boundary condition values with its frequencies.

        Returns
        -------
        pf.FrequencyData
            The boundary condition values with its frequencies.
        """
        return self._values

    @values.setter
    def values(self, values):
        if not isinstance(values, pf.FrequencyData):
            raise ValueError("values must be a pyfar.FrequencyData object.")
        self._values = values

    @property
    def kind(self):
        """Defines the kind of boundary condition.

        - ``BoundaryConditionType.PRES``
            A pressure boundary condition can be used to force a certain
            pressure on the boundary of the mesh. E.g., a pressure of 0 would
            define a sound soft boundary.
            Cannot be frequency dependent, this means frequency must be 0.
        - ``BoundaryConditionType.VELO``
            A velocity boundary condition can be used to force a certain
            velocity on the boundary of the mesh. E.g., a velocity of 0 would
            define a sound hard boundary.
            Cannot be frequency dependent, this means frequency must be 0.
        - ``BoundaryConditionType.ADMI``
            A normalized admittance boundary condition can be used to define
            arbitrary boundaries. The admittance must be normalized, i.e.,
            admittance/(rho*c) has to be provided, which rho being the density
            of air in kg/m**3 and c the speed of sound in m/s.
        - ``BoundaryConditionType.IMPE``
            A normalized impedance boundary condition can be used to define
            arbitrary boundaries. The impedance must be normalized, i.e.,
            impedance/(rho*c) has to be provided, which rho being the density
            of air in kg/m**3 and c the speed of sound in m/s.

        Returns
        -------
        BoundaryConditionType
            kind of boundary condition
        """
        return self._kind

    @kind.setter
    def kind(self, kind):
        if not isinstance(kind, BoundaryConditionType):
            raise ValueError("kind must be a BoundaryConditionType.")
        self._kind = kind

    @property
    def kind_str(self):
        """Get and set the kind of boundary condition as a string.

        - ``'PRES'``
            A pressure boundary condition can be used to force a certain
            pressure on the boundary of the mesh. E.g., a pressure of 0 would
            define a sound soft boundary.
            Cannot be frequency dependent, this means frequency must be 0.
        - ``'VELO'``
            A velocity boundary condition can be used to force a certain
            velocity on the boundary of the mesh. E.g., a velocity of 0 would
            define a sound hard boundary.
            Cannot be frequency dependent, this means frequency must be 0.
        - ``'ADMI'``
            A normalized admittance boundary condition can be used to define
            arbitrary boundaries. The admittance must be normalized, i.e.,
            admittance/(rho*c) has to be provided, which rho being the density
            of air in kg/m**3 and c the speed of sound in m/s.
        - ``'IMPE'``
            A normalized impedance boundary condition can be used to define
            arbitrary boundaries. The impedance must be normalized, i.e.,
            impedance/(rho*c) has to be provided, which rho being the density
            of air in kg/m**3 and c the speed of sound in m/s.

        Returns
        -------
        str
            The kind of boundary condition.
        """
        return self._kind.value

    @property
    def comment(self):
        """A comment that is written to the beginning of the material file.

        Returns
        -------
        str
            A comment for the material.
        """
        return self._comment

    @comment.setter
    def comment(self, comment):
        if not isinstance(comment, str):
            raise ValueError("comment must be a string.")
        self._comment = comment



