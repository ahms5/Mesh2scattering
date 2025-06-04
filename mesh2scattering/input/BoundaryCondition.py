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






class BoundaryConditionMapping():
    """Defines a mapping boundary condition for a mesh.

    Parameters
    ----------
    n_mesh_faces : int
        The number of mesh faces to which the boundary condition is applied.
    """

    _material_list: list[int] = None
    _n_mesh_faces: int = None

    def __init__(self, n_mesh_faces: int):
        if not isinstance(n_mesh_faces, int) or n_mesh_faces <= 0:
            raise ValueError("n_mesh_faces must be a positive integer.")

        self._material_list = []
        self._material_mapping = []
        self._n_mesh_faces = n_mesh_faces


    @property
    def n_mesh_faces(self):
        """The number of mesh faces to which the boundary condition is applied.

        Returns
        -------
        int
            The number of mesh faces.
        """
        return self._n_mesh_faces

    def apply_material(
            self, material: BoundaryCondition,
            first_element: int, last_element: int):
        """Add a material to the mapping.

        Parameters
        ----------
        material : BoundaryCondition
            The material to add to the mapping.
        first_element : int
            The index of the first mesh face to which the material is applied.
            Counting starts at 1.
        last_element : int
            The index of the last mesh face to which the material is applied.
            Counting starts at 1.
        """
        if not isinstance(material, BoundaryCondition):
            raise ValueError("material must be a BoundaryCondition object.")
        if not isinstance(first_element, int):
            raise ValueError("first_element must be an integer.")
        if not isinstance(last_element, int):
            raise ValueError("last_element must be an integer.")
        if first_element < 0 or last_element < 0:
            raise ValueError("first_element and last_element must be >= 0.")
        if first_element > last_element:
            raise ValueError("first_element must be <= last_element.")

        # Add material to list
        self._material_list.append(material)
        # Check if first_element and last_element are within the range
        self._material_mapping.append([first_element, last_element])

    @property
    def n_frequency_curves(self):
        """Get the number of frequency curves in the material list.

        Returns
        -------
        int
            The number of frequency curves.
        """
        n_frequency_curves = 0
        for material in self._material_list:
            if material.values.n_bins > 1 or any(
                    material.values.frequencies > 1e-12):
                n_frequency_curves += 2
        return n_frequency_curves


    def to_nc_out(self):
        """Convert the mapping to a dictionary for output.

        Returns
        -------
        dict
            A dictionary containing the mesh mapping and material list.
        """
        nc_boundary = ''
        n_frequency_curves = self.n_frequency_curves
        n_max_bins = np.max([m.values.n_bins for m in self._material_list])
        # first line
        if n_frequency_curves > 0:
            nc_frequency_curve = f'{n_frequency_curves} {n_max_bins}\n'
        else:
            nc_frequency_curve = ''
        current_curve = 1
        for i in range(len(self._material_list)):
            i_start = self._material_mapping[i][0]
            i_end = self._material_mapping[i][1]

            values = self._material_list[i].values
            if self._material_list[i].values.n_bins > 1:
                freq_curve_real = f'{current_curve} {values.n_bins}\n'
                curve_real = current_curve
                current_curve += 1
                freq_curve_imag = f'{current_curve} {values.n_bins}\n'
                curve_imag = current_curve
                current_curve += 1
                for j in range(values.n_bins):
                    real = values.freq.real[0, j]
                    imag = values.freq.imag[0, j]
                    f = values.frequencies[j]
                    freq_curve_real += f'{f:.6e} {real:.6e} 0.0\n'
                    freq_curve_imag += f'{f:.6e} {imag:.6e} 0.0\n'
                nc_frequency_curve += freq_curve_real
                nc_frequency_curve += freq_curve_imag
                real = 1.
                imag = 1.
            else:
                real = values.freq.real[0, 0]
                imag = values.freq.imag[0, 0]
                curve_real = -1
                curve_imag = -1
            material_kind = self._material_list[i].kind_str
            nc_boundary += (f"ELEM {i_start} TO {i_end} {material_kind} "
                            f"{real} {curve_real} {imag} {curve_imag}\n")
        return nc_boundary, nc_frequency_curve
