"""Defines a boundary condition for a mesh, e.g material property.

Frequency dependent boundary conditions can only be specified for
ADMI, but not for IMPE, PRES and VELO.
"""
import numpy as np
import pyfar as pf
from enum import Enum


class BoundaryConditionType(Enum):
    """Defines the type of the Boundary condition.
    """

    velocity = "VELO"
    """
    velocity boundary condition.

    Cannot be frequency dependent, this means it is
    constant for all frequencies.
    A velocity of 0 would define a sound hard boundary.
    """

    pressure = "PRES"
    """
    pressure boundary condition.

    Cannot be frequency dependent, this means it is
    constant for all frequencies.
    A pressure of 0 would define a sound soft boundary.
    """

    admittance = "ADMI"
    r"""
    normalized admittance boundary condition.

    Can be frequency depended.
    NumCalc expects normalized admittance, i.e.,
    :math:`(\rho c)/\text{admittance}`.
    :math:`\rho` is the density of air and :math:`c` the speed of sound.

    The normalization is beneficial because we dont need to know about the
    speed of sound and density of air in the simulation.
    """

    impedance = "IMPE"
    r"""
    normalized impedance boundary condition.

    Cannot be frequency dependent, this means it is
    constant for all frequencies.
    NumCalc expects normalized impedances, i.e.,
    :math:`\text{impedance}/(\rho c)`.
    :math:`\rho` is the density of air and :math:`c` the speed of sound.

    The normalization is beneficial because we dont need to know about the
    speed of sound and density of air in the simulation.
    """


class BoundaryCondition:
    """Defines a boundary condition for a mesh, e.g material property.

    Note
    ----
    Frequency dependent boundary conditions can only be specified for
    admittance but not for impedance, pressure and velocity.

    Parameters
    ----------
    values : :py:class:`pyfar.FrequencyData`, number
        Defines the boundary condition values. If a number is provided,
        it is
        assumed to be a constant value for all frequencies.
        If a FrequencyData object is provided, it must be frequency
        dependent and can only be used for admittance data.
    kind : BoundaryConditionType
        the kind of boundary condition.
    """

    _values: pf.FrequencyData = None
    _kind: BoundaryConditionType = None

    def __init__(
            self,
            values: pf.FrequencyData,
            kind: BoundaryConditionType,
            ):
        """Initialize the Material object."""
        if isinstance(values, pf.FrequencyData):
            if kind is not BoundaryConditionType.admittance:
                raise ValueError(
                    "Frequency dependent boundary conditions can only be "
                    "specified for ADMI but not for IMPE, PRES and VELO.")
        self.kind = kind
        self.values = values

    @property
    def values(self):
        """Get the boundary condition values.

        Returns
        -------
        :py:class:`pyfar.FrequencyData`, number
            The boundary condition values ad FrequencyData if frequency
            dependent or as single number.
        """
        return self._values

    @values.setter
    def values(self, values):
        if not isinstance(values, pf.FrequencyData):
            try:
                values = float(values)
            except ValueError as e:
                raise ValueError(
                    "values must be a pyfar.FrequencyData "
                    "object or a number.") from e
        self._values = values

    @property
    def frequency_dependent(self):
        """Whether the boundary condition is frequency dependent.

        Returns
        -------
        bool
            True if the boundary condition is frequency dependent, False
            otherwise.
        """
        if isinstance(self.values, pf.FrequencyData):
            return True
        return False

    @property
    def kind(self):
        """Get the kind of boundary condition as a
        :py:class:`BoundaryConditionType`.

        See :py:class:`BoundaryConditionType` for details.

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
        """Get the kind of boundary condition as a string.

        - ``'PRES'``
            if :py:attr:`BoundaryCondition.kind` is
            :py:attr:`BoundaryConditionType.pressure`.
        - ``'VELO'``
            if :py:attr:`BoundaryCondition.kind` is
            :py:attr:`BoundaryConditionType.velocity`.
        - ``'ADMI'``
            if :py:attr:`BoundaryCondition.kind` is
            :py:attr:`BoundaryConditionType.admittance`.
        - ``'IMPE'``
            if :py:attr:`BoundaryCondition.kind` is
            :py:attr:`BoundaryConditionType.impedance`.

        Returns
        -------
        str
            The string representation of the kind of boundary condition.
        """
        return self._kind.value


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
        """Get the number of mesh faces to which the boundary condition
        is applied.

        Returns
        -------
        int
            The number of mesh faces.
        """
        return self._n_mesh_faces

    def add_boundary_condition(
            self, material: BoundaryCondition,
            first_element: int, last_element: int):
        """Add a boundary condition to the mapping.

        Several boundary conditions can be added.
        The testing for consistency is not done here, but done in NumCalc.

        Parameters
        ----------
        material : BoundaryCondition
            The material to add to the mapping.
        first_element : int
            The index of the first mesh face to which the material is applied.
            Counting starts at 0.
        last_element : int
            The index of the last mesh face to which the material is applied.
            Counting starts at 0.
        """
        if not isinstance(material, BoundaryCondition):
            raise ValueError("material must be a BoundaryCondition object.")
        try:
            first_element = int(first_element)
        except ValueError as e:
            raise ValueError("first_element must be an integer.") from e
        try:
            last_element = int(last_element)
        except ValueError as e:
            raise ValueError("last_element must be an integer.") from e
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
            if material.frequency_dependent:
                n_frequency_curves += 2
        return n_frequency_curves


    def to_nc_out(self):
        """Convert the mapping to a string for the nc definition file.

        See https://github.com/Any2HRTF/Mesh2HRTF/wiki/Structure_of_NC.inp
        for mor details.

        Returns
        -------
        nc_boundary : str
            The NumCalc formatted boundary condition string.
        nc_frequency_curve : str
            The NumCalc formatted frequency curves for the boundary condition.
            If it is not frequency dependent, it is an empty.
        """
        nc_boundary = ''
        n_frequency_curves = self.n_frequency_curves
        # first line
        if n_frequency_curves > 0:
            n_max_bins = np.max([m.values.n_bins for m in self._material_list])
            nc_frequency_curve = f'{n_frequency_curves} {n_max_bins}\n'
        else:
            nc_frequency_curve = ''
        current_curve = 1
        for i in range(len(self._material_list)):
            i_start = self._material_mapping[i][0]
            i_end = self._material_mapping[i][1]

            values = self._material_list[i].values
            if self._material_list[i].frequency_dependent:
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
                real = np.real(values)
                imag = np.imag(values)
                curve_real = -1
                curve_imag = -1
            material_kind = self._material_list[i].kind_str
            nc_boundary += (f"ELEM {i_start} TO {i_end} {material_kind} "
                            f"{real} {curve_real} {imag} {curve_imag}\n")
        return nc_boundary, nc_frequency_curve
