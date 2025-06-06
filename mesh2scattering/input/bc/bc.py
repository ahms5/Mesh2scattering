"""Defines boundary conditions for a mesh, such as material properties.

Only admittance (ADMI) boundary conditions may be frequency dependent;
impedance (IMPE), pressure (PRES), and velocity (VELO) boundary conditions
must remain constant across all frequencies.
"""
import numpy as np
import pyfar as pf
from enum import Enum


class BoundaryConditionType(Enum):
    """Enumeration of possible boundary condition types."""

    velocity = "VELO"
    """
    Velocity boundary condition.

    This boundary condition must remain constant across all frequencies; it
    cannot be frequency dependent.
    A velocity value of 0 defines a sound hard boundary.
    """

    pressure = "PRES"
    """
    Pressure boundary condition.

    This boundary condition cannot be frequency dependent; it must remain
    constant across all frequencies.
    A pressure value of 0 defines a sound soft boundary.
    """

    admittance = "ADMI"
    r"""
    Normalized admittance boundary condition.

    This boundary condition can be frequency dependent.
    NumCalc expects normalized admittance, i.e.,
    :math:`(\rho c)/\text{admittance}`, where :math:`\rho` is the density of
    air and :math:`c` is the speed of sound.

    Normalization is advantageous because it eliminates the need to specify
    the speed of sound and air density.
    """

    impedance = "IMPE"
    r"""
    Normalized impedance boundary condition.

    This boundary condition cannot be frequency dependent; it must remain
    constant across all frequencies. NumCalc expects normalized impedance,
    i.e., :math:`\text{impedance}/(\rho c)`, where :math:`\rho` is the
    density of air and :math:`c` is the speed of sound.

    Normalization is advantageous because it removes the need to specify
    the speed of sound and air density.
    """


class BoundaryCondition:
    """
    Represents a boundary condition for a mesh, such as a material property.

    Notes
    -----
    Only admittance boundary conditions can be frequency dependent.
    Impedance, pressure, and velocity boundary conditions must be constant
    across all frequencies.

    Parameters
    ----------
    values : :py:class:`pyfar.FrequencyData` or float
        The value(s) of the boundary condition. If a float is given, it is
        treated as a constant across all frequencies.
        If a FrequencyData object is provided, it must be frequency dependent
        and is only valid for admittance.
    kind : BoundaryConditionType
        The type of boundary condition.
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
            The boundary condition values as a FrequencyData object if
            frequency dependent, or as a single number otherwise.
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
        """Indicates if the boundary condition depends on frequency.

        Returns
        -------
        bool
            True if the boundary condition is frequency dependent,
            otherwise False.
        """
        if isinstance(self.values, pf.FrequencyData):
            return True
        return False

    @property
    def kind(self):
        """GGet the type of boundary condition as a
        :py:class:`BoundaryConditionType`.

        See :py:class:`BoundaryConditionType` for more information.

        Returns
        -------
        BoundaryConditionType
            The type of boundary condition.
        """
        return self._kind

    @kind.setter
    def kind(self, kind):
        if not isinstance(kind, BoundaryConditionType):
            raise ValueError("kind must be a BoundaryConditionType.")
        self._kind = kind

    @property
    def kind_str(self):
        """Get the boundary condition type as a string.

        - ``'PRES'`` for:py:attr:`BoundaryConditionType.pressure`.
        - ``'VELO'`` for :py:attr:`BoundaryConditionType.velocity`.
        - ``'ADMI'`` for :py:attr:`BoundaryConditionType.admittance`.
        - ``'IMPE'`` for :py:attr:`BoundaryConditionType.impedance`.

        Returns
        -------
        str
            String representation of the boundary condition type.
        """
        return self._kind.value


class BoundaryConditionMapping():
    """Represents the assignment of boundary conditions to mesh faces.

    Parameters
    ----------
    n_mesh_faces : int
        Total number of mesh faces available for boundary condition assignment.
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
        """Return the total number of mesh faces independent of the assignment.

        Returns
        -------
        int
            Total number of mesh faces.
        """
        return self._n_mesh_faces

    def add_boundary_condition(
            self, material: BoundaryCondition,
            first_element: int, last_element: int):
        """Assign a boundary condition to a range of mesh faces.

        Multiple boundary conditions can be assigned.
        An error is raised if any mesh face in the specified range has
        already been assigned a material.

        Parameters
        ----------
        material : BoundaryCondition
            The boundary condition to assign.
        first_element : int
            Index of the first mesh face to assign the boundary condition
            to (starting from 0).
        last_element : int
            Index of the last mesh face to assign the boundary condition
            to (starting from 0).
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
        # check if smaller than n_mesh_faces
        if first_element >= self._n_mesh_faces or \
                last_element >= self._n_mesh_faces:
            raise ValueError(
                "first_element and last_element must be < n_mesh_faces.")
        # check if the same mesh faces are used for different materials
        all_indexes = np.zeros(self._n_mesh_faces, dtype=bool)
        for i in range(len(self._material_mapping)):
            first = self._material_mapping[i][1]
            last = self._material_mapping[i][2]
            all_indexes[first:last + 1] = True
        if np.sum(all_indexes[first_element:last_element + 1]) > 0:
            raise ValueError(
                "The same mesh faces are used for different materials.")

        # Add material to list
        if material in self._material_list:
            current_index = self._material_list.index(material)
        else:
            self._material_list.append(material)
            current_index = len(self._material_list) - 1

        # Add range to mapping
        self._material_mapping.append(
            [current_index, first_element, last_element])

    @property
    def n_frequency_curves(self):
        """Get the total number of frequency curves for all boundary
        conditions.

        Returns
        -------
        int
            Total number of frequency curves present in the material list.
        """
        n_frequency_curves = 0
        for material in self._material_list:
            if material.frequency_dependent:
                n_frequency_curves += 2
        return n_frequency_curves


    def to_nc_out(self):
        """Convert the boundary condition mapping to NumCalc input file format.

        For details, see:
        https://github.com/Any2HRTF/Mesh2HRTF/wiki/Structure_of_NC.inp

        Returns
        -------
        nc_boundary : str
            String containing the boundary condition definitions in NumCalc
            format.
        nc_frequency_curve : str
            String containing the frequency curve definitions for
            frequency-dependent boundary conditions in NumCalc format.
            Returns an empty string if not
            frequency dependent.
        """
        nc_boundary = ''
        n_frequency_curves = self.n_frequency_curves
        # first line
        if n_frequency_curves > 0:
            n_max_bins = 0
            for m in self._material_list:
                if m.frequency_dependent:
                    n_max_bins = max(n_max_bins, m.values.n_bins)
            nc_frequency_curve = f'{n_frequency_curves} {n_max_bins}\n'
        else:
            nc_frequency_curve = ''
        current_curve = 1
        frequency_curves_written = np.zeros(
            (len(self._material_list), 2), dtype=int)-1
        for i in range(len(self._material_mapping)):
            index_bc = self._material_mapping[i][0]
            i_start = self._material_mapping[i][1]
            i_end = self._material_mapping[i][2]

            values = self._material_list[index_bc].values
            if self._material_list[index_bc].frequency_dependent:
                if frequency_curves_written[index_bc, 0]<0:
                    # write frequency curves
                    freq_curve_real = f'{current_curve} {values.n_bins}\n'
                    frequency_curves_written[index_bc, 0] = current_curve
                    current_curve += 1
                    freq_curve_imag = f'{current_curve} {values.n_bins}\n'
                    frequency_curves_written[index_bc, 1] = current_curve
                    current_curve += 1
                    for j in range(values.n_bins):
                        real = values.freq.real[0, j]
                        imag = values.freq.imag[0, j]
                        f = values.frequencies[j]
                        freq_curve_real += f'{f:.6e} {real:.6e} 0.0\n'
                        freq_curve_imag += f'{f:.6e} {imag:.6e} 0.0\n'
                    nc_frequency_curve += freq_curve_real
                    nc_frequency_curve += freq_curve_imag
                curve_real = frequency_curves_written[index_bc, 0]
                curve_imag = frequency_curves_written[index_bc, 1]
                real = 1.
                imag = 1.
            else:
                real = np.real(values)
                imag = np.imag(values)
                curve_real = -1
                curve_imag = -1
            material_kind = self._material_list[index_bc].kind_str
            nc_boundary += (f"ELEM {i_start} TO {i_end} {material_kind} "
                            f"{real} {curve_real} {imag} {curve_imag}\n")
        return nc_boundary, nc_frequency_curve
