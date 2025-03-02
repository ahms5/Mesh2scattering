"""
This module provides functions to write input files for NumCalc.
"""
import os
import numpy as np
import pyfar as pf
from scipy.spatial import Delaunay, ConvexHull
import trimesh
import json
import datetime
import mesh2scattering as m2s
from packaging import version
from pathlib import Path
from enum import Enum

class SurfaceType(Enum):
    """Defines the type of a sample mesh.

    Can be a trimesh object or a path to a stl file.
    """

    PERIODIC = "Periodic"
    STOCHASTIC = "Stochastic"

class SampleShape(Enum):
    """Defines the shape of a sample mesh.

    Can be round or square.
    """

    ROUND = "Round"
    SQUARE = "Square"


class SurfaceDescription():
    """Initializes the SurfaceDescription object.

    Parameters
    ----------
    structural_wavelength_x : float, optional
        structural wavelength in x direction, by default 0.
    structural_wavelength_y : float, optional
        structural wavelength in y direction, by default 0.
    surface_type : SurfaceType, optional
        surface type, by default SurfaceType.PERIODIC.
    model_scale : float, optional
        model scale, by default 1.
    symmetry_azimuth : list, optional
        azimuth symmetry, by default [].
    symmetry_rotational : bool, optional
        rotational symmetry, by default False.
    comment : str, optional
        comment, by default "".
    """

    _structural_wavelength_x: float = 0
    _structural_wavelength_y: float = 0
    _model_scale: float = 1
    _symmetry_azimuth: list = []
    _symmetry_rotational: bool = False
    _surface_type: SurfaceType = SurfaceType.PERIODIC
    _comment: str = ""

    def __init__(
            self,
            structural_wavelength_x: float=0,
            structural_wavelength_y: float=0,
            surface_type: SurfaceType=SurfaceType.PERIODIC,
            model_scale: float=1,
            symmetry_azimuth: list=[],
            symmetry_rotational: bool=False,
            comment: str="") -> None:
        """Initializes the SurfaceDescription object.

        Parameters
        ----------
        structural_wavelength_x : float, optional
            structural wavelength in x direction, by default 0.
        structural_wavelength_y : float, optional
            structural wavelength in y direction, by default 0.
        surface_type : SurfaceType, optional
            surface type, by default SurfaceType.PERIODIC.
        model_scale : float, optional
            model scale, by default 1.
        symmetry_azimuth : list, optional
            azimuth symmetry, by default [].
        symmetry_rotational : bool, optional
            rotational symmetry, by default False.
        comment : str, optional
            comment, by default "".

        Returns
        -------
        SurfaceDescription
            surface description object.
        """
        if not isinstance(structural_wavelength_x, (int, float)):
            raise ValueError("structural_wavelength_x must be a float.")
        if not isinstance(structural_wavelength_y, (int, float)):
            raise ValueError("structural_wavelength_y must be a float.")
        if not isinstance(model_scale, (int, float)):
            raise ValueError("model_scale must be a float.")
        if not isinstance(symmetry_azimuth, list):
            raise ValueError("symmetry_azimuth must be a list.")
        if not isinstance(symmetry_rotational, bool):
            raise ValueError("symmetry_rotational must be a bool.")
        if not isinstance(comment, str):
            raise ValueError("comment must be a string.")
        if not isinstance(surface_type, SurfaceType):
            raise ValueError("surface_type must be a SurfaceType.")

        self._structural_wavelength_x = structural_wavelength_x
        self._structural_wavelength_y = structural_wavelength_y
        self._model_scale = model_scale
        self._symmetry_azimuth = symmetry_azimuth
        self._symmetry_rotational = symmetry_rotational
        self._comment = comment
        self._surface_type = surface_type


    @property
    def structural_wavelength_x(self):
        """Defines the structural wavelength in x direction.

        Returns
        -------
        float
            The structural wavelength in x direction.
        """
        return self._structural_wavelength_x

    @property
    def structural_wavelength_y(self):
        """Defines the structural wavelength in y direction.

        Returns
        -------
        float
            The structural wavelength in y direction.
        """
        return self._structural_wavelength_y

    @property
    def surface_type(self):
        """Defines the surface type.

        Returns
        -------
        SurfaceType
            The surface type.
        """
        return self._surface_type

    @property
    def model_scale(self):
        """Defines the model scale.

        Returns
        -------
        float
            The model scale.
        """
        return self._model_scale

    @property
    def symmetry_azimuth(self):
        """Defines the azimuth symmetry.

        Returns
        -------
        list
            The azimuth symmetry.
        """
        return self._symmetry_azimuth

    @property
    def symmetry_rotational(self):
        """Defines the rotational symmetry.

        Returns
        -------
        bool
            The rotational symmetry.
        """
        return self._symmetry_rotational

    @property
    def comment(self):
        """Defines the comment.

        Returns
        -------
        str
            The comment.
        """
        return self._comment


class SampleMesh():
    _mesh:trimesh.Trimesh = None
    _surface_description: SurfaceDescription = None
    _sample_diameter: float = 0.8
    _sample_shape: SampleShape = SampleShape.ROUND

    def __init__(
            self, mesh: trimesh.Trimesh,
            surface_description: SurfaceDescription,
            sample_diameter: float=0.8,
            sample_shape: SampleShape=SampleShape.ROUND) -> None:
        """Initializes the SampleMesh object.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            trimesh object representing the sample mesh.
        surface_description : SurfaceDescription
            surface description of the sample mesh.
        sample_diameter : float, optional
            diameter of the sample, by default 0.8
        sample_shape : str, optional
            shape of the sample, by default 'round'

        Returns
        -------
        SampleMesh
            sample mesh object.
        """
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("mesh must be a trimesh.Trimesh object.")
        if not isinstance(sample_diameter, (int, float)):
            raise ValueError("sample_diameter must be a float or int.")
        if not isinstance(sample_shape, SampleShape):
            raise ValueError("sample_shape must be a SampleShape.")
        if not isinstance(surface_description, SurfaceDescription):
            raise ValueError(
                "surface_description must be a SurfaceDescription object.")

        self._mesh = mesh
        self._surface_description = surface_description
        self._sample_diameter = sample_diameter
        self._sample_shape = sample_shape

    @property
    def mesh(self):
        """Defines the sample mesh.

        Returns
        -------
        trimesh.Trimesh
            The sample mesh.
        """
        return self._mesh


    @property
    def sample_diameter(self):
        """Defines the diameter of the sample.

        Returns
        -------
        float
            The diameter of the sample.
        """
        return self._sample_diameter



def write_scattering_project(
        project_path, frequencies,
        
        receiver_coords,
        # sources
        source_coords,
        sourceType,
        # sample
        sample_path,
        structural_wavelength_x=0, structural_wavelength_y=0,
        model_scale=1, symmetry_azimuth=[90, 180],
        symmetry_rotational=False, sample_diameter=0.8,

        speed_of_sound='346.18',
        density_of_medium='1.1839',
        sample_offset_z=0):

    if not os.path.isdir(project_path):
        os.mkdir(project_path)

    frequencyStepSize = 0
    title = 'scattering coefficient Sample'
    method = 'ML-FMM BEM'
    project_path_sample = os.path.join(project_path, 'sample')
    write_project(
        project_path_sample, title, frequencies, frequencyStepSize,
        sample_path,
        receiver_coords, source_coords, sourceType,
        method=method, materialSearchPaths=None,
        speedOfSound=speed_of_sound,
        densityOfMedium=density_of_medium, materials=None, 
        sample_offset_z=sample_offset_z)

    source_list = [list(i) for i in list(source_coords.cartesian)]
    if source_coords.weights is None:
        raise ValueError('sources have to contain weights.')
    source_weights = list(source_coords.weights)

    if isinstance(receiver_coords, pf.Coordinates):
        if receiver_coords.weights is None:
            raise ValueError('receivers have to contain weights.')
        receiver_list = [list(i) for i in list(receiver_coords.get_cart())]
        receiver_weights = list(receiver_coords.weights)
    else:
        for coord in receiver_coords:
            if coord.weights is None:
                raise ValueError('all receivers have to contain weights.')
        receiver_list = [
            r.cartesian.flatten().tolist() for rs in receiver_coords for r in rs]
        receiver_weights = [r.weights for rs in receiver_coords for r in rs]

    title = 'scattering pattern'
    frequencies = np.array(frequencies, dtype=float)
    parameters = {
        # project Info
        'project_title': 'scattering pattern',
        'mesh2scattering_path': m2s.utils.program_root(),
        'mesh2scattering_version': m2s.__version__,
        'bem_version': 'ML-FMM BEM',
        # Constants
        'speed_of_sound': float(speed_of_sound),
        'density_of_medium': float(density_of_medium),
        # Sample Information, post processing
        'structural_wavelength': structural_wavelength_x,
        'structural_wavelength_x': structural_wavelength_x,
        'structural_wavelength_y': structural_wavelength_y,
        'model_scale': model_scale,
        'sample_diameter': sample_diameter,
        # symmetry information
        'symmetry_azimuth': symmetry_azimuth,
        'symmetry_rotational': symmetry_rotational,
        # frequencies
        'num_frequencies': len(frequencies),
        'min_frequency': frequencies[0],
        'max_frequency': frequencies[-1],
        'frequencies': list(frequencies),
        'min_lbyl': lbyls[0],
        'max_lbyl': lbyls[-1],
        'lbyls': list(lbyls),
        # Source definition
        'source_type': sourceType,
        'sources_num': len(source_list),
        'sources': source_list,
        'sources_weights': source_weights,
        # Receiver definition
        'receivers_num': len(receiver_list),
        'receivers': receiver_list,
        'receiver_weights': receiver_weights,

    }
    with open(os.path.join(project_path, 'parameters.json'), 'w') as file:
        json.dump(parameters, file, indent=4)


def write_project(
        project_path, title, frequencies, mesh_path,
        evaluationPoints, sourcePositions,
        sourceType='Point source', method='ML-FMM BEM',
        materialSearchPaths=None, speedOfSound='346.18',
        densityOfMedium='1.1839', materials=None, sample_offset_z=0):

    # programPath = utils.program_root()
    # defaultPath = os.path.join(
    #     programPath, 'Mesh2Input', 'Materials', 'Data')
    # if materialSearchPaths is None:
    #     materialSearchPaths = defaultPath
    # else:
    #     materialSearchPaths += f";  {defaultPath}"

    # create folders
    if not os.path.isdir(project_path):
        os.mkdir(project_path)
    if not os.path.isdir(os.path.join(project_path, 'ObjectMeshes')):
        os.mkdir(os.path.join(project_path, 'ObjectMeshes'))
    if not os.path.isdir(os.path.join(project_path, 'NumCalc')):
        os.mkdir(os.path.join(project_path, 'NumCalc'))
    if not os.path.isdir(os.path.join(project_path, 'EvaluationGrids')):
        os.mkdir(os.path.join(project_path, 'EvaluationGrids'))

    # copy stl file
    mesh = trimesh.load(mesh_path)
    mesh.vertices[:, 2] += sample_offset_z
    print(
        f'mesh x min {np.min(mesh.vertices[:, 0]):.5f} '
        f'max {np.max(mesh.vertices[:, 0]):.5f}')
    print(
        f'mesh y min {np.min(mesh.vertices[:, 1]):.5f} '
        f'max {np.max(mesh.vertices[:, 1]):.5f}')
    print(
        f'mesh z min {np.min(mesh.vertices[:, 2]):.5f} '
        f'max {np.max(mesh.vertices[:, 2]):.5f}')
    path = os.path.join(project_path, 'ObjectMeshes', 'Reference')
    write_mesh(mesh.vertices, mesh.faces, path, start=0)
    mesh.export(os.path.join(
        project_path, 'ObjectMeshes', mesh_path.split(os.sep)[-1]))

    # write evaluation grid
    grids = []
    if isinstance(evaluationPoints, list):
        start = 200000
        for i in range(len(evaluationPoints)):
            write_evaluation_grid(
                evaluationPoints[i],
                os.path.join(project_path, 'EvaluationGrids', f'grid_{i}'),
                start=start)
            start+= evaluationPoints[i].csize
            grids.append(f'grid_{i}')

    else:
        write_evaluation_grid(
            evaluationPoints,
            os.path.join(project_path, 'EvaluationGrids', 'grid'))
        grids.append('grid')

    # Write NumCalc input files for all sources (NC.inp) ----------------------
    _write_nc_inp(
        project_path, version.parse(m2s.__version__), title, speedOfSound,
        densityOfMedium, frequencies, grids, materials, method, sourceType,
        sourcePositions, len(mesh.faces), len(mesh.vertices))


def write_mesh(vertices, faces, path, start=200000):
    """
    Write mesh to Mesh2HRTF input format.

    Mesh2HRTF meshes consist of two text files Nodes.txt and Elements.txt.
    The Nodes.txt file contains the coordinates of the vertices and the
    Elements.txt file contains the indices of the vertices that form the faces
    of the mesh.

    Parameters
    ----------
    vertices : pyfar Coordinates, numpy array
        pyfar Coordinates object or 2D numpy array containing the cartesian
        points of the mesh in meter. The array must be of shape (N, 3) with N
        > 2.
    faces : numpy array
        2D numpy array containing the indices of the vertices that form the
        faces of the mesh. The array must be of shape (M, 3) with M > 0.
    path : str
        Path to the directory where the mesh is saved.
    start : int, optional
        The nodes and elements of the mesh are numbered and the first element
        will have the number `start`. In Mesh2HRTF, each Node must have a
        unique number. The nodes/elements of the mesh for which the HRTFs are
        simulated start at 1. Thus `start` must at least be greater than the
        number of nodes/elements in the mesh.

    """

    if vertices.ndim != 2 or vertices.shape[0] < 3 \
            or vertices.shape[1] != 3:
        raise ValueError(
            "vertices must be a 2D array of shape (N, 3) with N > 2")

    # check output directory
    if not os.path.isdir(path):
        os.mkdir(path)

    # write nodes
    N = int(vertices.shape[0])
    start = int(start)

    nodes = f"{N}\n"
    for nn in range(N):
        nodes += (f"{int(start + nn)} "
                  f"{vertices[nn, 0]} "
                  f"{vertices[nn, 1]} "
                  f"{vertices[nn, 2]}\n")

    with open(os.path.join(path, "Nodes.txt"), "w") as f_id:
        f_id.write(nodes)

    # write elements
    N = int(faces.shape[0])
    elements = f"{N}\n"
    for nn in range(N):
        elements += (
            f"{int(start + nn)} "
            f"{faces[nn, 0] + start} "
            f"{faces[nn, 1] + start} "
            f"{faces[nn, 2] + start} "
            "0 0 0\n")

    with open(os.path.join(path, "Elements.txt"), "w") as f_id:
        f_id.write(elements)


def _write_nc_inp(
        project_path: str, version: str, project_title: str,
        speed_of_sound: float, density_of_medium: float,
        frequencies: np.ndarray,
        evaluation_grid_names: list[str],
        source_type: str, source_positions: pf.Coordinates,
        n_mesh_elements: int, n_mesh_nodes: int, method:str='ML-FMM BEM',
        materials: dict=None):
    """Write NC.inp file that is read by NumCalc to start the simulation.

    The file format is documented at:
    https://github.com/Any2HRTF/Mesh2HRTF/wiki/Structure_of_NC.inp

    Parameters
    ----------
    project_path : str, Path
        root path of the NumCalc project.
    version : str
        current version of Mesh2scattering.
    project_title : str
        project title.
    speed_of_sound : float
        Speed of sound in m/s.
    density_of_medium : float
        density of the medium in kg/m^3.
    frequencies : np.ndarray
        frequency vector in Hz for NumCalc.
    evaluation_grid_names : list[str]
        evaluation grid names. Evaluation grids need to be written before the
        NC.inp file.
    source_type : str
        Type of the sound source. Options are 'Point source' or 'Plane wave'.
    source_positions : pf.Coordinates
        source positions in meter.
    n_mesh_elements : int
        number of mesh elements.
    n_mesh_nodes : int
        number of mesh nodes.
    method : str
        solving method for the NumCalc. Options are 'BEM', 'SL-FMM BEM', or
        'ML-FMM BEM'. By default 'ML-FMM BEM' is used.
    materials : dict, None
        _description_
    """
    if not isinstance(source_positions, pf.Coordinates):
        raise ValueError(
            "source_positions must be a pyfar.Coordinates object.")

    # check the BEM method
    if method == 'BEM':
        method_id = 0
    elif method == 'SL-FMM BEM':
        method_id = 1
    elif method == 'ML-FMM BEM':
        method_id = 4
    else:
        ValueError(
            f"Method must be BEM, SL-FMM BEM or ML-FMM BEM but is {method}")

    for i_source in range(source_positions.cshape[0]):

        # create directory
        filepath2 = os.path.join(
            project_path, "NumCalc", f"source_{i_source+1}")
        if not os.path.exists(filepath2):
            os.mkdir(filepath2)

        # write NC.inp
        file = open(os.path.join(filepath2, "NC.inp"), "w",
                    encoding="utf8", newline="\n")
        fw = file.write

        # header --------------------------------------------------------------
        fw("##-------------------------------------------\n")
        fw("## This file was created by mesh2scattering\n")
        fw("## Date: %s\n" % datetime.date.today())
        fw("##-------------------------------------------\n")
        fw("mesh2scattering %s\n" % version)
        fw("##\n")
        fw("%s\n" % project_title)
        fw("##\n")

        # control parameter I (hard coded, not documented) --------------------
        fw("## Controlparameter I\n")
        fw("0 0 0 0 7 0\n")
        fw("##\n")

        # control parameter II ------------------------------------------------
        fw("## Controlparameter II\n")
        fw("1 %d 0.000001 0.00e+00 1 0 0\n" % (
            len(frequencies)))
        fw("##\n")
        fw("## Load Frequency Curve \n")
        fw("0 %d\n" % (len(frequencies)+1))
        fw("0.000000 0.000000e+00 0.0\n")
        for ii in range(len(frequencies)):
            fw("%f %fe+04 0.0\n" % (
                0.000001*(ii+1),
                frequencies[ii] / 10000))
        fw("##\n")

        # main parameters I ---------------------------------------------------
        fw("## 1. Main Parameters I\n")
        numNodes = 0
        numElements = 0
        for evaluationGrid in evaluation_grid_names:
            # read number of nodes
            nodes = open(os.path.join(
                project_path, "EvaluationGrids", evaluationGrid,
                "Nodes.txt"))
            line = nodes.readline()
            numNodes = numNodes+int(line)
            # read number of elements
            elements = open(os.path.join(
                project_path, "EvaluationGrids", evaluationGrid,
                "Elements.txt"))
            line = elements.readline()
            numElements = numElements+int(line)
        fw("2 %d " % (n_mesh_elements+numElements))
        fw("%d 0 " % (n_mesh_nodes+numNodes))
        fw("0")
        fw(" 2 1 %s 0\n" % (method_id))
        fw("##\n")

        # main parameters II --------------------------------------------------
        fw("## 2. Main Parameters II\n")
        if "plane" in source_type:
            fw("1 ")
        else:
            fw("0 ")
        if "ear" in source_type:
            fw("0 ")
        else:
            fw("1 ")
        fw("0 0.0000e+00 0 0 0\n")
        fw("##\n")

        # main parameters III -------------------------------------------------
        fw("## 3. Main Parameters III\n")
        fw("0 0 0 0\n")
        fw("##\n")

        # main parameters IV --------------------------------------------------
        fw("## 4. Main Parameters IV\n")
        fw("%s %se+00 1.0 0.0e+00 0.0 e+00 0.0e+00 0.0e+00\n" % (
            speed_of_sound, density_of_medium))
        fw("##\n")

        # nodes ---------------------------------------------------------------
        fw("NODES\n")
        fw("../../ObjectMeshes/Reference/Nodes.txt\n")
        # write file path of nodes to input file
        for grid in evaluation_grid_names:
            fw(f"../../EvaluationGrids/{grid}/Nodes.txt\n")
        fw("##\n")
        fw("ELEMENTS\n")
        fw("../../ObjectMeshes/Reference/Elements.txt\n")
        # write file path of elements to input file
        for grid in evaluation_grid_names:
            fw(f"../../EvaluationGrids/{grid}/Elements.txt\n")
        fw("##\n")

        # SYMMETRY ------------------------------------------------------------
        fw("# SYMMETRY\n")
        fw("# 0 0 0\n")
        fw("# 0.0000e+00 0.0000e+00 0.0000e+00\n")
        fw("##\n")

        # assign mesh elements to boundary conditions -------------------------
        # (including both, left, right ear)
        fw("BOUNDARY\n")
        # write velocity condition for the ears if using vibrating
        # elements as the sound source
        if "ear" in source_type:
            if i_source == 0 and \
                    source_type in ['Both ears', 'Left ear']:
                tmpEar = 'Left ear'
            else:
                tmpEar = 'Right ear'
            fw(f"# {tmpEar} velocity source\n")
            fw("ELEM %i TO %i VELO 0.1 -1 0.0 -1\n" % (
                materials[tmpEar]["index_start"],
                materials[tmpEar]["index_end"]))
        # remaining conditions defined by frequency curves
        curves = 0
        steps = 0
        if materials is not None:
            for m in materials:
                if materials[m]["path"] is None:
                    continue
                # write information
                fw(f"# Material: {m}\n")
                fw("ELEM %i TO %i %s 1.0 %i 1.0 %i\n" % (
                    materials[m]["index_start"],
                    materials[m]["index_end"],
                    materials[m]["boundary"],
                    curves + 1, curves + 2))
                # update metadata
                steps = max(steps, len(materials[m]["freqs"]))
                curves += 2

        fw("RETU\n")
        fw("##\n")

        # source information: point source and plane wave ---------------------
        if source_type == "Point source":
            fw("POINT SOURCES\n")
        elif source_type == "Plane wave":
            fw("PLANE WAVES\n")
        if source_type in ["Point source", "Plane wave"]:
            fw("0 %s %s %s 0.1 -1 0.0 -1\n" % (
                source_positions.x[i_source], source_positions.y[i_source],
                source_positions.z[i_source]))
        fw("##\n")

        # curves defining boundary conditions of the mesh ---------------------
        if curves > 0:
            fw("CURVES\n")
            # number of curves and maximum number of steps
            fw(f"{curves} {steps}\n")
            curves = 0
            for m in materials:
                if materials[m]["path"] is None:
                    continue
                # write curve for real values
                curves += 1
                fw(f"{curves} {len(materials[m]['freqs'])}\n")
                for f, v in zip(materials[m]['freqs'],
                                materials[m]['real']):
                    fw(f"{f} {v} 0.0\n")
                # write curve for imaginary values
                curves += 1
                fw(f"{curves} {len(materials[m]['freqs'])}\n")
                for f, v in zip(materials[m]['freqs'],
                                materials[m]['imag']):
                    fw(f"{f} {v} 0.0\n")

        else:
            fw("# CURVES\n")
        fw("##\n")

        # post process --------------------------------------------------------
        fw("POST PROCESS\n")
        fw("##\n")
        fw("END\n")
        file.close()



def write_evaluation_grid(
        points, folder_path, start=200000, discard=None):
    """
    Write evaluation grid for use in Mesh2HRTF.

    Mesh2HRTF evaluation grids consist of the two text files Nodes.txt and
    Elements.txt. Evaluations grids are always triangulated.

    Parameters
    ----------
    points : pyfar.Coordinates
        pyfar Coordinates object containing the cartesian
        points of the evaluation grid in meter. The array must be of shape
        (N, 3) with N > 2.
    folder_path : str
        folder path under which the evaluation grid is saved. If the
        folder does not exist, it is created.
    start : int, optional
        The nodes and elements of the evaluation grid are numbered and the
        first element will have the number `start`. In Mesh2HRTF, each Node
        must have a unique number. The nodes/elements of the mesh for which the
        HRTFs are simulated start at 1. Thus `start` must at least be greater
        than the number of nodes/elements in the evaluation grid.
    discard : "x", "y", "z", None optional
        In case all values of the evaluation grid are constant for one
        dimension, this dimension has to be discarded during the
        triangularization. E.g. if all points have a z-value of 0 (or any other
        constant), discarded must be "z". The default ``None`` does not discard
        any dimension.

    Examples
    --------
    Generate a spherical sampling grid with pyfar and write it to the current
    working directory

    .. plot::

        >>> import mesh2scattering as m2s
        >>> import pyfar as pf
        >>>
        >>> points = pf.samplings.sph_lebedev(sh_order=10)
        >>> m2s.input.write_evaluation_grid(
        ...     points, "Lebedev_N10", discard=None)
    """

    if isinstance(points, pf.Coordinates):
        if points.cdim != 1:
            raise ValueError("cdim of pyfar.Coordinates must be 1.")
        points = points.cartesian
    else:
        raise ValueError("points must be a pyfar.Coordinates object.")

    if not isinstance(start, int) or  start < 0:
        raise ValueError("start must be a positive integer.")

    if discard not in (None, "x", "y", "z"):
        raise ValueError("discard must be None, 'x', 'y', or 'z'.")

    # discard dimension
    if discard == "x":
        mask = (1, 2)
    elif discard == "y":
        mask = (0, 2)
    elif discard == "z":
        mask = (0, 1)
    else:
        mask = (0, 1, 2)

    # triangulate
    if discard is None:
        tri = ConvexHull(points[:, mask])
    else:
        tri = Delaunay(points[:, mask])

    # check output directory
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    # write nodes
    N = int(points.shape[0])
    start = int(start)

    nodes = f"{N}\n"
    for nn in range(N):
        nodes += (f"{int(start + nn)} "
                  f"{points[nn, 0]} "
                  f"{points[nn, 1]} "
                  f"{points[nn, 2]}\n")

    with open(os.path.join(folder_path, "Nodes.txt"), "w") as f_id:
        f_id.write(nodes)

    # write elements
    N = int(tri.simplices.shape[0])
    elems = f"{N}\n"
    for nn in range(N):
        elems += (f"{int(start + nn)} "
                  f"{tri.simplices[nn, 0] + start} "
                  f"{tri.simplices[nn, 1] + start} "
                  f"{tri.simplices[nn, 2] + start} "
                  "2 0 1\n")

    with open(os.path.join(folder_path, "Elements.txt"), "w") as f_id:
        f_id.write(elems)
