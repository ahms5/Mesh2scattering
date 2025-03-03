import mesh2scattering as m2s
import os
import numpy as np
import pyfar as pf
import json
import numpy.testing as npt
import pytest
from mesh2scattering.input.input import write_scattering_project_numcalc
from mesh2scattering.input.SampleMesh import SampleMesh, SurfaceDescription
from mesh2scattering.input.SoundSource import SoundSource, SoundSourceType
from mesh2scattering.input.EvaluationGrid import EvaluationGrid


def test_import():
    from mesh2scattering import input  # noqa: A004
    assert input



@pytest.mark.parametrize('source_type',[
    m2s.input.SoundSourceType.POINT_SOURCE,
    m2s.input.SoundSourceType.PLANE_WAVE,
])
@pytest.mark.parametrize('bem_method', [
    'BEM',
    'SL-FMM BEM',
    'ML-FMM BEM',
])
def test_write_project(source_type, bem_method, tmpdir, simple_mesh):
    project_path = os.path.join(tmpdir, 'project')
    project_title = 'test_project'
    frequencies = np.array([500])
    sound_sources = m2s.input.SoundSource(
        pf.Coordinates(1, 0, 1, weights=1),
        source_type,
        )
    points = pf.samplings.sph_lebedev(sh_order=10)
    evaluation_grid = m2s.input.EvaluationGrid.from_spherical(
        points,
        'example_grid')
    surface_description = m2s.input.SurfaceDescription()
    sample_mesh = m2s.input.SampleMesh(
        simple_mesh,
        surface_description,
        0.01,
        0.8,
        m2s.input.SampleShape.ROUND,
    )
    m2s.input.write_scattering_project_numcalc(
        project_path,
        project_title,
        frequencies,
        sound_sources,
        [evaluation_grid],
        sample_mesh,
        bem_method,
        speed_of_sound=346.18,
        density_of_medium=1.1839,
        )

    # check create project folders
    assert os.path.isdir(os.path.join(
        project_path))
    assert os.path.isdir(os.path.join(
        project_path, 'ObjectMeshes'))
    assert os.path.isdir(os.path.join(
        project_path, 'EvaluationGrids'))
    assert os.path.isdir(os.path.join(
        project_path, 'NumCalc'))

    # check mesh files
    assert os.path.isfile(os.path.join(
        project_path, 'ObjectMeshes', 'Reference.stl'))
    assert os.path.isfile(os.path.join(
        project_path, 'ObjectMeshes', 'Reference', 'Nodes.txt'))
    assert os.path.isfile(os.path.join(
        project_path, 'ObjectMeshes', 'Reference', 'Elements.txt'))

    # check evaluation grid files
    assert os.path.isfile(os.path.join(
        project_path, 'EvaluationGrids',
        evaluation_grid.name, 'Nodes.txt'))
    assert os.path.isfile(os.path.join(
        project_path, 'EvaluationGrids',
        evaluation_grid.name, 'Elements.txt'))

    # check numcalc files
    assert os.path.isfile(os.path.join(
        project_path, 'NumCalc', 'source_1', 'NC.inp'))
    assert not os.path.isfile(os.path.join(
        project_path, 'NumCalc', 'source_2', 'NC.inp'))

    # check parameters.json
    json_path = os.path.join(
        project_path, 'parameters.json')
    assert os.path.isfile(json_path)

    with open(json_path, 'r') as file:
        params = json.load(file)

    assert params['project_title'] == project_title
    assert params['bem_method'] == bem_method

    assert params['speed_of_sound'] == 346.18
    assert params['density_of_medium'] == 1.1839
    surf = surface_description
    assert params['structural_wavelength'] == surf.structural_wavelength_x
    assert params['structural_wavelength_x'] == surf.structural_wavelength_x
    assert params['structural_wavelength_y'] == surf.structural_wavelength_y
    assert params['model_scale'] == surf.model_scale
    assert params['symmetry_azimuth'] == surf.symmetry_azimuth
    assert params['symmetry_rotational'] == surf.symmetry_rotational
    assert params['sample_diameter'] == sample_mesh.sample_diameter

    npt.assert_array_almost_equal(params['frequencies'], frequencies)

    assert params['source_type'] == source_type.value
    npt.assert_array_almost_equal(
        params['sources'], sound_sources.source_coordinates.cartesian)
    npt.assert_array_almost_equal(
        params['sources_weights'], sound_sources.source_coordinates.weights)

    evaluation_grids = [
        np.array(r) for r in params['evaluation_grids_coordinates']]
    npt.assert_array_almost_equal(
        evaluation_grids[0], points.cartesian)
    evaluation_grids_weights = np.array(params['evaluation_grids_weights'])
    npt.assert_array_almost_equal(
        evaluation_grids_weights[0], points.weights)
    assert evaluation_grids_weights[0].shape[0] == evaluation_grids[0].shape[0]


@pytest.mark.parametrize('source_type',[
    m2s.input.SoundSourceType.POINT_SOURCE,
    m2s.input.SoundSourceType.PLANE_WAVE,
])
@pytest.mark.parametrize('bem_method', [
    'BEM',
    'SL-FMM BEM',
    'ML-FMM BEM',
])
def test__write_nc_inp(source_type, bem_method, tmpdir):
    # test write nc inp file vs online documentation
    # https://github.com/Any2HRTF/Mesh2HRTF/wiki/Structure_of_NC.inp
    version = '0.1.0'
    project_path = os.path.join(tmpdir, 'project')
    project_title = 'test_project'
    speed_of_sound = 346.18
    density_of_medium = 1.1839
    frequencies = np.array([500])
    evaluation_grid_names = ['example_grid']
    source_positions = pf.Coordinates(1, 0, 1, weights=1)
    n_mesh_elements = 100
    n_mesh_nodes = 50
    n_grid_elements = 200
    n_grid_nodes = 70
    os.mkdir(project_path)
    os.mkdir(os.path.join(project_path, 'NumCalc'))
    m2s.input.input._write_nc_inp(
        project_path, version, project_title,
        speed_of_sound, density_of_medium,
        frequencies,
        evaluation_grid_names,
        source_type, source_positions,
        n_mesh_elements, n_mesh_nodes, n_grid_elements, n_grid_nodes,
        bem_method)

    # test if files are there
    assert os.path.isdir(os.path.join(project_path, 'NumCalc', 'source_1'))
    assert not os.path.isdir(os.path.join(project_path, 'NumCalc', 'source_2'))
    assert os.path.isfile(os.path.join(
        project_path, 'NumCalc', 'source_1', 'NC.inp'))
    assert not os.path.isfile(os.path.join(
        project_path, 'NumCalc', 'source_2', 'NC.inp'))

    # test if the file is correct
    n_bins = frequencies.size
    NC_path = os.path.join(project_path, 'NumCalc', 'source_1', 'NC.inp')
    with open(NC_path, 'r') as f:
        content = "".join(f.readlines())

    # Controlparameter I [%d]
    assert (
        '## Controlparameter I\n'
        '0\n') in content

    # Controlparameter II [%d %d %f %f]
    assert (
        '## Controlparameter II\n'
        f'1 {n_bins} 0.000001 0.00e+00 1 0 0\n') in content

    # check Frequency Curve
    frequency_curve = (
        '## Load Frequency Curve \n'
        f'0 {n_bins+1}\n'
        '0.000000 0.000000e+00 0.0\n')
    frequency_curve += ''.join([
        f'0.{(i+1):06d} {(frequencies[i]/10000):04f}e+04 0.0\n' \
            for i in range(n_bins)])
    assert frequency_curve in content

    # Main Parameters I [%d %d %d %d %d %d %d %d]
    bem_method_id = 0 if bem_method == 'BEM' else 1 if (
        bem_method == 'SL-FMM BEM') else 4
    n_nodes = n_mesh_nodes + n_grid_nodes
    n_elements = n_mesh_elements + n_grid_elements
    main_1 = (
        '## 1. Main Parameters I\n'
        f'2 {n_elements} {n_nodes} 0 0 2 1 {bem_method_id} 0\n')
    assert main_1 in content

    # MainParameters II [%d %d %d %d %d]
    n_plane_waves = source_positions.csize if (
        source_type == m2s.input.SoundSourceType.PLANE_WAVE) else 0
    n_point_sources = source_positions.csize if (
        source_type == m2s.input.SoundSourceType.POINT_SOURCE) else 0
    main_2 = (
        '## 2. Main Parameters II\n'
        f'{1 if n_plane_waves>0 else 0} {1 if n_point_sources>0 else 0} '
        '0 0.0000e+00 0 0\n')
    assert main_2 in content

    #Main Parameters III [%d]
    main_3 = (
        '## 3. Main Parameters III\n'
        '0\n')
    assert main_3 in content

    #Main Parameters IV [%f %f %f]
    main_4 = (
        '## 4. Main Parameters IV\n'
        f'{speed_of_sound} {density_of_medium} 1.0\n')
    assert main_4 in content

    #NODES [%s ... %s]
    nodes = (
        'NODES\n'
        '../../ObjectMeshes/Reference/Nodes.txt\n')
    nodes += ''.join([
        f'../../EvaluationGrids/{grid}/Nodes.txt\n' \
            for grid in evaluation_grid_names])
    assert nodes in content

    #ELEMENTS [%s ... %s]
    elements = (
        'ELEMENTS\n'
        '../../ObjectMeshes/Reference/Elements.txt\n')
    elements += ''.join([
        f'../../EvaluationGrids/{grid}/Elements.txt\n' \
            for grid in evaluation_grid_names])
    assert elements in content

    #SYMMETRY [%d %d %d %f %f %f]
    symmetry = (
        '# SYMMETRY\n'
        '# 0 0 0\n'
        '# 0.0000e+00 0.0000e+00 0.0000e+00\n')
    assert symmetry in content

    #BOUNDARY : ELEM [%d] TO [%d %s %f %d %f %d]
    boundary = '##\nBOUNDARY\nRETU\n'
    assert boundary in content

    #PLANE WAVES [%d %f %f %f %d %f %d %f]
    xx = source_positions.x
    yy = source_positions.y
    zz = source_positions.z
    if n_plane_waves:
        plane_waves = (
            'PLANE WAVES\n')
        plane_waves += ''.join([
            f'{i} {xx[i]} {yy[i]} {zz[i]} 0.1 -1 0.0 -1\n' \
                for i in range(source_positions.csize)])
        assert plane_waves in content

    if n_point_sources:
        point_sources = (
            'POINT SOURCES\n')
        point_sources += ''.join([
            f'{i} {xx[i]} {yy[i]} {zz[i]} 0.1 -1 0.0 -1\n' \
                for i in range(source_positions.csize)])
        assert point_sources in content

    # no curves in this case
    assert '\n# CURVES\n' in content
    assert '\nPOST PROCESS\n' in content
    assert '\nEND\n' in content


@pytest.fixture
def valid_inputs(simple_mesh):
    return {
        "project_path": "test_project",
        "project_title": "Test Project",
        "frequencies": np.array([1000, 2000, 3000]),
        "sound_sources": SoundSource(
            pf.Coordinates(0, 0, 0, weights=1), SoundSourceType.POINT_SOURCE),
        "evaluation_grids": [EvaluationGrid.from_spherical(
            pf.samplings.sph_gaussian(10), "grid1")],
        "sample_mesh": SampleMesh(simple_mesh, SurfaceDescription()),
    }

def test_invalid_sound_sources(valid_inputs):
    inputs = valid_inputs.copy()
    inputs["sound_sources"] = "invalid"
    with pytest.raises(
        ValueError,
        match="sound_sources must be a SoundSource object."):
        write_scattering_project_numcalc(**inputs)

def test_invalid_evaluation_grids(valid_inputs):
    inputs = valid_inputs.copy()
    inputs["evaluation_grids"] = "invalid"
    with pytest.raises(
        ValueError,
        match="evaluation_grids must be a list of EvaluationGrid objects."):
        write_scattering_project_numcalc(**inputs)

def test_invalid_sample_mesh(valid_inputs):
    inputs = valid_inputs.copy()
    inputs["sample_mesh"] = "invalid"
    with pytest.raises(
        ValueError,
        match="sample_mesh must be a SampleMesh object."):
        write_scattering_project_numcalc(**inputs)

def test_invalid_bem_method(valid_inputs):
    inputs = valid_inputs.copy()
    inputs["bem_method"] = "invalid"
    with pytest.raises(
        ValueError,
        match="method must be 'BEM', 'SL-FMM BEM', or 'ML-FMM BEM'."):
        write_scattering_project_numcalc(**inputs)

def test_invalid_speed_of_sound(valid_inputs):
    inputs = valid_inputs.copy()
    inputs["speed_of_sound"] = "invalid"
    with pytest.raises(
        ValueError, match="speed_of_sound must be a float or int."):
        write_scattering_project_numcalc(**inputs)

def test_invalid_density_of_medium(valid_inputs):
    inputs = valid_inputs.copy()
    inputs["density_of_medium"] = "invalid"
    with pytest.raises(
        ValueError, match="density_of_medium must be a float or int."):
        write_scattering_project_numcalc(**inputs)

def test_invalid_frequencies(valid_inputs):
    inputs = valid_inputs.copy()
    inputs["frequencies"] = "invalid"
    with pytest.raises(ValueError, match="frequencies must be a numpy array."):
        write_scattering_project_numcalc(**inputs)

def test_invalid_frequencies_dimension(valid_inputs):
    inputs = valid_inputs.copy()
    inputs["frequencies"] = np.array([[1000, 2000], [3000, 4000]])
    with pytest.raises(ValueError, match="frequencies must be a 1D array."):
        write_scattering_project_numcalc(**inputs)

def test_invalid_project_title(valid_inputs):
    inputs = valid_inputs.copy()
    inputs["project_title"] = 123
    with pytest.raises(ValueError, match="project_title must be a string."):
        write_scattering_project_numcalc(**inputs)

def test_invalid_project_path(valid_inputs):
    inputs = valid_inputs.copy()
    inputs["project_path"] = 123
    with pytest.raises(
            ValueError, match="project_path must be a string or Path."):
        write_scattering_project_numcalc(**inputs)
