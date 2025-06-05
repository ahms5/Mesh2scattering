import mesh2scattering as m2s
import os
import numpy as np
import pyfar as pf
import pytest


@pytest.mark.parametrize('source_type',[
    m2s.input.SoundSourceType.POINT_SOURCE,
    m2s.input.SoundSourceType.PLANE_WAVE,
])
@pytest.mark.parametrize('bem_method', [
    'BEM',
    'SL-FMM BEM',
    'ML-FMM BEM',
])
@pytest.mark.parametrize('bc', [
    None,
    m2s.input.bc.BoundaryCondition(
        .5, # R = 1/3 -> alpha = 0.88
        m2s.input.bc.BoundaryConditionType.impedance),
    m2s.input.bc.BoundaryCondition(
        pf.FrequencyData([0.5, 0.5], [100, 200]), # R = 1/3 -> alpha = 0.88
        m2s.input.bc.BoundaryConditionType.admittance),
    ])
def test_write_run_project(source_type, bem_method, tmpdir, simple_mesh, bc):
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
    bcm = None
    surface_description = m2s.input.SurfaceDescription()
    if bc is not None:
        n_mesh_faces = simple_mesh.faces.shape[0]
        bcm = m2s.input.bc.BoundaryConditionMapping(n_mesh_faces)
        bcm.add_boundary_condition(
            bc, 0, n_mesh_faces-1,
        )

    sample_mesh = m2s.input.SampleMesh(
        simple_mesh,
        surface_description,
        0.01,
        0.8,
        m2s.input.SampleShape.ROUND,
        bc_mapping=bcm,
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

    # run NumCalc estimate ram to check inputs
    numcalc_executable = m2s.numcalc.build_or_fetch_numcalc()
    m2s.numcalc.calc_and_read_ram(
        project_path, numcalc_executable)
