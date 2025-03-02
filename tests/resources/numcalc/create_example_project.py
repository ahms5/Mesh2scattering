# %%
import mesh2scattering as m2s
import os
import pyfar as pf
import numpy as np
import trimesh

# %%
project_path = os.path.join(
    os.path.dirname(__file__), 'low_complexity_project')

mesh_path = os.path.join(
    os.path.dirname(__file__), '..', 'mesh', 'reference_n4_0.5', 'sample.stl')
mesh = trimesh.load_mesh(mesh_path)
project_title = 'low_complexity_project'
frequencies = np.array([100, 200])
sound_sources = m2s.input.SoundSource(
    pf.Coordinates(0, 0, [3, 4], weights=[1, 1]),
    m2s.input.SoundSourceType.POINT_SOURCE)
points = pf.samplings.sph_lebedev(sh_order=10)
evaluation_grid = m2s.input.EvaluationGrid.from_spherical(
    points,
    'example_grid')
surface_description = m2s.input.SurfaceDescription()
sample_mesh = m2s.input.SampleMesh(
    mesh,
    surface_description,
    0.5,
    m2s.input.SampleShape.ROUND,
)
bem_method = 'ML-FMM BEM'
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

# %%
project_path = os.path.join(
    os.path.dirname(__file__), 'low_complexity_project_solved')
project_title = 'low_complexity_project_solved'
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

# %%
numcalc = os.path.join(
    m2s.utils.repository_root(),
    'mesh2scattering', 'numcalc', 'bin', 'NumCalc')
m2s.numcalc.manage_numcalc(project_path, numcalc, wait_time=0)

# %%
