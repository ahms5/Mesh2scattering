# %%
import mesh2scattering as m2s
import os
import pyfar as pf
import numpy as np
import trimesh

# %%
frequencies = np.array([500, 1000])
sound_sources = m2s.input.SoundSource(
    pf.Coordinates(0, 0, [3, 4], weights=[1, 1]),
    m2s.input.SoundSourceType.POINT_SOURCE)
points = pf.samplings.sph_gaussian(sh_order=63)
evaluation_grid = m2s.input.EvaluationGrid.from_spherical(
    points,
    'gaussian_63')
base_path = os.path.join(
    m2s.utils.repository_root(), 'tests', 'resources')
for sample in ['sine', 'reference']:
    project_path = os.path.join(
        base_path, 'output', sample)
    project_title = sample
    symmetry_azimuth = [90, 180] if sample == 'sine' else []
    symmetry_rotational = False if sample == 'sine' else True

    surface_description = m2s.input.SurfaceDescription(
        .08, 0, 0.051/2.5, symmetry_azimuth=symmetry_azimuth,
        symmetry_rotational=symmetry_rotational)

    mesh_path = os.path.join(
        base_path,
        'mesh', f'{sample}_n10_0.5', 'sample.stl')
    sample_mesh = m2s.input.SampleMesh(
        trimesh.load_mesh(mesh_path),
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
numcalc = os.path.join(
    m2s.utils.repository_root(),
    'mesh2scattering', 'numcalc', 'bin', 'NumCalc')
m2s.numcalc.manage_numcalc(
    os.path.join(base_path, 'output'), numcalc, wait_time=0)

# %%
for sample in ['sine', 'reference']:
    project_path = os.path.join(
        base_path, 'output', sample)

    m2s.output.write_pressure(project_path)
# %%
