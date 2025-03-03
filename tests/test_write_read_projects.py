import mesh2scattering as m2s
import os
import numpy as np
import pyfar as pf
import subprocess
import warnings
import trimesh
import sofar as sf


# directory of this file
base_dir = os.path.dirname(__file__)

# ignore tests for windows since its difficult to build the exe
if os.name == 'nt':
    numcalc = os.path.join(
        m2s.utils.program_root(), "numcalc", "bin", "NumCalc.exe")
    numcalc_path = numcalc
    warnings.warn(
        ('Under Windows the code is not compiling but an executable is '
         f'expected in {numcalc}.'), UserWarning, stacklevel=2)

else:
    # Build NumCalc locally to use for testing
    numcalc = os.path.join(
        m2s.utils.program_root(), "numcalc", "bin", "NumCalc")
    numcalc_path = numcalc

    if os.path.isfile(numcalc):
        os.remove(numcalc)

    subprocess.run(
        ["make"], cwd=os.path.join(
            m2s.utils.program_root(), "numcalc", "src"), check=True)


def test_write_project(tmpdir):

    s2 = pf.Coordinates(0, 0, [3, 4], weights=[1, 1])
    s1 = pf.Coordinates(0, 0, 3, weights=[1])

    for frequencies in [np.array([500, 501]), np.array([500])]:
        for source_type in [
                m2s.input.SoundSourceType.POINT_SOURCE,
                m2s.input.SoundSourceType.PLANE_WAVE]:
            for sources in [s1, s2]:
                sound_sources = m2s.input.SoundSource(
                    sources,
                    source_type)
                points = pf.samplings.sph_gaussian(sh_order=63)
                evaluation_grid = m2s.input.EvaluationGrid.from_spherical(
                    points,
                    'gaussian_63')
                surface_description = m2s.input.SurfaceDescription(
                    .08, 0, 0.051/2.5, symmetry_azimuth=[90, 180])
                base_path = os.path.join(
                    m2s.utils.repository_root(), 'tests', 'resources')
                sample = (
                    f'reference_{source_type.name}_s{sources.csize}_'
                    f'f{frequencies.size}')
                project_path = os.path.join(
                    tmpdir, sample)
                project_title = sample

                mesh_path = os.path.join(
                    base_path,
                    'mesh', 'reference_n4_0.5', 'sample.stl')
                sample_mesh = m2s.input.SampleMesh(
                    trimesh.load_mesh(mesh_path),
                    surface_description,
                    0.01,
                    0.8,
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
    m2s.numcalc.manage_numcalc(
        tmpdir, numcalc, wait_time=0)
    for frequencies in [1, 2]:
        for source_type in ['POINT_SOURCE', 'PLANE_WAVE']:
            for sources in [1, 2]:
                sample = f'reference_{source_type}_s{sources}_f{frequencies}'
                project_path = os.path.join(tmpdir, sample)
                m2s.output.write_pressure(project_path)


                sofa = sf.read_sofa(
                    os.path.join(
                        project_path, '..',
                        f'{sample}_gaussian_63.pressure.sofa'))
                pressure, sources, receivers = pf.io.convert_sofa(sofa)

                # check if the sofa file is correct
                assert pressure.cshape[0] == sources.csize
                assert pressure.cshape[1] == receivers.csize
                assert isinstance(sofa.SpeedOfSound, float)
                assert isinstance(sofa.SampleStructuralWavelengthX, float)
                assert isinstance(sofa.SampleStructuralWavelengthY, float)
                assert isinstance(sofa.SampleDiameter, float)
                assert isinstance(sofa.ReceiverWeights, np.ndarray)
                assert sofa.ReceiverWeights.size == receivers.csize
                assert np.atleast_1d(sofa.SourceWeights).size == sources.csize


