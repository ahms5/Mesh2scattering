import mesh2scattering as m2s
import pyfar as pf
import os
import numpy.testing as npt


def test_import():
    from mesh2scattering import process
    assert process


def test_write_scattering():
    project_path = os.path.join(
        m2s.utils.repository_root(), "examples", "project")
    m2s.process.calculate_scattering(project_path)
    scattering_coefficient, source_coords, receiver_coords = pf.io.read_sofa(
        os.path.join(project_path, 'project.scattering.sofa'))
    scattering_coefficient_rand, source_coords_rand, receiver_coords_rand = \
        pf.io.read_sofa(os.path.join(
            project_path, 'project.scattering_rand.sofa'))
    assert source_coords_rand.csize == 1
    assert receiver_coords_rand.csize == 1
    assert receiver_coords.csize == 1
    assert source_coords.csize == 27
    npt.assert_equal(
        scattering_coefficient_rand.frequencies,
        scattering_coefficient.frequencies)
