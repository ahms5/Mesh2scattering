import mesh2scattering as m2s
import pyfar as pf
import os
import numpy.testing as npt
import shutil


def test_import():
    from mesh2scattering import process
    assert process


def test_write_scattering(tmpdir):
    project_path = os.path.join(
        m2s.utils.repository_root(), "examples", "project")
    test_dir = os.path.join(tmpdir, 'project')
    shutil.copytree(project_path, test_dir)
    m2s.output.write_pattern(test_dir)
    m2s.process.calculate_scattering(test_dir)
    scattering_coefficient, source_coords, receiver_coords = pf.io.read_sofa(
        os.path.join(test_dir, 'project.scattering.sofa'))
    scattering_coefficient_rand, source_coords_rand, receiver_coords_rand = \
        pf.io.read_sofa(os.path.join(
            test_dir, 'project.scattering_rand.sofa'))
    assert source_coords_rand.csize == 1
    assert receiver_coords_rand.csize == 1
    assert receiver_coords.csize == 1
    assert source_coords.csize == 27
    npt.assert_equal(
        scattering_coefficient_rand.frequencies,
        scattering_coefficient.frequencies)


def test_write_scattering_one_source(tmpdir):
    project_path = os.path.join(
        m2s.utils.repository_root(), "tests", "resources", 'project_one_source')
    test_dir = os.path.join(tmpdir, 'project_one_source')
    shutil.copytree(project_path, test_dir)

    m2s.output.write_pattern(test_dir)
    
    m2s.process.calculate_scattering(test_dir)
    scattering_coefficient, source_coords, receiver_coords = pf.io.read_sofa(
        os.path.join(test_dir, 'project_one_source.scattering.sofa'))
    scattering_coefficient_rand, source_coords_rand, receiver_coords_rand = \
        pf.io.read_sofa(os.path.join(
            test_dir, 'project_one_source.scattering_rand.sofa'))
    assert source_coords_rand.csize == 1
    assert receiver_coords_rand.csize == 1
    assert receiver_coords.csize == 1
    assert source_coords.csize == 1
    npt.assert_equal(
        scattering_coefficient_rand.frequencies,
        scattering_coefficient.frequencies)
    npt.assert_equal(
        scattering_coefficient_rand.freq,
        scattering_coefficient.freq)


