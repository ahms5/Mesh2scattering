import mesh2scattering as m2s
import pyfar as pf
import os
import numpy.testing as npt
import shutil
import numpy as np


def test_import():
    from mesh2scattering import process
    assert process


def test_write_scattering_one_source(tmpdir):
    project_path = os.path.join(
        m2s.utils.repository_root(), "tests", "resources",
        'project_one_source')
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


def test_write_scattering_sine():
    # Test simulation and validate with Mommertz previous simulation,
    # results are same up to a diff to 0.06
    project_path = os.path.join(
        m2s.utils.repository_root(), "tests", "resources",
        'sine')
    # read simulated data
    scattering_coefficient_rand, _, _ = \
        pf.io.read_sofa(os.path.join(
            project_path, 'sine.scattering_rand.sofa'))

    # reference from Mommertz
    mommertz_ref = pf.FrequencyData(
        np.array([[[
            0.00592885, 0.00197628, 0.00790514, 0.01976285, 0.00790514,
            0.03754941, 0.09288538, 0.20553360, 0.32015810, 0.41501976,
            0.53162055, 0.63636364, 0.70158103]]]),
        np.array([
            491.04984069,  550.97743741,  618.21858267,  698.38026206,
            778.32061898,  873.30666786,  979.88478981, 1099.46967844,
            1233.64867623, 1384.20284453, 1553.13060495, 1742.67426596,
            1942.15012515]))

    # compare to reference
    npt.assert_equal(
        scattering_coefficient_rand.frequencies,
        mommertz_ref.frequencies)
    npt.assert_allclose(
        scattering_coefficient_rand.freq,
        mommertz_ref.freq, atol=6e-2)


def test_write_diffusion_one_source(tmpdir):
    project_path = os.path.join(
        m2s.utils.repository_root(), "tests", "resources",
        'project_one_source')
    test_dir = os.path.join(tmpdir, 'project_one_source')
    shutil.copytree(project_path, test_dir)

    m2s.output.write_pattern(test_dir)

    m2s.process.calculate_diffusion(test_dir)
    diffusion_coefficient, source_coords, receiver_coords = pf.io.read_sofa(
        os.path.join(test_dir, 'project_one_source.diffusion.sofa'))
    diffusion_coefficient_rand, source_coords_rand, receiver_coords_rand = \
        pf.io.read_sofa(os.path.join(
            test_dir, 'project_one_source.diffusion_rand.sofa'))
    assert source_coords_rand.csize == 1
    assert receiver_coords_rand.csize == 1
    assert receiver_coords.csize == 1
    assert source_coords.csize == 1
    npt.assert_equal(
        diffusion_coefficient_rand.frequencies,
        diffusion_coefficient.frequencies)
    npt.assert_equal(
        diffusion_coefficient_rand.freq,
        diffusion_coefficient.freq)
