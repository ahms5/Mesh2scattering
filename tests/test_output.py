import os
import mesh2scattering as m2s
import numpy.testing as npt
import pyfar as pf
import shutil


def test_import():
    from mesh2scattering import output
    assert output


def test_write_pattern(tmpdir):
    project_path = os.path.join(
        m2s.utils.repository_root(), "examples", "project")
    test_dir = os.path.join(tmpdir, 'project_one_source')
    shutil.copytree(project_path, test_dir)
    m2s.output.write_pattern(test_dir)
    reference, source_coords_ref, receiver_coords_ref = pf.io.read_sofa(
        os.path.join(test_dir, 'reference.pattern.sofa'))
    sample, source_coords, receiver_coords = pf.io.read_sofa(
        os.path.join(test_dir, 'sample.pattern.sofa'))
    assert sample.cshape[0] == source_coords.csize
    assert sample.cshape[1] == receiver_coords.csize
    assert sample.cshape[0] == source_coords.csize
    assert sample.cshape[1] == receiver_coords.csize
    assert reference.cshape[0] == source_coords_ref.csize
    assert reference.cshape[1] == receiver_coords_ref.csize
    assert reference.cshape[0] == source_coords_ref.csize
    assert reference.cshape[1] == receiver_coords_ref.csize
    assert reference.cshape == sample.cshape
    npt.assert_equal(reference.frequencies, sample.frequencies)


def test_write_pattern_one_source(tmpdir):
    project_path = os.path.join(
        m2s.utils.repository_root(), "tests", "resources",
        'project_one_source')
    test_dir = os.path.join(tmpdir, 'project_one_source')
    shutil.copytree(project_path, test_dir)

    m2s.output.write_pattern(test_dir)

    reference, source_coords_ref, receiver_coords_ref = pf.io.read_sofa(
        os.path.join(test_dir, 'reference.pattern.sofa'))
    sample, source_coords, receiver_coords = pf.io.read_sofa(
        os.path.join(test_dir, 'sample.pattern.sofa'))
    assert sample.cshape[0] == source_coords.csize
    assert sample.cshape[1] == receiver_coords.csize
    assert sample.cshape[0] == source_coords.csize
    assert sample.cshape[1] == receiver_coords.csize
    assert reference.cshape[0] == source_coords_ref.csize
    assert reference.cshape[1] == receiver_coords_ref.csize
    assert reference.cshape[0] == source_coords_ref.csize
    assert reference.cshape[1] == receiver_coords_ref.csize
    assert reference.cshape == sample.cshape
    npt.assert_equal(reference.frequencies, sample.frequencies)
