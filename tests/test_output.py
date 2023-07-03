import os
import mesh2scattering as m2s
import numpy.testing as npt
import pyfar as pf
import shutil
import pytest


def test_import():
    from mesh2scattering import output
    assert output


@pytest.mark.parametrize("folders,issue,errors,nots", (
    # no issues single NC.out filejoin
    [["case_0"], False, [], []],
    # issues in NC.out that are corrected by second file NC1-1.out
    [["case_4"], False, [], []],
    # missing frequencies
    [["case_1"], True,
     ["Frequency steps that were not calculated:\n59, 60"], []],
    # convergence issues
    [["case_2"], True,
     ["Frequency steps that did not converge:\n18, 42"], []],
    # input/mesh issues
    [["case_3"], True,
     ["Frequency steps that were not calculated:\n59, 60",
      "Frequency steps with bad input:\n58"], []],
    # no isses in source 1 but issues in source 2
    [["case_0", "case_1"], True,
     ["Detected issues for source 2",
      "Frequency steps that were not calculated:\n59, 60"],
     ["Detected issues for source 1"]]
))
def test_project_report(folders, issue, errors, nots, tmpdir):
    """Test issues found by the project report"""

    cwd = os.path.dirname(__file__)
    data_nc = os.path.join(cwd, 'resources', 'nc.out')
    # create fake project structure
    os.mkdir(os.path.join(tmpdir, "NumCalc"))
    os.mkdir(os.path.join(tmpdir, "Output2HRTF"))
    shutil.copyfile(os.path.join(data_nc, "parameters.json"),
                    os.path.join(tmpdir, "parameters.json"))
    for ff, folder in enumerate(folders):
        shutil.copytree(os.path.join(data_nc, folder),
                        os.path.join(tmpdir, "NumCalc", f"source_{ff + 1}"))

    # run the project report
    issues, report = m2s.output.write_output_report(tmpdir)

    # test the output
    assert issues is issue
    for error in errors:
        assert error in report
    for no in nots:
        assert no not in report
    if issue:
        assert os.path.isfile(os.path.join(
            tmpdir, "Output2HRTF", "report_issues.txt"))
        assert ("For more information check Output2HRTF/report_source_*.csv "
                "and the NC*.out files located at NumCalc/source_*") in report
    else:
        assert not os.path.isfile(os.path.join(
            tmpdir, "Output2HRTF", "report_issues.txt"))


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


def test_apply_symmetry_mirror(
        quarter_half_sphere, half_sphere,
        pressure_data_mics_incident_directions):
    data, new_coords = m2s.output.apply_symmetry_mirror(
        pressure_data_mics_incident_directions, half_sphere,
        quarter_half_sphere, 0, 90)
    npt.assert_equal(new_coords[:quarter_half_sphere.csize], quarter_half_sphere)
