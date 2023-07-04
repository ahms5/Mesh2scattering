import os
import mesh2scattering as m2s
import numpy.testing as npt
import pyfar as pf
import shutil
import pytest
import numpy as np
import re


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
        m2s.utils.repository_root(), 'release', 'examples', 'project')
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


def test_apply_symmetry_mirror_90(
        half_sphere):
    hemisphere_inc = pf.samplings.sph_gaussian(10)
    hemisphere_inc = hemisphere_inc[
        hemisphere_inc.colatitude <= np.pi/2]
    half_hemisphere_inc = hemisphere_inc[
        hemisphere_inc.azimuth <= np.pi]
    quarter_hemisphere_inc = hemisphere_inc[
        hemisphere_inc.azimuth <= np.pi/2]

    data_in = pf.FrequencyData(np.zeros(
        (quarter_hemisphere_inc.csize, half_sphere.csize, 2)), [100, 200])
    data_in.freq[0, 0, 0] = 1
    data_in.freq[11, 100, 1] = 2
    data, new_coords = m2s.output.apply_symmetry_mirror(
        data_in, half_sphere,
        quarter_hemisphere_inc, 90)

    # test new_coords object
    npt.assert_almost_equal(new_coords.azimuth, half_hemisphere_inc.azimuth)
    npt.assert_almost_equal(
        np.sort(new_coords.colatitude),
        np.sort(half_hemisphere_inc.colatitude))
    npt.assert_almost_equal(
        new_coords.radius, half_hemisphere_inc.radius)
    npt.assert_equal(
        new_coords[:quarter_hemisphere_inc.csize].cartesian,
        quarter_hemisphere_inc.cartesian)

    # test new_coords and data relationship
    npt.assert_equal(data.cshape[0], new_coords.csize)
    npt.assert_equal(data.cshape[1], half_sphere.csize)

    # test data shift
    assert data.freq[0, 0, 0] == 1
    assert data.freq[-1, 441, 0] == 1
    assert np.sum(data.freq[..., 0]) == 2
    assert data.freq[11, 100, 1] == 2
    assert data.freq[18, 184, 1] == 2
    assert np.sum(data.freq[..., 1]) == 4


def test_apply_symmetry_mirror_180(
        quarter_hemisphere_inc, half_sphere,
        data_in):
    data_in.freq[0, 0, 0] = 1
    data_in.freq[11, 100, 1] = 2
    data, new_coords = m2s.output.apply_symmetry_mirror(
        data_in, half_sphere,
        quarter_hemisphere_inc, 90)
    data, new_coords = m2s.output.apply_symmetry_mirror(
        data, half_sphere, new_coords, 180)

    # test new_coords object
    new_coords_desired = pf.samplings.sph_gaussian(10)
    new_coords_desired = new_coords_desired[
        new_coords_desired.colatitude <= np.pi/2]
    npt.assert_almost_equal(new_coords.azimuth, new_coords_desired.azimuth)
    npt.assert_almost_equal(
        np.sort(new_coords.colatitude), np.sort(new_coords_desired.colatitude))
    npt.assert_almost_equal(
        new_coords.radius, new_coords_desired.radius)
    npt.assert_equal(
        new_coords[:quarter_hemisphere_inc.csize].cartesian,
        quarter_hemisphere_inc.cartesian)

    # test new_coords and data relationship
    npt.assert_equal(data.cshape[0], new_coords.csize)
    npt.assert_equal(data.cshape[1], half_sphere.csize)

    # test data shift
    assert data.freq[0, 0, 0] == 1
    assert data.freq[29, 441, 0] == 1
    assert np.sum(data.freq[..., 0]) == 2
    assert data.freq[11, 100, 1] == 2
    assert data.freq[18, 184, 1] == 2
    assert data.freq[36, 541, 1] == 2
    assert data.freq[43, 625, 1] == 2
    assert np.sum(data.freq[..., 1]) == 8


def test_apply_symmetry_mirror_20_90(half_sphere):
    hemisphere_inc = pf.samplings.sph_gaussian(20)
    hemisphere_inc = hemisphere_inc[
        hemisphere_inc.colatitude <= np.pi/2]
    quarter_hemisphere_inc = hemisphere_inc[
        hemisphere_inc.azimuth <= np.pi/2]
    half_hemisphere_inc = hemisphere_inc[
        hemisphere_inc.azimuth <= np.pi]

    data_in = pf.FrequencyData(np.zeros(
        (quarter_hemisphere_inc.csize, half_sphere.csize, 2)), [100, 200])
    data_in.freq[0, 0, 0] = 1
    data_in.freq[11, 100, 1] = 2
    data, new_coords = m2s.output.apply_symmetry_mirror(
        data_in, half_sphere,
        quarter_hemisphere_inc, 90)

    # test new_coords object
    npt.assert_almost_equal(new_coords.azimuth, half_hemisphere_inc.azimuth)
    npt.assert_almost_equal(
        np.sort(new_coords.colatitude),
        np.sort(half_hemisphere_inc.colatitude))
    npt.assert_almost_equal(
        new_coords.radius, half_hemisphere_inc.radius)
    npt.assert_equal(
        new_coords[:quarter_hemisphere_inc.csize].cartesian,
        quarter_hemisphere_inc.cartesian)

    # test new_coords and data relationship
    npt.assert_equal(data.cshape[0], new_coords.csize)
    npt.assert_equal(data.cshape[1], half_sphere.csize)

    # test data shift
    assert data.freq[0, 0, 0] == 1
    assert data.freq[109, 441, 0] == 1
    assert np.sum(data.freq[..., 0]) == 2
    assert data.freq[11, 100, 1] == 2
    assert data.freq[98, 457, 1] == 2
    assert np.sum(data.freq[..., 1]) == 4


def test_apply_symmetry_mirror_20_180(half_sphere):
    hemisphere_inc = pf.samplings.sph_gaussian(20)
    hemisphere_inc = hemisphere_inc[
        hemisphere_inc.colatitude <= np.pi/2]
    quarter_hemisphere_inc = hemisphere_inc[
        hemisphere_inc.azimuth <= np.pi/2]

    data_in = pf.FrequencyData(np.zeros(
        (quarter_hemisphere_inc.csize, half_sphere.csize, 2)), [100, 200])
    data_in.freq[0, 0, 0] = 1
    data_in.freq[11, 100, 1] = 2
    data, new_coords = m2s.output.apply_symmetry_mirror(
        data_in, half_sphere,
        quarter_hemisphere_inc, 90)
    data, new_coords = m2s.output.apply_symmetry_mirror(
        data, half_sphere, new_coords, 180)

    # test new_coords object
    npt.assert_almost_equal(new_coords.azimuth, hemisphere_inc.azimuth)
    npt.assert_almost_equal(
        np.sort(new_coords.colatitude),
        np.sort(hemisphere_inc.colatitude))
    npt.assert_almost_equal(
        new_coords.radius, hemisphere_inc.radius)
    npt.assert_equal(
        new_coords[:quarter_hemisphere_inc.csize].cartesian,
        quarter_hemisphere_inc.cartesian)

    # test new_coords and data relationship
    npt.assert_equal(data.cshape[0], new_coords.csize)
    npt.assert_equal(data.cshape[1], half_sphere.csize)

    # test data shift
    assert data.freq[0, 0, 0] == 1
    assert data.freq[109, 441, 0] == 1
    assert np.sum(data.freq[..., 0]) == 2
    assert data.freq[11, 100, 1] == 2
    assert data.freq[98, 457, 1] == 2
    assert data.freq[111, 541, 1] == 2
    assert data.freq[198, 16, 1] == 2
    assert np.sum(data.freq[..., 1]) == 8


def test_apply_symmetry_mirror_with_top_point(
        half_sphere):
    hemisphere_inc = pf.samplings.sph_equal_angle(30, 10)
    hemisphere_inc = hemisphere_inc[
        hemisphere_inc.colatitude <= np.pi/2]
    half_hemisphere_inc = hemisphere_inc[
        hemisphere_inc.azimuth <= np.pi]
    quarter_hemisphere_inc = hemisphere_inc[
        hemisphere_inc.azimuth <= np.pi/2]

    data_in = pf.FrequencyData(np.zeros(
        (quarter_hemisphere_inc.csize, half_sphere.csize, 2)), [100, 200])
    data, new_coords = m2s.output.apply_symmetry_mirror(
        data_in, half_sphere,
        quarter_hemisphere_inc, 90)

    # test new_coords object
    npt.assert_almost_equal(
        np.sort(new_coords.azimuth), np.sort(half_hemisphere_inc.azimuth))
    npt.assert_almost_equal(
        np.sort(new_coords.colatitude),
        np.sort(half_hemisphere_inc.colatitude))
    npt.assert_almost_equal(
        new_coords.radius, half_hemisphere_inc.radius)
    npt.assert_equal(
        new_coords[:quarter_hemisphere_inc.csize].cartesian,
        quarter_hemisphere_inc.cartesian)

    # test new_coords and data relationship
    npt.assert_equal(data.cshape[0], new_coords.csize)
    npt.assert_equal(data.cshape[1], half_sphere.csize)


def test_apply_symmetry_mirror_error(half_sphere):
    hemisphere_inc = pf.samplings.sph_gaussian(10)
    hemisphere_inc = hemisphere_inc[
        hemisphere_inc.colatitude <= np.pi/2]
    quarter_hemisphere_inc = hemisphere_inc[
        hemisphere_inc.azimuth <= np.pi/2]

    data_in_correct = pf.FrequencyData(np.zeros(
        (quarter_hemisphere_inc.csize, half_sphere.csize, 2)), [100, 200])

    with pytest.raises(
            ValueError, match=re.escape(
            "data_in.cshape[-2] needs to have the dimension of coords_inc "
            "15 != 50.")):
        m2s.output.apply_symmetry_mirror(
            data_in_correct, half_sphere, hemisphere_inc, 90)

    with pytest.raises(
            ValueError, match=re.escape(
            "data_in.cshape[-1] needs to have the dimension"
            " of coords_mic 882 != 50.")):
        m2s.output.apply_symmetry_mirror(
            data_in_correct, hemisphere_inc, quarter_hemisphere_inc, 90)

    with pytest.raises(TypeError):
        m2s.output.apply_symmetry_mirror(
            data_in_correct, hemisphere_inc, quarter_hemisphere_inc,
            quarter_hemisphere_inc)

    with pytest.raises(ValueError, match="0 >= symmetry_azimuth_deg >= 360"):
        m2s.output.apply_symmetry_mirror(
            data_in_correct, hemisphere_inc, quarter_hemisphere_inc,
            -5)

    with pytest.raises(
            ValueError, match="coords_inc needs to be of type pf.Coordinates"):
        m2s.output.apply_symmetry_mirror(
            data_in_correct, hemisphere_inc, 90, 90)

    with pytest.raises(
            ValueError, match="coords_mic needs to be of type pf.Coordinates"):
        m2s.output.apply_symmetry_mirror(
            data_in_correct, 7, quarter_hemisphere_inc, 90)

    with pytest.raises(
            ValueError, match="data_in needs to be of type pf.FrequencyData"):
        m2s.output.apply_symmetry_mirror(
            3, hemisphere_inc, quarter_hemisphere_inc, 90)

    with pytest.raises(
            ValueError, match="coords_inc needs to be of type pf.Coordinates"):
        m2s.output.apply_symmetry_mirror(
            data_in_correct, hemisphere_inc, 1, 90)


def test_apply_symmetry_circular(half_sphere):
    hemisphere_inc = pf.samplings.sph_gaussian(20)
    hemisphere_inc = hemisphere_inc[
        hemisphere_inc.colatitude <= np.pi/2]
    hemisphere_inc_in = hemisphere_inc[
        hemisphere_inc.azimuth == 0]

    data_in = pf.FrequencyData(np.zeros(
        (hemisphere_inc_in.csize, half_sphere.csize, 2)), [100, 200])
    data_in.freq[0, 0, 0] = 1
    data_in.freq[5, 200, 1] = 2
    data = m2s.output.apply_symmetry_circular(
        data_in, half_sphere, hemisphere_inc_in, hemisphere_inc)

    # test new_coords and data relationship
    npt.assert_equal(data.cshape[0], hemisphere_inc.csize)
    npt.assert_equal(data.cshape[1], half_sphere.csize)

    # test data shift
    assert data.freq[0, 0, 0] == 1
    assert data.freq[10, 42, 0] == 1
    assert data.freq[20, 2*42, 0] == 1
    assert data.freq[30, 3*42, 0] == 1
    assert data.freq[40, 4*42, 0] == 1
    assert data.freq[50, 5*42, 0] == 1
    assert data.freq[60, 273, 0] == 1
    assert data.freq[70, 315, 0] == 1
    assert data.freq[80, 357, 0] == 1
    assert data.freq[90, 399, 0] == 1
    assert data.freq[100, 441, 0] == 1
    assert data.freq[110, 483, 0] == 1
    assert data.freq[120, 525, 0] == 1
    assert data.freq[130, 567, 0] == 1
    assert data.freq[140, 609, 0] == 1
    assert data.freq[150, 651, 0] == 1
    assert data.freq[160, 714, 0] == 1
    assert data.freq[170, 756, 0] == 1
    assert data.freq[180, 798, 0] == 1
    assert data.freq[190, 840, 0] == 1
    assert np.sum(data.freq[..., 0]) == 20


def test_apply_symmetry_circular_error(half_sphere):
    hemisphere_inc = pf.samplings.sph_gaussian(10)
    hemisphere_inc = hemisphere_inc[
        hemisphere_inc.colatitude <= np.pi/2]
    hemisphere_inc_in = hemisphere_inc[
        hemisphere_inc.azimuth == 0]
    hemisphere_inc_error = hemisphere_inc[
        hemisphere_inc.azimuth <= np.pi/2]

    data_in_correct = pf.FrequencyData(np.zeros(
        (hemisphere_inc_in.csize, half_sphere.csize, 2)), [100, 200])

    with pytest.raises(
            ValueError, match=re.escape(
            "data_in.cshape[-2] needs to have the dimension of coords_inc")):
        m2s.output.apply_symmetry_circular(
            data_in_correct, half_sphere, hemisphere_inc_error, hemisphere_inc)

    with pytest.raises(
            ValueError, match=re.escape(
            "data_in.cshape[-1] needs to have the dimension"
            " of coords_mic")):
        m2s.output.apply_symmetry_circular(
            data_in_correct, hemisphere_inc_error, hemisphere_inc_in,
            hemisphere_inc)

    with pytest.raises(
            ValueError, match="coords_inc needs to be of type pf.Coordinates"):
        m2s.output.apply_symmetry_circular(
            data_in_correct, half_sphere, 1, hemisphere_inc)

    with pytest.raises(
            ValueError, match="coords_mic needs to be of type pf.Coordinates"):
        m2s.output.apply_symmetry_circular(
            data_in_correct, 1, hemisphere_inc_in, hemisphere_inc)

    with pytest.raises(
            ValueError, match="data_in needs to be of type pf.FrequencyData"):
        m2s.output.apply_symmetry_circular(
            1, half_sphere, hemisphere_inc_in, hemisphere_inc)

    with pytest.raises(
            ValueError, match="coords_inc_out needs to be of type pf.Coordinates"):
        m2s.output.apply_symmetry_circular(
            data_in_correct, half_sphere, hemisphere_inc_in, 1)

    hemisphere_inc_in.azimuth = [1., 0., 0., 0., 0.]
    with pytest.raises(
            ValueError, match="coords_inc needs to have constant azimuth angle."):
        m2s.output.apply_symmetry_circular(
            data_in_correct, half_sphere, hemisphere_inc_in, hemisphere_inc)


