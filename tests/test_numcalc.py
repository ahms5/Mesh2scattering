import pytest
import subprocess
import shutil
import os
import mesh2scattering as m2s
import glob
import warnings
import numpy.testing as npt
import numpy as np

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


def test_import():
    from mesh2scattering import numcalc
    assert numcalc


def test_numcalc_invalid_parameter(capfd):
    """
    Test if NumCalc throws an error in case of invalid command line
    parameter.
    """
    try:
        # run NumCalc with subprocess
        if os.name == 'nt':  # Windows detected
            # run NumCalc and route all printouts to a log file
            subprocess.run(
                f'{numcalc} -invalid_parameter',
                stdout=subprocess.DEVNULL, check=True)
        else:  # elif os.name == 'posix': Linux or Mac detected
            # run NumCalc and route all printouts to a log file
            subprocess.run(
                [f'{numcalc} -invalid_parameter'],
                shell=True, stdout=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        _, err = capfd.readouterr()
        assert "NumCalc was called with an unknown parameter or flag." \
            in err
    else:
        ValueError("Num calc did not throw an error")


@pytest.mark.parametrize(('nitermax'), [
    1,
    ])
def test_numcalc_commandline_nitermax(nitermax, tmpdir):
    """Test if command line parameter nitermax behaves as expected."""
    # Setup

    # copy test directory
    shutil.copytree(
        os.path.join(
            base_dir, 'resources', 'numcalc', 'low_complexity_project'),
        os.path.join(tmpdir, 'project'))

    commandLineArgument = f' -nitermax {nitermax}'

    # Exercise

    # run NumCalc with subprocess
    tmp_path = os.path.join(tmpdir, "project", "NumCalc", "source_1")
    if os.name == 'nt':  # Windows detected
        # run NumCalc and route all printouts to a log file
        subprocess.run(
            f'{numcalc}{commandLineArgument}',
            stdout=subprocess.DEVNULL, cwd=tmp_path, check=True)
    else:  # elif os.name == 'posix': Linux or Mac detected
        # run NumCalc and route all printouts to a log file
        subprocess.run(
            [f'{numcalc}{commandLineArgument}'],
            shell=True, stdout=subprocess.DEVNULL, cwd=tmp_path, check=True)

    # Verify
    out_filename = 'NC.out'
    out_filepath = os.path.join(
        tmpdir, "project", "NumCalc", "source_1", out_filename)

    out_file = open(out_filepath)
    out_text = out_file.read()

    assert f'CGS solver: number of iterations = {nitermax}' in out_text
    assert 'Warning: Maximum number of iterations is reached!' \
        in out_text


@pytest.mark.parametrize(('istart', 'iend'), [
    (1, False), (False, 2), (1, 2)])
def test_numcalc_commandline_istart_iend(istart, iend, tmpdir):
    """Test if command line parameters istart and iend behave as expected.
    """
    # copy test directory
    shutil.copytree(
        os.path.join(
            base_dir, 'resources', 'numcalc', 'low_complexity_project'),
        os.path.join(tmpdir, 'project'))

    commandLineArgument = ''
    if istart > 0:
        commandLineArgument += f' -istart {istart}'
    if iend > 0:
        commandLineArgument += f' -iend {iend}'

    # Exercise
    # run NumCalc with subprocess
    tmp_path = os.path.join(tmpdir, "project", "NumCalc", "source_1")
    if os.name == 'nt':  # Windows detected
        # run NumCalc and route all printouts to a log file
        subprocess.run(
            f'{numcalc}{commandLineArgument}',
            stdout=subprocess.DEVNULL, cwd=tmp_path, check=True)
    else:  # elif os.name == 'posix': Linux or Mac detected
        # run NumCalc and route all printouts to a log file
        subprocess.run(
            [f'{numcalc}{commandLineArgument}'],
            shell=True, stdout=subprocess.DEVNULL, cwd=tmp_path, check=True)

    # Verify
    if (not istart and not iend):
        out_filename = 'NC.out'
    elif ((istart > 0) and not iend):
        out_filename = f'NCfrom{istart}.out'
    elif (not istart and (iend > 0)):
        out_filename = f'NCuntil{iend}.out'
    elif ((istart > 0) and (iend > 0)):
        out_filename = f'NC{istart}-{iend}.out'
    else:
        raise Exception("Wrong istart and/or iend parameters chosen")

    out_filepath = os.path.join(
        tmpdir, "project", "NumCalc", "source_1", out_filename)

    with open(out_filepath) as out_file:
        out_text = out_file.read()

    if istart > 0:
        assert f'Step {istart-1}' not in out_text
        assert f'Step {istart}' in out_text
    else:
        assert 'Step 1' in out_text

    if iend > 0:
        assert f'Step {iend}' in out_text
        assert f'Step {iend+1}' not in out_text

    if istart > 0 and iend > 0:
        nStepsActual = out_text.count((
            '>> S T E P   N U M B E R   A N D   F R E Q U E N C Y <<'))
        nStepsExpected = iend - istart + 1
        assert nStepsActual == nStepsExpected


def test_numcalc_commandline_estimate_ram(tmpdir):
    """Test NumCalc's RAM estimation using -estimate_ram."""
    # copy test directory
    shutil.copytree(
        os.path.join(
            base_dir, 'resources', 'numcalc', 'low_complexity_project'),
        os.path.join(tmpdir, 'project'))

    tmp_path = os.path.join(tmpdir, "project", "NumCalc", "source_1")
    if os.name == 'nt':  # Windows detected
        # run NumCalc and route all printouts to a log file
        subprocess.run(
            f"{numcalc} -estimate_ram",
            stdout=subprocess.DEVNULL, cwd=tmp_path, check=True)
    else:  # elif os.name == 'posix': Linux or Mac detected
        # run NumCalc and route all printouts to a log file
        subprocess.run(
            [f"{numcalc} -estimate_ram"],
            shell=True, stdout=subprocess.DEVNULL, cwd=tmp_path, check=True)

    # check if Memory.txt exists
    assert os.path.isfile(os.path.join(tmp_path, 'Memory.txt'))


@pytest.mark.parametrize("boundary", [(False), (True)])
@pytest.mark.parametrize("grid", [(False), (True)])
@pytest.mark.parametrize("log", [(False), (True)])
def test_remove_outputs(boundary, grid, log, tmpdir):
    """Test purging the processed data in report."""
    # copy test directory
    test_dir = os.path.join(tmpdir, 'project')
    shutil.copytree(
        os.path.join(
            base_dir, 'resources', 'numcalc', 'low_complexity_project_solved'),
        test_dir)

    m2s.numcalc.remove_outputs(
        test_dir,
        boundary=boundary, grid=grid, log=log)

    assert len(glob.glob(
            os.path.join(test_dir, "*.sofa"))) == 0

    # Test boundary and grid
    for source in glob.glob(
            os.path.join(test_dir, "NumCalc", "source_*")):
        if boundary and grid:
            assert not os.path.isdir(os.path.join(source, "be.out"))
        elif boundary:
            assert os.path.isdir(os.path.join(source, "be.out"))
            for be in glob.glob(os.path.join(source, "be.out", "be.*")):
                assert glob.glob(os.path.join(be, "*Boundary")) == []
        elif grid:
            assert os.path.isdir(os.path.join(source, "be.out"))
            for be in glob.glob(os.path.join(source, "be.out", "be.*")):
                assert glob.glob(os.path.join(be, "*EvalGrid")) == []


def test_read_ram_estimates_assertions():
    """Test assertions for read_ram_estimates."""
    with pytest.raises(ValueError, match="does not contain a Memory.txt"):
        m2s.numcalc.read_ram_estimates(os.getcwd())


def test_calc_and_read_ram_estimation(tmpdir):
    shutil.copytree(
        os.path.join(
            base_dir, 'resources', 'numcalc', 'low_complexity_project'),
        os.path.join(tmpdir, 'project'))

    ram = m2s.numcalc.calc_and_read_ram(
        os.path.join(tmpdir, 'project'), numcalc)
    npt.assert_array_almost_equal(ram.shape, np.array([2, 3]))
    npt.assert_array_almost_equal(ram, np.array([
        [1, 100, 0.0121576],
        [2, 200, 0.0121576]]))


def test_calc_and_read_ram_estimation_error(tmpdir):
    with pytest.raises(ValueError, match='No such directory'):
        m2s.numcalc.calc_and_read_ram(os.path.join(tmpdir, 'bla'), numcalc)


def test_manage_numcalc(tmpdir):
    # copy test directory
    shutil.copytree(
        os.path.join(
            base_dir, 'resources', 'numcalc', 'low_complexity_project'),
        os.path.join(tmpdir, 'project'))

    # run as function
    m2s.numcalc.manage_numcalc(
        os.path.join(tmpdir, 'project'), wait_time=0)

    # check if files exist
    assert len(glob.glob(
        os.path.join(tmpdir, 'project', "manage_numcalc_*txt")))

    base = os.path.join(tmpdir, "project", "NumCalc", "source_1")
    assert os.path.isfile(os.path.join(base, "Memory.txt"))
    for step in range(1, 2):
        assert os.path.isfile(os.path.join(base, f"NC{step}-{step}.out"))


def test__download_windows_build():
    path = m2s.numcalc.numcalc._download_windows_build()
    assert os.path.isfile(path)
    path.endswith("NumCalc.exe")


@pytest.mark.parametrize('replace', [True, False])
def test_build_or_fetch_numcalc(replace):
    path = m2s.numcalc.numcalc.build_or_fetch_numcalc(replace)
    assert os.path.isfile(path)
