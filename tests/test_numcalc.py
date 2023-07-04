import pytest
import subprocess
import shutil
import os
import mesh2scattering as m2s
import glob
import warnings
import numpy.testing as npt
import numpy as np
import filecmp

# directory of this file
base_dir = os.path.dirname(__file__)

# ignore tests for windows since its difficult to build the exe
if os.name == 'nt':
    numcalc = os.path.join(
        m2s.utils.program_root(), "numcalc", "bin", "NumCalc", "NumCalc.exe")
    numcalc_path = os.path.dirname(numcalc)
    warnings.warn(
        ('Under Windows the code is not compiling but an executable is '
         f'expected in {numcalc}.'), UserWarning)

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


@pytest.mark.parametrize("nitermax, use", [
    (0, True), (1, True), (2, True), ([], False)])
def test_numcalc_commandline_nitermax(nitermax, use, tmpdir):
    """Test if command line parameter nitermax behaves as expected"""
    # Setup

    # copy test directory
    shutil.copytree(
        os.path.join(
            base_dir, 'resources', 'test_numcalc', 'project_folder_pspw'),
        os.path.join(tmpdir, 'project'))
    # copy correct input file and rename it to NC.inp
    os.mkdir(os.path.join(tmpdir, 'project', 'NumCalc'))
    os.mkdir(os.path.join(tmpdir, 'project', 'NumCalc', 'source_1'))
    shutil.copyfile(
        os.path.join(
            base_dir, 'resources', 'test_numcalc',
            'ncinp_files', 'NC_commandline_parameters.inp'),
        os.path.join(tmpdir, 'project', 'NumCalc', 'source_1', 'NC.inp'))

    if use:
        commandLineArgument = f' -nitermax {nitermax}'
    else:
        commandLineArgument = ''

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
    out_filepath = os.path.join(tmpdir, "project", "NumCalc",
                                "source_1", out_filename)

    out_file = open(out_filepath)
    out_text = out_file.read()

    if use:
        assert f'CGS solver: number of iterations = {nitermax}' in out_text
        assert 'Warning: Maximum number of iterations is reached!' \
            in out_text
    else:
        assert 'Warning: Maximum number of iterations is reached!' \
            not in out_text


@pytest.mark.parametrize("istart, iend", [
    (False, False), (3, False), (False, 3), (2, 3)])
def test_numcalc_commandline_istart_iend(istart, iend, tmpdir):
    """Test if command line parameters istart and iend behave as expected
    """
    # copy test directory
    shutil.copytree(
        os.path.join(
            base_dir, 'resources', 'test_numcalc', 'project_folder_pspw'),
        os.path.join(tmpdir, 'project'))
    # copy correct input file and rename it to NC.inp
    os.mkdir(os.path.join(tmpdir, 'project', 'NumCalc'))
    os.mkdir(os.path.join(tmpdir, 'project', 'NumCalc', 'source_1'))
    shutil.copyfile(
        os.path.join(
            base_dir, 'resources', 'test_numcalc',
            'ncinp_files', 'NC_commandline_parameters.inp'),
        os.path.join(tmpdir, 'project', 'NumCalc', 'source_1', 'NC.inp'))

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

    out_filepath = os.path.join(tmpdir, "project", "NumCalc",
                                "source_1", out_filename)

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
    """Test NumCalc's RAM estimation using -estimate_ram"""

    # copy test data
    data_cwd = os.path.join(tmpdir, 'SHTF', 'NumCalc', 'source_1')
    data_shtf = os.path.join(
        os.path.dirname(__file__), 'resources', 'SHTF')
    shutil.copytree(data_shtf, os.path.join(tmpdir, 'SHTF'))

    if os.name == 'nt':  # Windows detected
        # run NumCalc and route all printouts to a log file
        subprocess.run(
            f"{numcalc} -estimate_ram",
            stdout=subprocess.DEVNULL, cwd=data_cwd, check=True)
    else:  # elif os.name == 'posix': Linux or Mac detected
        # run NumCalc and route all printouts to a log file
        subprocess.run(
            [f"{numcalc} -estimate_ram"],
            shell=True, stdout=subprocess.DEVNULL, cwd=data_cwd, check=True)

    # check if Memory.txt exists
    assert os.path.isfile(os.path.join(data_cwd, 'Memory.txt'))
    # check if output files still exist
    assert os.path.isfile(os.path.join(
        data_cwd, 'be.out', 'be.1', 'pBoundary'))

    # check Memory.txt against reference
    with open(os.path.join(data_cwd, 'Memory.txt'), 'r') as file:
        current = file.readlines()

    with open(os.path.join(
            data_shtf, 'NumCalc', 'source_1', 'Memory.txt'), 'r') as file:
        reference = file.readlines()

    assert current == reference


def test_defaults(tmpdir):
    """
    Test numcalc manager with default parameters by
    - directly calling the functions
    - running the script that calls the function
    """
    cwd = os.path.dirname(__file__)
    data_shtf = os.path.join(cwd, 'resources', 'SHTF')

    # copy test data to temporary directory and remove test critical data
    shutil.copytree(data_shtf, os.path.join(tmpdir, "SHTF"))
    os.remove(os.path.join(
        tmpdir, "SHTF", "NumCalc", "source_1", "Memory.txt"))
    shutil.rmtree(
        os.path.join(tmpdir, "SHTF", "NumCalc", "source_1", "be.out"))
    shutil.rmtree(os.path.join(tmpdir, "SHTF", "NumCalc", "source_2"))

    # run as function
    m2s.numcalc.manage_numcalc(
        tmpdir, numcalc_path=numcalc_path, wait_time=0)
    # check if files exist
    assert len(glob.glob(os.path.join(tmpdir, "manage_numcalc_*txt")))

    base = os.path.join(tmpdir, "SHTF", "NumCalc", "source_1")
    assert os.path.isfile(os.path.join(base, "Memory.txt"))
    for step in range(1, 61):
        assert os.path.isfile(os.path.join(base, f"NC{step}-{step}.out"))


@pytest.mark.parametrize("boundary", [(False), (True),])
@pytest.mark.parametrize("grid", [(False), (True),])
@pytest.mark.parametrize("scattering", [(False), (True),])
@pytest.mark.parametrize("log", [(False), (True),])
def test_remove_outputs(boundary, grid, scattering, log, tmpdir):
    """Test purging the processed data in Output2HRTF"""
    test_folder = os.path.join('examples', 'project')
    project_path = os.path.join(m2s.utils.repository_root(), test_folder)
    test_dir = os.path.join(tmpdir, os.path.split(test_folder)[-1])
    shutil.copytree(project_path, test_dir)

    m2s.numcalc.remove_outputs(
        test_dir,
        boundary=boundary, grid=grid, scattering=scattering, log=log)

    for subfolder in ['sample', 'reference']:
        assert len(glob.glob(
                os.path.join(test_dir, subfolder, "*.sofa"))) == 0

        # Test boundary and grid
        for source in glob.glob(
                os.path.join(test_dir, subfolder, "NumCalc", "source_*")):
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


def test_read_ram_estimates():

    estimates = m2s.numcalc.read_ram_estimates(os.path.join(
        os.path.dirname(__file__), "resources", "SHTF", "NumCalc", "source_1"))

    assert isinstance(estimates, np.ndarray)
    assert estimates.shape == (60, 3)
    npt.assert_allclose([1.00000e+00, 1.00000e+02, 4.16414e-02], estimates[0])
    npt.assert_allclose([6.00000e+01, 6.00000e+03, 7.22010e-02], estimates[-1])


def test_read_ram_estimates_assertions():
    """test assertions for read_ram_estimates"""

    with pytest.raises(ValueError, match="does not contain a Memory.txt"):
        m2s.numcalc.read_ram_estimates(os.getcwd())


@pytest.mark.parametrize("test_folder", [
    (os.path.join('tests', 'resources', 'project_one_source')),
    ])
def test_calc_and_read_ram_estimation(test_folder, tmpdir):
    project_path = os.path.join(m2s.utils.repository_root(), test_folder)
    project_name = os.path.split(test_folder)[-1]
    test_dir = os.path.join(tmpdir, project_name)
    shutil.copytree(project_path, test_dir)

    ram = m2s.numcalc.calc_and_read_ram(test_dir, numcalc)
    npt.assert_array_almost_equal(ram[0:3, :], np.array([
        [1, 1250, 1.63636, 0, 1],
        [2, 2500, 1.68203, 0, 1],
        [3, 5000, 2.36223, 0, 1]]))
    npt.assert_array_almost_equal(ram[3:, :], np.array([
        [1, 1250, 0.737776, 1, 1],
        [2, 2500, 0.773617, 1, 1],
        [3, 5000, 1.21888, 1, 1]]))


def test_calc_and_read_ram_estimation_error(tmpdir):
    with pytest.raises(ValueError, match='No such directory'):
        m2s.numcalc.calc_and_read_ram(os.path.join(tmpdir, 'bla'), numcalc)


@pytest.mark.parametrize("test_folder", [
    (os.path.join('tests', 'resources', 'project_one_source')),
    (os.path.join('examples', 'project'))
    ])
def test_write_hpc_empty(test_folder, tmpdir):
    project_path = os.path.join(m2s.utils.repository_root(), test_folder)
    test_dir = os.path.join(tmpdir, os.path.split(test_folder)[-1])
    shutil.copytree(project_path, test_dir)
    files = m2s.numcalc.create_hpc_files(
        test_dir, numcalc,
        template_path=None, times='00-03:00:00')
    assert files == []


@pytest.mark.parametrize("test_folder", [
    (os.path.join('examples', 'project'))
    ])
def test_write_hpc_one(test_folder, tmpdir):
    project_path = os.path.join(m2s.utils.repository_root(), test_folder)
    test_dir = os.path.join(tmpdir, os.path.split(test_folder)[-1])
    shutil.copytree(project_path, test_dir)
    # delete source_1 results
    shutil.rmtree(os.path.join(
        test_dir, 'sample', 'NumCalc', 'source_1', 'be.out'))
    os.remove(os.path.join(
        test_dir, 'sample', 'NumCalc', 'source_1', 'Memory.txt'))
    source_dir = os.path.join(
        test_dir, 'sample', 'NumCalc', 'source_1')
    path_all = os.listdir(source_dir)
    extension = ('txt', 'out')
    path_remove = [fn for fn in path_all if fn.endswith(extension)]
    [os.remove(os.path.join(source_dir, filePath)) for filePath in path_remove]

    files = m2s.numcalc.create_hpc_files(
        test_dir, numcalc,
        template_path=None, times='00-03:00:00')
    assert len(files) == 3
    assert files == [
        'project_sample_1.sh', 'project_sample_2.sh', 'project_sample_3.sh']
    for file in files:
        assert filecmp.cmp(
            os.path.join(test_dir, 'hpc', file),
            os.path.join(
                m2s.utils.repository_root(), 'tests', 'references',
                'hpc_1', file)
        )


@pytest.mark.parametrize("test_folder", [
    (os.path.join('examples', 'project'))
    ])
def test_write_hpc_complete(test_folder, tmpdir):
    project_path = os.path.join(m2s.utils.repository_root(), test_folder)
    test_dir = os.path.join(tmpdir, os.path.split(test_folder)[-1])
    shutil.copytree(project_path, test_dir)
    # delete all results
    m2s.numcalc.remove_outputs(test_dir, True, True, True, True)

    files = m2s.numcalc.create_hpc_files(
        test_dir, numcalc,
        template_path=None, times='00-03:00:00')
    assert len(files) == 6
    assert files == [
        'project_sample_1.sh', 'project_sample_2.sh', 'project_sample_3.sh',
        'project_reference_1.sh', 'project_reference_2.sh',
        'project_reference_3.sh']
    for file in files:
        assert filecmp.cmp(
            os.path.join(test_dir, 'hpc', file),
            os.path.join(
                m2s.utils.repository_root(), 'tests', 'references',
                'hpc_2', file)
        )
