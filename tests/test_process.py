import os
import numpy.testing as npt
import sofar as sf
import mesh2scattering as m2s


def test_scattering_calculation():
    base_path = os.path.join(
        m2s.utils.repository_root(), 'tests', 'resources', 'output')
    path_sample = os.path.join(base_path, 'sine_gaussian_63.pressure.sofa')
    path_ref = os.path.join(base_path, 'reference_gaussian_63.pressure.sofa')
    m2s.process.calculate_scattering(path_sample, path_ref, 'test')
    assert os.path.isfile(os.path.join(base_path, 'test.scattering.sofa'))
    sofa = sf.read_sofa(os.path.join(base_path, 'test.scattering.sofa'))

    npt.assert_array_less(sofa.Data_Real, 1)
    npt.assert_array_less(0, sofa.Data_Real)
    npt.assert_array_equal(sofa.Data_Imag, 0)
    # scattering coefficients are low, because Lambda/lambda < 0.5
    npt.assert_array_less(sofa.Data_Real, 0.04)
    # higher frequency has higher scattering coefficient
    assert sofa.Data_Real[0, 0, 0] < sofa.Data_Real[0, 0, 1]


def test_scattering_calculation_baseline():
    base_path = os.path.join(
        m2s.utils.repository_root(), 'tests', 'resources', 'output')
    path_sample = os.path.join(base_path, 'sine_gaussian_63.pressure.sofa')
    path_ref = os.path.join(base_path, 'reference_gaussian_63.pressure.sofa')
    m2s.process.calculate_scattering(path_sample, path_ref, 'test')
    assert os.path.isfile(os.path.join(base_path, 'test.scattering.sofa'))
    sofa = sf.read_sofa(os.path.join(base_path, 'test.scattering.sofa'))
    sofa_baseline = sf.read_sofa(
        os.path.join(base_path, 'baseline.scattering.sofa'))

    npt.assert_array_equal(sofa.Data_Real, sofa_baseline.Data_Real)
    npt.assert_array_equal(sofa.Data_Imag, sofa_baseline.Data_Imag)
    npt.assert_array_equal(sofa.SourcePosition, sofa_baseline.SourcePosition)
    npt.assert_array_equal(sofa.SourceWeights, sofa_baseline.SourceWeights)
    npt.assert_array_equal(
        sofa.ReceiverPosition, sofa_baseline.ReceiverPosition)
    npt.assert_array_equal(
        sofa.ReceiverWeights, sofa_baseline.ReceiverWeights)
