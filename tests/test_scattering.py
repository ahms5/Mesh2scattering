import pytest
import numpy as np

import mesh2scattering as m2s
import pyfar as pf


@pytest.fixture
def half_sphere():
    """Return 42th order gaussian sampling for the half sphere and radius 1.

    Returns
    -------
    :py:class:`~pyfar.classes.coordinates.Coordinates`
        half sphere sampling grid
    """
    mics = pf.samplings.sph_gaussian(42)
    # delete lower part of sphere
    return mics[mics.elevation <= np.pi/2]


@pytest.fixture
def quarter_half_sphere():
    """Return 10th order gaussian sampling for the quarter half sphere
    and radius 1.

    Returns
    -------
    :py:class:`~pyfar.classes.coordinates.Coordinates`
        quarter half sphere sampling grid
    """
    incident_directions = pf.samplings.sph_gaussian(10)
    incident_directions = incident_directions[
        incident_directions.elevation <= np.pi/2]
    return incident_directions[
        incident_directions.azimuth <= np.pi/2]


@pytest.fixture
def pressure_data_mics(half_sphere):
    """Return a sound pressure data example, with sound pressure 0 and
    two frequency bins.

    Parameters
    ----------
    half_sphere : :py:class:`~pyfar.classes.coordinates.Coordinates`
        half sphere sampling grid for mics

    Returns
    -------
    pyfar.FrequencyData
        output sound pressure data
    """
    frequencies = [200, 300]
    shape_new = np.append(half_sphere.cshape, len(frequencies))
    return pf.FrequencyData(np.zeros(shape_new), frequencies)


@pytest.fixture
def pressure_data_mics_incident_directions(
        half_sphere, quarter_half_sphere):
    """Return a sound pressure data example, with sound pressure 0 and
    two frequency bins.

    Parameters
    ----------
    half_sphere : :py:class:`~pyfar.classes.coordinates.Coordinates`
        half sphere sampling grid for mics
    quarter_half_sphere : :py:class:`~pyfar.classes.coordinates.Coordinates`
        quarter half sphere sampling grid for incident directions

    Returns
    -------
    pyfar.FrequencyData
        output sound pressure data
    """
    frequencies = [200, 300]
    shape_new = np.append(
        quarter_half_sphere.cshape, half_sphere.cshape)
    shape_new = np.append(shape_new, len(frequencies))
    return pf.FrequencyData(np.zeros(shape_new), frequencies)


def test_scattering_freefield_1(
        half_sphere, pressure_data_mics):
    mics = half_sphere
    p_sample = pressure_data_mics.copy()
    p_sample.freq.fill(1)
    p_reference = pressure_data_mics.copy()
    p_sample.freq[5, :] = 0
    p_reference.freq[5, :] = np.sum(p_sample.freq.flatten())/2
    s = m2s.process.scattering_freefield(p_sample, p_reference, mics.weights)
    np.testing.assert_allclose(s.freq, 1)


def test_scattering_freefield_wrong_input(
        half_sphere, pressure_data_mics):
    mics = half_sphere
    p_sample = pressure_data_mics.copy()
    p_reference = pressure_data_mics.copy()

    with pytest.raises(ValueError, match='sample_pressure'):
        m2s.process.scattering_freefield(1, p_reference, mics.weights)
    with pytest.raises(ValueError, match='reference_pressure'):
        m2s.process.scattering_freefield(p_sample, 1, mics.weights)
    with pytest.raises(ValueError, match='microphone_weights'):
        m2s.process.scattering_freefield(p_sample, p_reference, 1)
    with pytest.raises(ValueError, match='shape'):
        m2s.process.scattering_freefield(
            p_sample[:-2, ...], p_reference, mics.weights)
    with pytest.raises(ValueError, match='microphone_weights'):
        m2s.process.scattering_freefield(
            p_sample, p_reference, mics.weights[:10])
    p_sample.frequencies[0] = 1
    with pytest.raises(ValueError, match='same frequencies'):
        m2s.process.scattering_freefield(p_sample, p_reference, mics.weights)


def test_scattering_freefield_05(
        half_sphere, pressure_data_mics):
    mics = half_sphere
    p_sample = pressure_data_mics.copy()
    p_reference = pressure_data_mics.copy()
    p_sample.freq[7, :] = 1
    p_sample.freq[28, :] = 1
    p_reference.freq[7, :] = 1
    s = m2s.process.scattering_freefield(
        p_sample, p_reference, np.ones_like(mics.weights))
    np.testing.assert_allclose(s.freq, 0.5)


def test_scattering_freefield_0(
        half_sphere, pressure_data_mics):
    mics = half_sphere
    p_sample = pressure_data_mics.copy()
    p_reference = pressure_data_mics.copy()
    p_reference.freq[5, :] = 1
    p_sample.freq[5, :] = 1
    s = m2s.process.scattering_freefield(p_sample, p_reference, mics.weights)
    np.testing.assert_allclose(s.freq, 0)
    assert s.freq.shape[-1] == p_sample.n_bins


def test_scattering_freefield_0_with_incident(
        half_sphere, quarter_half_sphere,
        pressure_data_mics_incident_directions):
    mics = half_sphere
    incident_directions = quarter_half_sphere
    p_sample = pressure_data_mics_incident_directions.copy()
    p_reference = pressure_data_mics_incident_directions.copy()
    p_reference.freq[:, 2, :] = 1
    p_sample.freq[:, 2, :] = 1
    s = m2s.process.scattering_freefield(p_sample, p_reference, mics.weights)
    np.testing.assert_allclose(s.freq, 0)
    assert s.freq.shape[-1] == p_sample.n_bins
    assert s.cshape == incident_directions.cshape


def test_scattering_freefield_1_with_incidence(
        half_sphere, quarter_half_sphere,
        pressure_data_mics_incident_directions):
    mics = half_sphere
    incident_directions = quarter_half_sphere
    p_sample = pressure_data_mics_incident_directions.copy()
    p_reference = pressure_data_mics_incident_directions.copy()
    p_reference.freq[:, 2, :] = 1
    p_sample.freq[:, 3, :] = 1
    s = m2s.process.scattering_freefield(p_sample, p_reference, mics.weights)
    np.testing.assert_allclose(s.freq, 1)
    assert s.freq.shape[-1] == p_sample.n_bins
    assert s.cshape == incident_directions.cshape


def test_scattering_freefield_05_with_inci(
        half_sphere, quarter_half_sphere,
        pressure_data_mics_incident_directions):
    mics = half_sphere
    incident_directions = quarter_half_sphere
    p_sample = pressure_data_mics_incident_directions.copy()
    p_reference = pressure_data_mics_incident_directions.copy()
    p_sample.freq[:, 7, :] = 1
    p_sample.freq[:, 28, :] = 1
    p_reference.freq[:, 7, :] = 1
    s = m2s.process.scattering_freefield(
        p_sample, p_reference, np.ones_like(mics.weights))
    np.testing.assert_allclose(s.freq, 0.5)
    assert s.freq.shape[-1] == p_sample.n_bins
    assert s.cshape == incident_directions.cshape
