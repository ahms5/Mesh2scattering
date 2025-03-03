"""Contains functions to calculate scattering
coefficients from SOFA files and save them in new SOFA files.
"""
import pyfar as pf
import numpy as np
import sofar as sf
import os
from .coefficients import scattering_freefield


def calculate_scattering(sample_path, reference_path, project_name):
    """Read pattern data sample_path and reference_path
    and calculate and export the scattering coefficient for each incident angle
    to ``project_name.scattering.sofa``.

    Parameters
    ----------
    sample_path : str, path
        path to the sofa file with the sample data
    reference_path : str, path
        path to the sofa file with the reference data
    project_name : str
        name of the project, the scattering coefficient will be written into#
        the same directory as ``sample_path`` with the given name here:
        ``project_name.scattering.sofa``. The meta data are taken from the
        sample file.
    """
    sofa_sample = sf.read_sofa(sample_path)
    data, _, _ = pf.io.convert_sofa(
        sofa_sample)
    sofa_reference = sf.read_sofa(reference_path)
    data_ref, _, _ = pf.io.convert_sofa(
        sofa_reference)

    # calculate scattering coefficient
    scattering_coefficient = scattering_freefield(
        data, data_ref, sofa_sample.ReceiverWeights)
    scattering_coefficient.freq = scattering_coefficient.freq[:, np.newaxis]

    # write to sofa file
    sofa_sample.Data_Real = np.real(scattering_coefficient.freq)
    sofa_sample.Data_Imag = np.imag(scattering_coefficient.freq)
    sofa_sample.ReceiverPosition = [0, 0, 0]
    sofa_sample.ReceiverWeights = 0

    # write scattering coefficient data to SOFA file
    sf.write_sofa(os.path.join(
        os.path.dirname(sample_path), f'{project_name}.scattering.sofa'),
        sofa_sample)
