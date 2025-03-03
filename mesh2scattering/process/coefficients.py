"""Coefficient calculation functions for the scattering analysis."""
import numpy as np
import pyfar as pf


def scattering_freefield(
        sample_pressure, reference_pressure, microphone_weights):
    r"""
    Calculate the direction dependent free-field scattering coefficient.

    Uses the Mommertz correlation method [#]_ to calculate the scattering
    coefficient of the input data:

    .. math::
        s = 1 -
            \frac{|\sum_w \underline{p}_{\text{sample}}(\vartheta,\varphi)
            \cdot \underline{p}_{\text{reference}}^*(\vartheta,\varphi)
            \cdot w(\vartheta,\varphi)|^2}
            {\sum_w |\underline{p}_{\text{sample}}(\vartheta,\varphi)|^2
            \cdot w(\vartheta,\varphi) \cdot \sum_w
            |\underline{p}_{\text{reference}}(\vartheta,\varphi)|^2
            \cdot w(\vartheta,\varphi) }

    with the reflected sound pressure of the the sample under investigation
    :math:`\underline{p}_{\text{sample}}`, the reflected sound pressure from
    the reference sample (same dimension as the sample under investigation,
    but with flat surface) :math:`\underline{p}_{\text{reference}}`, the
    area weights of the sampling :math:`w`, and :math:`\vartheta` and
    :math:`\varphi` are the ``colatitude``
    angle and ``azimuth`` angles from the
    :py:class:`~pyfar.classes.coordinates.Coordinates` object.
    In other words, the test sample lies in the x-y-plane.

    Parameters
    ----------
    sample_pressure : :py:class:`~pyfar.classes.audio.FrequencyData`
        Reflected sound pressure or directivity of the test sample. Its cshape
        needs to be (..., microphone_weights.size).
    reference_pressure : :py:class:`~pyfar.classes.audio.FrequencyData`
        Reflected sound pressure or directivity of the
        reference sample. Needs to have the same cshape and frequencies as
        `sample_pressure`.
    microphone_weights : numpy.ndarray
        Array containing the area weights for the microphone positions,
        no normalization required.
        Its shape needs to match the last dimension in the cshape of
        `sample_pressure` and `reference_pressure`.

    Returns
    -------
    scattering_coefficients : :py:class:`~pyfar.classes.audio.FrequencyData`
        The scattering coefficient for each incident direction depending on
        frequency.


    References
    ----------
    .. [#]  E. Mommertz, „Determination of scattering coefficients from the
            reflection directivity of architectural surfaces“, Applied
            Acoustics, Bd. 60, Nr. 2, S. 201-203, June 2000,
            doi: 10.1016/S0003-682X(99)00057-2.

    """
    # check inputs
    if not isinstance(sample_pressure, pf.FrequencyData):
        raise ValueError(
            'sample_pressure has to be a pyfar.FrequencyData object')
    if not isinstance(reference_pressure, pf.FrequencyData):
        raise ValueError(
            'reference_pressure has to be a pyfar.FrequencyData object')
    microphone_weights = np.atleast_1d(
        np.asarray(microphone_weights, dtype=float))
    # if sample_pressure.cshape != reference_pressure.cshape:
    #     raise ValueError(
    #         'sample_pressure and reference_pressure have to have the '
    #         'same cshape.')
    if microphone_weights.shape[0] != sample_pressure.cshape[-1]:
        raise ValueError(
            'the last dimension of sample_pressure needs be same as the '
            'microphone_weights.shape.')
    if not np.allclose(
            sample_pressure.frequencies, reference_pressure.frequencies):
        raise ValueError(
            'sample_pressure and reference_pressure have to have the '
            'same frequencies.')

    # calculate according to mommertz correlation method Equation (5)
    p_sample = np.moveaxis(sample_pressure.freq, -1, 0)
    p_reference = np.moveaxis(reference_pressure.freq, -1, 0)

    p_sample_sum = np.sum(microphone_weights * np.abs(p_sample)**2, axis=-1)
    p_ref_sum = np.sum(microphone_weights * np.abs(p_reference)**2, axis=-1)
    final_shape = np.broadcast_shapes(p_sample.shape, p_reference.shape)
    result = 1
    for x in final_shape:
        result = result * x
    if result < 1e9:
        p_cross_sum = np.sum(
            p_sample * np.conj(p_reference) * microphone_weights, axis=-1)
    else:
        p_cross_sum = np.empty(final_shape[:-1], dtype=complex)
        for i in range(final_shape[0]):
            p_cross_sum[i] = np.sum(
                p_sample[i] * np.conj(p_reference[i]) * microphone_weights,
                axis=-1)
    del p_sample, p_reference

    data_scattering_coefficient \
        = 1 - ((np.abs(p_cross_sum)**2)/(p_sample_sum*p_ref_sum))
    del p_cross_sum, p_sample_sum, p_ref_sum

    scattering_coefficients = pf.FrequencyData(
        np.moveaxis(data_scattering_coefficient, 0, -1),
        sample_pressure.frequencies)

    return scattering_coefficients
