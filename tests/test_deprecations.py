"""
Test deprecations. For each deprecation two things must be tested:
1. Is a proper warning raised. This is done using
   with pytest.warns(PyfarDeprecationWarning, match="some text"):
       call_of_function()
2. Was the function properly deprecated. This is done using:
   if version.parse(pf.__version__) >= version.parse('0.5.0'):
        with pytest.raises(AttributeError):
            # remove get_nearest_k() from pyfar 0.5.0!
            coords.get_nearest_k(1, 0, 0).

"""
from packaging import version

import pytest
import numpy as np
import pyfar as pf
import mesh2scattering as m2s



# deprecate in 1.3.0 ----------------------------------------------------------
def test_from_parallel_to_plane():
    coordinates = pf.Coordinates(
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        weights=[1, 1, 1, 1],
    )
    with pytest.warns(DeprecationWarning):
        m2s.input.EvaluationGrid.from_parallel_to_plane(
            coordinates, 'xy', 'test')

    match = "type object 'EvaluationGrid' has no attribute"
    if version.parse(m2s.__version__) >= version.parse('1.3.0'):
        with pytest.raises(AttributeError, match=match):
            m2s.input.EvaluationGrid.from_parallel_to_plane(
                coordinates, 'xy', 'test')


def test_from_spherical():
    coordinates = pf.samplings.sph_gaussian(sh_order=5)

    with pytest.warns(DeprecationWarning):
        m2s.input.EvaluationGrid.from_spherical(
            coordinates, 'test')

    match = "type object 'EvaluationGrid' has no attribute"
    if version.parse(m2s.__version__) >= version.parse('1.3.0'):
        with pytest.raises(AttributeError, match=match):
            m2s.input.EvaluationGrid.from_spherical(
                coordinates, 'test')


def test_EvaluationGrid_faces():
    coordinates = pf.Coordinates(
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        weights=[1, 1, 1, 1],
    )
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    match = (
        "will be deprecated in v1.3.0, because NumCalc ")

    with pytest.warns(DeprecationWarning, match=match):
            m2s.input.EvaluationGrid(coordinates, faces, 'test')

    match = "type object 'EvaluationGrid' has no attribute"
    if version.parse(m2s.__version__) >= version.parse('1.3.0'):
        with pytest.raises(AttributeError, match=match):
            m2s.input.EvaluationGrid(coordinates, faces, 'test')