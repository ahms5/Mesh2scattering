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
import numpy.testing as npt
import os



# deprecate in 1.3.0 ----------------------------------------------------------
def test_deprecation_from_parallel_to_plane():
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


def test_deprecation_from_spherical():
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


def test_from_spherical():
    points = pf.samplings.sph_lebedev(sh_order=10)
    with pytest.warns(DeprecationWarning):
        grid = m2s.input.EvaluationGrid.from_spherical(
        points, "Lebedev_N10")
    npt.assert_almost_equal(grid.coordinates.cartesian, points.cartesian)
    assert grid.name == "Lebedev_N10"
    npt.assert_almost_equal(grid.weights, points.weights)
    assert isinstance(grid.faces, np.ndarray)
    npt.assert_array_equal(grid.faces.shape, (336, 3))


@pytest.mark.parametrize("start", [0, 100])
def test_write(start, tmpdir):
    points = pf.samplings.sph_lebedev(sh_order=10)
    with pytest.warns(DeprecationWarning):
        grid = m2s.input.EvaluationGrid.from_spherical(
        points, "Lebedev_N10")

    filename = os.path.join(tmpdir, "test_evaluation_grid")

    grid.export_numcalc(filename, start)
    # read and check Nodes
    with open(filename + "/Nodes.txt", "r") as f_id:
        file = f_id.readlines()
    file = "".join(file)

    x = points.x
    y = points.y
    z = points.z
    first_row = f"{points.csize}\n"
    second_row = f"{start} {x[0]} {y[0]} {z[0]}\n"
    assert file.startswith(first_row+second_row)
    assert file.endswith(
        f"\n{start+points.csize-1} {x[-1]} {y[-1]} {z[-1]}\n")

    # read and check Elements
    with open(filename + "/Elements.txt", "r") as f_id:
        file = f_id.readlines()
    assert len(file) == int(file[0]) + 1
    assert int(file[0]) == grid.faces.shape[0]


@pytest.mark.parametrize("plane", ['xy', 'yz', 'xz'])
def test_from_parallel_to_plane(plane):
    x = np.arange(0, 50, 10)
    y = x
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    if plane == 'xy':
        points = pf.Coordinates(xx, yy, 0, weights=xx)
    elif plane == 'xz':
        points = pf.Coordinates(xx, 0, yy, weights=xx)
    elif plane == 'yz':
        points = pf.Coordinates(0, xx, yy, weights=xx)

    with pytest.warns(DeprecationWarning):
        grid = m2s.input.EvaluationGrid.from_parallel_to_plane(
            points, plane, f"{plane}_plane")

    assert grid.name == f"{plane}_plane"
    npt.assert_almost_equal(grid.coordinates.cartesian, points.cartesian)
    npt.assert_almost_equal(grid.weights, points.weights)
    assert isinstance(grid.faces, np.ndarray)
    npt.assert_array_equal(grid.faces.shape, (32, 3))


def test_from_parallel_to_plane_invalid_plane():
    points = pf.Coordinates(0, 0, 0, weights=1)

    with pytest.raises(
            ValueError,
            match="plane must be 'xy', 'yz', or 'xz'."):
        m2s.input.EvaluationGrid.from_parallel_to_plane(
            points, 'xyz', "plane")


# functions removed -----------------------------------------------------------

def test_removed_from_parallel_to_plane():
    coordinates = pf.Coordinates(
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        weights=[1, 1, 1, 1],
    )

    match = "type object 'EvaluationGrid' has no attribute"
    if version.parse(m2s.__version__) >= version.parse('1.3.0'):
        with pytest.raises(AttributeError, match=match):
            m2s.input.EvaluationGrid.from_parallel_to_plane(
                coordinates, 'xy', 'test')


def test_removed_from_spherical():
    coordinates = pf.samplings.sph_gaussian(sh_order=5)

    match = "type object 'EvaluationGrid' has no attribute"
    if version.parse(m2s.__version__) >= version.parse('1.3.0'):
        with pytest.raises(AttributeError, match=match):
            m2s.input.EvaluationGrid.from_spherical(
                coordinates, 'test')


def test_removed_EvaluationGrid_faces():
    coordinates = pf.Coordinates(
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        weights=[1, 1, 1, 1],
    )
    faces = np.array([[0, 1, 2], [1, 2, 3]])

    match = "type object 'EvaluationGrid' has no attribute"
    if version.parse(m2s.__version__) >= version.parse('1.3.0'):
        with pytest.raises(AttributeError, match=match):
            m2s.input.EvaluationGrid(coordinates, faces, 'test')

