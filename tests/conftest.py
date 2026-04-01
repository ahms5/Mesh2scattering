import pytest
import trimesh
import numpy as np
import pyfar as pf


@pytest.fixture
def simple_mesh():
    """Return a simple triangle mesh.

    Returns
    -------
    trimesh.Trimesh
        simple triangle
    """
    return trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])


@pytest.fixture
def simple_hemispherical_grid():
    """Return a simple hemispherical grid.

    Returns
    -------
    mesh2scattering.input.EvaluationGrid
        simple hemispherical grid
    """
    delta_phi = 30
    delta_theta = 30
    radius = 1
    # get the angles
    phi_angles = np.arange(0, 360, delta_phi)
    theta_angles = np.arange(delta_theta, 180, delta_theta)

    # stack the angles
    phi = np.tile(phi_angles, theta_angles.size)
    theta = np.repeat(theta_angles, phi_angles.size)

    # add North and South Pole
    phi = np.concatenate(([0], phi, [0]))
    theta = np.concatenate(([0], theta, [180]))

    # make Coordinates object
    sampling = pf.Coordinates.from_spherical_colatitude(
        phi/180*np.pi, theta/180*np.pi, radius,
        weights=np.ones_like(phi),
        comment='equal angle spherical sampling grid')
    return sampling


@pytest.fixture
def simple_hemispherical_grid_dense():
    """Return a simple hemispherical grid.

    Returns
    -------
    mesh2scattering.input.EvaluationGrid
        simple hemispherical grid
    """
    delta_phi = 5
    delta_theta = 5
    radius = 1
    # get the angles
    phi_angles = np.arange(0, 360, delta_phi)
    theta_angles = np.arange(delta_theta, 180, delta_theta)

    # stack the angles
    phi = np.tile(phi_angles, theta_angles.size)
    theta = np.repeat(theta_angles, phi_angles.size)

    # add North and South Pole
    phi = np.concatenate(([0], phi, [0]))
    theta = np.concatenate(([0], theta, [180]))

    # make Coordinates object
    sampling = pf.Coordinates.from_spherical_colatitude(
        phi/180*np.pi, theta/180*np.pi, radius,
        weights=np.sin(theta/180*np.pi),
        comment='equal angle spherical sampling grid')
    return sampling
