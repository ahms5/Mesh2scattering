import pytest
import trimesh


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

