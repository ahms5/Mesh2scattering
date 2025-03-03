import pytest


def test_import_mesh2scattering():
    try:
        import mesh2scattering           # noqa
    except ImportError:
        pytest.fail('import mesh2scattering failed')
