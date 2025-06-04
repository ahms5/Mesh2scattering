import pytest
import pyfar as pf
import numpy as np
import numpy.testing as npt
from mesh2scattering.input import (
    BoundaryConditionType,
    BoundaryCondition,
    BoundaryConditionMapping,
    )

@pytest.fixture
def material_frequency_data():
    return pf.FrequencyData(
        data=np.ones(5),
        frequencies=np.linspace(100, 1000, 5),
    )

@pytest.fixture
def material_data():
    return pf.FrequencyData(
        data=1,
        frequencies=0,
    )

@pytest.fixture
def material(material_data):
    return BoundaryCondition(
        values=material_data,
        kind=BoundaryConditionType.PRES,
    )

def test_enum_members_exist():
    assert hasattr(BoundaryConditionType, "VELO")
    assert hasattr(BoundaryConditionType, "PRES")
    assert hasattr(BoundaryConditionType, "ADMI")
    assert hasattr(BoundaryConditionType, "IMPE")


def test_enum_values():
    assert BoundaryConditionType.VELO.value == "VELO"
    assert BoundaryConditionType.PRES.value == "PRES"
    assert BoundaryConditionType.ADMI.value == "ADMI"
    assert BoundaryConditionType.IMPE.value == "IMPE"


def test_init_sets_attributes(material_data):
    bc = BoundaryCondition(
        values=material_data,
        kind=BoundaryConditionType.ADMI,
        comment="test comment",
    )
    assert bc.values is material_data
    assert bc.kind == BoundaryConditionType.ADMI
    assert bc.comment == "test comment"

def test_values_setter_accepts_only_frequencydata():
    with pytest.raises(
            ValueError, match="values must be a pyfar.FrequencyData object"):
        BoundaryCondition(
            values=123,
            kind=BoundaryConditionType.IMPE,
            comment="fail",
        )

def test_kind_setter_accepts_only_enum(material_frequency_data):
    with pytest.raises(
            ValueError, match="kind must be a BoundaryConditionType"):
        BoundaryCondition(
            values=material_frequency_data,
            kind="VELO",
            comment="fail",
        )

def test_comment_setter_accepts_only_str(material_frequency_data):
    with pytest.raises(ValueError, match="comment must be a string"):
        BoundaryCondition(
            values=material_frequency_data,
            kind=BoundaryConditionType.IMPE,
            comment=123,
        )

def test_kind_property_returns_enum(material_frequency_data):
    bc = BoundaryCondition(
        values=material_frequency_data,
        kind=BoundaryConditionType.IMPE,
        comment="c",
    )
    assert bc.kind == BoundaryConditionType.IMPE

def test_values_property_returns_freqdata(material_frequency_data):
    bc = BoundaryCondition(
        values=material_frequency_data,
        kind=BoundaryConditionType.IMPE,
        comment="c",
    )
    assert isinstance(bc.values, pf.FrequencyData)

def test_comment_property_returns_str(material_frequency_data):
    bc = BoundaryCondition(
        values=material_frequency_data,
        kind=BoundaryConditionType.ADMI,
        comment="hello",
    )
    assert bc.comment == "hello"

def test_setters_update_values(material_frequency_data, material_data):
    bc = BoundaryCondition(
        values=material_frequency_data,
        kind=BoundaryConditionType.ADMI,
        comment="init",
    )
    bc.values = material_data
    bc.kind = BoundaryConditionType.IMPE
    bc.comment = "updated"
    assert bc.values is material_data
    assert bc.kind == BoundaryConditionType.IMPE
    assert bc.comment == "updated"


@pytest.mark.parametrize(
    ("kind_enum", "expected_str"),
    [
        (BoundaryConditionType.VELO, "VELO"),
        (BoundaryConditionType.PRES, "PRES"),
        (BoundaryConditionType.ADMI, "ADMI"),
        (BoundaryConditionType.IMPE, "IMPE"),
    ],
)
def test_kind_str_property_returns_expected_string(
        material_data, kind_enum, expected_str):
    bc = BoundaryCondition(
        values=material_data,
        kind=kind_enum,
        comment="test",
    )
    assert bc.kind_str == expected_str


@pytest.mark.parametrize(
    ("kind", "values"),
    [
        (BoundaryConditionType.VELO, 'material_data'),
        (BoundaryConditionType.PRES, 'material_data'),
        (BoundaryConditionType.ADMI, 'material_data'),
        (BoundaryConditionType.IMPE, 'material_data'),
        (BoundaryConditionType.ADMI, 'material_frequency_data'),
        (BoundaryConditionType.IMPE, 'material_frequency_data'),
    ],
    )
def test_frequency_dependence_allowed(request, kind, values):
    """Test that frequency dependence is allowed for ADMI and IMPE."""
    values = request.getfixturevalue(values)
    bc = BoundaryCondition(
        values=values,
        kind=kind,
    )
    assert bc.kind is kind
    assert isinstance(bc.values, pf.FrequencyData)
    assert bc.values == values


@pytest.mark.parametrize(
    ("kind", "values"),
    [
        (BoundaryConditionType.VELO, 'material_frequency_data'),
        (BoundaryConditionType.PRES, 'material_frequency_data'),
    ],
    )
def test_frequency_dependence_not_allowed(request, kind, values):
    values = request.getfixturevalue(values)
    match = (
        "Frequency dependent boundary conditions can only be specified "
        "for ADMI and IMPE")
    with pytest.raises(ValueError, match=match):
        BoundaryCondition(
            values=values,
            kind=kind,
        )


def test_MappingBoundaryCondition():
    bcm = BoundaryConditionMapping(10)
    assert bcm._material_list == []
    assert bcm._material_mapping == []
    assert bcm.n_mesh_faces == 10


def test_MappingBoundaryCondition_apply_material(material):
    bcm = BoundaryConditionMapping(10)
    ## add material by indexes
    bcm.apply_material(material, 1, 5)
    assert len(bcm._material_list) == 1
    assert bcm._material_list[0].kind == material.kind
    npt.assert_almost_equal(bcm._material_mapping[0], [1, 5])

    bcm.apply_material(material, 6, 10)
    assert len(bcm._material_list) == 2
    assert bcm._material_list[1].kind == material.kind
    npt.assert_almost_equal(bcm._material_mapping[1], [6, 10])


def test_MappingBoundaryCondition_out(material):
    bcm = BoundaryConditionMapping(12)
    bcm.apply_material(material, 1, 10)
    nc_boundary, nc_frequency_curve = bcm.to_nc_out()
    npt.assert_string_equal(
        nc_boundary,
        "ELEM 1 TO 10 PRES 1.0 -1 0.0 -1\n",
    )
    npt.assert_string_equal(
        nc_frequency_curve,
        "",
    )


def test_MappingBoundaryCondition_out_freqData():
    bcm = BoundaryConditionMapping(2411)
    material2 = BoundaryCondition(
        values=pf.FrequencyData(
            data=np.array([0, 1, 2]),
            frequencies=np.array([0, 1000, 2000]),
        ),
        kind=BoundaryConditionType.ADMI,
    )
    bcm.apply_material(material2, 0, 2411)
    nc_boundary, nc_frequency_curve = bcm.to_nc_out()
    npt.assert_string_equal(
        nc_boundary,
        "ELEM 0 TO 2411 ADMI 1.0 1 1.0 2\n",
    )
    npt.assert_string_equal(
        nc_frequency_curve,
        (
            "2 3\n"
            "1 3\n"
            "0.000000e+00 0.000000e+00 0.0\n"
            "1.000000e+03 1.000000e+00 0.0\n"
            "2.000000e+03 2.000000e+00 0.0\n"
            "2 3\n"
            "0.000000e+00 0.000000e+00 0.0\n"
            "1.000000e+03 0.000000e+00 0.0\n"
            "2.000000e+03 0.000000e+00 0.0\n"
        ),
    )


def test_MappingBoundaryCondition_n_frequency_curves(material):
    bcm = BoundaryConditionMapping(12)
    bcm.apply_material(material, 1, 10)
    assert bcm.n_frequency_curves == 0

    # Add another material with different frequency data
    material2 = BoundaryCondition(
        values=pf.FrequencyData(
            data=np.array([2.0]),
            frequencies=np.array([1000]),
        ),
        kind=BoundaryConditionType.IMPE,
    )
    bcm.apply_material(material2, 11, 12)
    assert bcm.n_frequency_curves == 2
