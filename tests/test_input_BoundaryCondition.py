import pytest
import pyfar as pf
import numpy as np
import numpy.testing as npt
from mesh2scattering.input.bc import (
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
def material():
    return BoundaryCondition(
        values=1,
        kind=BoundaryConditionType.pressure,
    )

def test_enum_members_exist():
    assert hasattr(BoundaryConditionType, "velocity")
    assert hasattr(BoundaryConditionType, "pressure")
    assert hasattr(BoundaryConditionType, "admittance")
    assert hasattr(BoundaryConditionType, "impedance")


def test_enum_values():
    assert BoundaryConditionType.velocity.value == "VELO"
    assert BoundaryConditionType.pressure.value == "PRES"
    assert BoundaryConditionType.admittance.value == "ADMI"
    assert BoundaryConditionType.impedance.value == "IMPE"


def test_init_sets_attributes():
    bc = BoundaryCondition(
        values=1,
        kind=BoundaryConditionType.admittance,
    )
    assert bc.values == 1
    assert bc.kind == BoundaryConditionType.admittance

def test_values_setter_accepts_only_frequencydata():
    with pytest.raises(
            ValueError,
            match="values must be a pyfar.FrequencyData object or a number"):
        BoundaryCondition(
            values='error',
            kind=BoundaryConditionType.admittance,
        )

def test_kind_setter_accepts_only_enum():
    with pytest.raises(
            ValueError, match="kind must be a BoundaryConditionType"):
        BoundaryCondition(
            values=1,
            kind="VELO",
        )


def test_kind_property_returns_enum(material_frequency_data):
    bc = BoundaryCondition(
        values=material_frequency_data,
        kind=BoundaryConditionType.admittance,
    )
    assert bc.kind == BoundaryConditionType.admittance

def test_values_property_returns_freqdata(material_frequency_data):
    bc = BoundaryCondition(
        values=material_frequency_data,
        kind=BoundaryConditionType.admittance,
    )
    assert isinstance(bc.values, pf.FrequencyData)


def test_setters_update_values(material_frequency_data):
    bc = BoundaryCondition(
        values=material_frequency_data,
        kind=BoundaryConditionType.admittance,
    )
    bc.values = 1
    bc.kind = BoundaryConditionType.impedance
    assert bc.values == 1
    assert bc.kind == BoundaryConditionType.impedance


@pytest.mark.parametrize(
    ("kind_enum", "expected_str"),
    [
        (BoundaryConditionType.velocity, "VELO"),
        (BoundaryConditionType.pressure, "PRES"),
        (BoundaryConditionType.admittance, "ADMI"),
        (BoundaryConditionType.impedance, "IMPE"),
    ],
)
def test_kind_str_property_returns_expected_string(
        kind_enum, expected_str):
    bc = BoundaryCondition(
        values=1,
        kind=kind_enum,
    )
    assert bc.kind_str == expected_str


@pytest.mark.parametrize(
    ("kind", "values"),
    [
        (BoundaryConditionType.velocity, 1),
        (BoundaryConditionType.pressure, 1),
        (BoundaryConditionType.admittance, 1),
        (BoundaryConditionType.impedance, 1),
    ],
    )
def test_frequency_dependence_allowed(kind, values):
    """Test that frequency dependence is allowed for ADMI."""
    bc = BoundaryCondition(
        values=values,
        kind=kind,
    )
    assert bc.kind is kind
    assert bc.values == 1
    assert bc.values == values


def test_frequency_dependence_allowed_frequency_data(material_frequency_data):
    """Test that frequency dependence is allowed for ADMI."""
    bc = BoundaryCondition(
        values=material_frequency_data,
        kind=BoundaryConditionType.admittance,
    )
    assert bc.kind is BoundaryConditionType.admittance
    assert isinstance(bc.values, pf.FrequencyData)
    assert bc.values == material_frequency_data


@pytest.mark.parametrize(
    ("kind", "values"),
    [
        (BoundaryConditionType.velocity, 'material_frequency_data'),
        (BoundaryConditionType.pressure, 'material_frequency_data'),
        (BoundaryConditionType.impedance, 'material_frequency_data'),
    ],
    )
def test_frequency_dependence_not_allowed(request, kind, values):
    values = request.getfixturevalue(values)
    match = (
        "Frequency dependent boundary conditions can only be specified "
        "for ADMI")
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
    bcm.add_boundary_condition(material, 1, 5)
    assert len(bcm._material_list) == 1
    assert bcm._material_list[0].kind == material.kind
    npt.assert_almost_equal(bcm._material_mapping[0], [1, 5])

    bcm.add_boundary_condition(material, 6, 10)
    assert len(bcm._material_list) == 2
    assert bcm._material_list[1].kind == material.kind
    npt.assert_almost_equal(bcm._material_mapping[1], [6, 10])


def test_MappingBoundaryCondition_out(material):
    bcm = BoundaryConditionMapping(12)
    bcm.add_boundary_condition(material, 1, 10)
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
        kind=BoundaryConditionType.admittance,
    )
    bcm.add_boundary_condition(material2, 0, 2411)
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
    bcm.add_boundary_condition(material, 1, 10)
    assert bcm.n_frequency_curves == 0

    # Add another material with different frequency data
    material2 = BoundaryCondition(
        values=pf.FrequencyData(
            data=np.array([2.0]),
            frequencies=np.array([1000]),
        ),
        kind=BoundaryConditionType.admittance,
    )
    bcm.add_boundary_condition(material2, 11, 12)
    assert bcm.n_frequency_curves == 2
