import pytest
import pyfar as pf
import numpy as np
from mesh2scattering.input import BoundaryConditionType, BoundaryCondition


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
            kind=BoundaryConditionType.PRES,
            comment=123,
        )

def test_kind_property_returns_enum(material_frequency_data):
    bc = BoundaryCondition(
        values=material_frequency_data,
        kind=BoundaryConditionType.VELO,
        comment="c",
    )
    assert bc.kind == BoundaryConditionType.VELO

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
    ]
)
def test_kind_str_property_returns_expected_string(
        material_data, kind_enum, expected_str):
    bc = BoundaryCondition(
        values=material_data,
        kind=kind_enum,
        comment="test",
    )
    assert bc.kind_str == expected_str
