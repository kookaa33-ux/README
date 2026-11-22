from pathlib import Path

import pytest

from tds_model.tds_parser import parse_tds, parse_tds_text

EX = Path(__file__).resolve().parent.parent / "examples"


def test_english_tds():
    p = parse_tds(EX / "sample_tds_en.txt").properties
    assert p == pytest.approx({"mfi": 10.5, "density": 1.04, "tensile_strength": 28,
                               "elongation": 85, "flexural_modulus": 1780,
                               "flexural_strength": 40, "izod_impact": 13, "hdt": 110})


def test_korean_tds_unit_conversion():
    p = parse_tds(EX / "sample_tds_ko.txt").properties
    assert p["mfi"] == pytest.approx(14.6)
    assert p["tensile_strength"] == pytest.approx(785 * 0.0980665, rel=1e-3)
    assert p["flexural_modulus"] == pytest.approx(49700 * 0.0980665, rel=1e-3)
    assert p["izod_impact"] == pytest.approx(5.4 * 9.80665 / 4, rel=1e-3)
    assert p["hdt"] == 143


def test_hdt_prefers_045mpa_and_izod_prefers_room_temp():
    txt = """HDT (1.8 MPa) ISO 75 °C 60
HDT (0.45 MPa) ISO 75 °C 95
Notched Izod -30°C kJ/m2 3
Notched Izod 23°C kJ/m2 9
Tensile strength psi 4000"""
    p = parse_tds_text(txt).properties
    assert p["hdt"] == 95
    assert p["izod_impact"] == 9
    assert p["tensile_strength"] == pytest.approx(4000 * 0.00689476, rel=1e-3)
