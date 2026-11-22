from pathlib import Path

import numpy as np
import pytest

from tds_model.model import CompositionPredictor
from tds_model.schema import encode_value
from tds_model.synthetic_polyester import make_dataset
from tds_model.tds_parser import parse_tds, parse_tds_text

EX = Path(__file__).resolve().parent.parent / "examples"


def test_parse_pbt_tds():
    p = parse_tds(EX / "sample_tds_pbt_gf30_fr_en.txt").properties
    assert p["hdt"] == 219 and p["hdt_18"] == 204
    assert p["ul94"] == 3 and p["flexural_modulus"] == 11200


def test_parse_korean_hdt_loads_and_ul94():
    p = parse_tds(EX / "sample_tds_pbt_pet_im_ko.txt").properties
    assert p["hdt"] == 150 and p["hdt_18"] == 59 and p["ul94"] == 0


@pytest.mark.parametrize("txt,lv", [("UL94 (0.8mm) V-2", 1), ("Flammability UL 94 V0", 3),
                                    ("난연등급 UL94 V-1", 2), ("UL-94 5VA", 3)])
def test_ul94_ratings(txt, lv):
    assert parse_tds_text(txt).properties["ul94"] == lv


def test_encode_value():
    assert encode_value("ul94", "v0") == 3 and encode_value("mfi", "12") == 12


@pytest.fixture(scope="module")
def model():
    return CompositionPredictor(n_starts=10).fit(make_dataset(300, seed=1))


def test_polyester_roundtrip(model):
    test = make_dataset(5, seed=99)
    errs = []
    for _, r in test.iterrows():
        comp = model.predict({c: r[c] for c in model.prop_cols})[0].composition
        assert sum(comp.values()) == pytest.approx(100, abs=0.1)
        errs.append(np.mean([abs(comp[c] - r[c]) for c in model.comp_cols]))
    assert np.mean(errs) < 6.0


def test_fr_grade_detected(model):
    cands = model.predict({"density": 1.70, "flexural_modulus": 11200, "ul94": "V-0", "hdt_18": 204})
    c = cands[0].composition
    assert c["comp_gf"] > 15 and c["comp_fr"] > 5
