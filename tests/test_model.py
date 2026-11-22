import numpy as np
import pytest

from tds_model.model import CompositionPredictor, project_simplex
from tds_model.synthetic import make_dataset


def test_project_simplex():
    x = project_simplex(np.array([50.0, 60.0, -5.0]))
    assert x.sum() == pytest.approx(100)
    assert (x >= 0).all()


@pytest.fixture(scope="module")
def model():
    return CompositionPredictor(n_starts=10).fit(make_dataset(300, seed=1))


def test_roundtrip(model):
    test = make_dataset(5, seed=99)
    errs = []
    for _, r in test.iterrows():
        cands = model.predict({c: r[c] for c in model.prop_cols}, top_k=2)
        comp = cands[0].composition
        assert sum(comp.values()) == pytest.approx(100, abs=0.1)
        errs.append(np.mean([abs(comp[c] - r[c]) for c in model.comp_cols]))
    assert np.mean(errs) < 6.0


def test_missing_properties(model):
    cands = model.predict({"density": 1.12, "flexural_modulus": 4800, "mfi": 14})
    assert cands and cands[0].composition["comp_gf"] > 15
