"""데모용 합성 학습 데이터 생성기.

실제 사내 배합-물성 데이터가 없을 때 파이프라인을 검증하기 위한 것으로,
PP 컴파운드의 일반적인 경향(혼합 법칙, 필러 보강, 엘라스토머 강인화 등)을
단순화한 경험식 + 측정 노이즈로 만들었다. **실제 설계에는 반드시 실측 데이터로
재학습** 해야 한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import PP_COMPONENTS as DEMO_COMPONENTS

# 배합 아키타입: (이름, 성분별 (min, max) wt%) — 나머지는 PP 로 채운다
ARCHETYPES = {
    "unfilled":    {"comp_poe": (0, 5)},
    "talc":        {"comp_talc": (10, 40), "comp_poe": (0, 10)},
    "gf":          {"comp_gf": (10, 40)},
    "tpo":         {"comp_poe": (10, 30), "comp_talc": (0, 25)},
    "caco3":       {"comp_caco3": (10, 40), "comp_poe": (0, 8)},
    "hybrid":      {"comp_talc": (5, 20), "comp_gf": (5, 20), "comp_poe": (0, 10)},
}


def sample_compositions(n: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    names = list(ARCHETYPES)
    for _ in range(n):
        arch = ARCHETYPES[names[rng.integers(len(names))]]
        r = {k: 0.0 for k in DEMO_COMPONENTS}
        for k, (lo, hi) in arch.items():
            r[k] = rng.uniform(lo, hi)
        r["comp_additive"] = rng.uniform(0.2, 1.5)
        rest = 100.0 - sum(r.values())
        split = rng.beta(0.6, 0.6)  # Homo / Block 비율
        r["comp_hpp"] = rest * split
        r["comp_bpp"] = rest * (1 - split)
        rows.append(r)
    return pd.DataFrame(rows)[list(DEMO_COMPONENTS)].round(2)


def forward_properties(comp: pd.DataFrame, rng: np.random.Generator | None = None,
                       noise: float = 1.0) -> pd.DataFrame:
    """배합(wt%) -> 물성 경험식."""
    w = {k: comp[k].to_numpy() / 100.0 for k in DEMO_COMPONENTS}
    hpp, bpp, poe, talc, gf, caco3, add = (w[k] for k in DEMO_COMPONENTS)
    resin = hpp + bpp + 1e-9
    f_h = hpp / resin  # 수지 중 homo 분율

    # 밀도: 역 혼합 법칙
    density = 1.0 / sum(w[k] / d for k, (_, d) in DEMO_COMPONENTS.items())

    # MFI: 수지 로그혼합, 엘라스토머/필러가 감소시킴
    log_mfi = (f_h * np.log(25) + (1 - f_h) * np.log(12)) * resin / (resin + poe) \
        + np.log(1.0) * poe / (resin + poe)
    log_mfi -= 1.2 * talc + 1.8 * gf + 0.9 * caco3
    mfi = np.exp(log_mfi)

    # 굴곡탄성률
    base_mod = (f_h * 1650 + (1 - f_h) * 1150) * resin + 15 * poe
    flex_mod = base_mod / (resin + poe + 1e-9) * (1 - 1.6 * poe)
    flex_mod = flex_mod * (1 + 3.2 * talc + 6.5 * gf + 1.3 * caco3 + 0.4 * add)

    # 인장강도
    ts = (f_h * 35 + (1 - f_h) * 25) * (1 - 1.5 * poe) + 140 * gf + 5 * talc - 10 * caco3

    flex_str = 1.25 * ts + 25 * talc + 60 * gf

    # 신율
    elong = (f_h * 120 + (1 - f_h) * 350) * (1 + 4 * poe) * np.exp(-6 * talc - 14 * gf - 4 * caco3)

    # 노치 아이조드 충격 (kJ/m2): 엘라스토머 강인화 S-curve
    izod = (f_h * 3 + (1 - f_h) * 10) * (1 - 1.0 * talc - 0.5 * caco3) \
        + 55 / (1 + np.exp(-(poe - 0.15) * 25)) * (1 - 1.5 * talc) + 30 * gf
    izod = np.clip(izod, 1.0, None)

    # HDT 0.45MPa
    hdt = (f_h * 105 + (1 - f_h) * 90) - 120 * poe + 110 * talc + 200 * gf * (1 - 1.2 * gf) + 50 * caco3
    hdt = np.minimum(hdt, 162)

    props = pd.DataFrame({
        "mfi": mfi, "density": density, "tensile_strength": ts, "elongation": elong,
        "flexural_modulus": flex_mod, "flexural_strength": flex_str,
        "izod_impact": izod, "hdt": hdt,
    })
    if rng is not None and noise > 0:
        rel = {"mfi": 0.08, "density": 0.003, "tensile_strength": 0.04, "elongation": 0.15,
               "flexural_modulus": 0.04, "flexural_strength": 0.04, "izod_impact": 0.10, "hdt": 0.02}
        for k, s in rel.items():
            props[k] *= np.exp(rng.normal(0, s * noise, len(props)))
    dec = {"mfi": 2, "density": 3, "tensile_strength": 1, "elongation": 0,
           "flexural_modulus": 0, "flexural_strength": 1, "izod_impact": 1, "hdt": 0}
    return props.round(dec)


def make_dataset(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    comp = sample_compositions(n, rng)
    props = forward_properties(comp, rng)
    df = pd.concat([props, comp], axis=1)
    df.insert(0, "grade", [f"DEMO-{i:04d}" for i in range(n)])
    return df
