"""폴리에스터(PBT / PET) 컴파운드 데모용 합성 학습 데이터 생성기.

GF 강화, 브롬계 난연(+Sb2O3), 충격보강, 미네랄 충진 등 일반적인 경향을 단순화한
경험식 + 측정 노이즈다. **실제 설계에는 반드시 실측 데이터로 재학습** 해야 한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import POLYESTER_COMPONENTS as COMPS

# 아키타입: 성분별 (min, max) wt%, pet_frac = 수지(PBT+PET) 중 PET 비율 범위
ARCHETYPES = {
    "pbt":        {"comp_im": (0, 5), "pet_frac": (0, 0)},
    "pbt_gf":     {"comp_gf": (10, 45), "pet_frac": (0, 0)},
    "pbt_gf_fr":  {"comp_gf": (10, 35), "comp_fr": (7, 16), "comp_sb2o3": (2, 6), "pet_frac": (0, 0.2)},
    "pbt_fr":     {"comp_fr": (7, 16), "comp_sb2o3": (2, 6), "comp_im": (0, 5), "pet_frac": (0, 0)},
    "blend_gf":   {"comp_gf": (15, 40), "pet_frac": (0.2, 0.6)},
    "pet_gf":     {"comp_gf": (15, 45), "comp_mineral": (0, 5), "pet_frac": (0.85, 1.0)},
    "pbt_im":     {"comp_im": (5, 20), "comp_gf": (0, 20), "pet_frac": (0, 0.3)},
    "pbt_min":    {"comp_mineral": (10, 30), "comp_gf": (0, 15), "pet_frac": (0, 0.3)},
}


def sample_compositions(n: int, rng: np.random.Generator) -> pd.DataFrame:
    names = list(ARCHETYPES)
    rows = []
    for _ in range(n):
        arch = ARCHETYPES[names[rng.integers(len(names))]]
        r = {k: 0.0 for k in COMPS}
        for k, (lo, hi) in arch.items():
            if k.startswith("comp_"):
                r[k] = rng.uniform(lo, hi)
        r["comp_additive"] = rng.uniform(0.2, 1.5)
        rest = 100.0 - sum(r.values())
        pf = rng.uniform(*arch["pet_frac"])
        r["comp_pet"] = rest * pf
        r["comp_pbt"] = rest * (1 - pf)
        rows.append(r)
    return pd.DataFrame(rows)[list(COMPS)].round(2)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def forward_properties(comp: pd.DataFrame, rng: np.random.Generator | None = None,
                       noise: float = 1.0) -> pd.DataFrame:
    w = {k: comp[k].to_numpy() / 100.0 for k in COMPS}
    pbt, pet, gf, mn, im, fr, sb, add = (w[k] for k in COMPS)
    resin = pbt + pet + 1e-9
    fp = pet / resin  # 수지 중 PET 분율

    density = 1.0 / sum(w[k] / d for k, (_, d) in COMPS.items())

    # MFR (250~265°C/2.16kg)
    log_mfi = fp * np.log(12) + (1 - fp) * np.log(22) - 2.0 * gf - 1.2 * mn - 3.0 * im + 0.8 * fr
    mfi = np.exp(log_mfi)

    flex_mod = (fp * 2900 + (1 - fp) * 2500) * (1 - 2.0 * im) \
        * (1 + 11 * gf + 4 * mn + 1.0 * fr + 0.8 * sb)

    ts = (fp * 60 + (1 - fp) * 55) * (1 - 1.6 * im) + 300 * gf * (1 - 0.6 * gf) \
        - 15 * mn - 20 * fr
    flex_str = 1.55 * ts + 20 * mn

    elong = (fp * 30 + (1 - fp) * 50) * (1 + 10 * im) \
        * np.exp(-14 * gf - 7 * mn - 4 * (fr + sb)) + 2.5 * (1 - np.exp(-20 * gf))

    izod = (fp * 4 + (1 - fp) * 5) + 50 * _sig((im - 0.12) * 30) * (1 - 2 * gf) \
        + 18 * gf - 4 * mn - 5 * fr
    izod = np.clip(izod, 1.5, None)

    base18 = fp * 75 + (1 - fp) * 60
    hdt_18 = base18 + (205 + 30 * fp - base18) * (1 - np.exp(-12 * gf)) + 150 * mn - 40 * im + 20 * fr
    base45 = fp * 170 + (1 - fp) * 160
    hdt = base45 + (220 + 20 * fp - base45) * (1 - np.exp(-15 * gf)) + 150 * mn - 100 * im
    hdt = np.minimum(hdt, 245)
    hdt_18 = np.minimum(hdt_18, hdt - 5)

    # UL94: 브롬계 FR + Sb2O3 상승효과 (Sb/FR ≈ 0.3 이상이면 포화)
    eff = fr * (0.4 + 0.6 * np.minimum(1.0, sb / (0.3 * fr + 1e-9)))
    ul_cont = 3 * _sig((eff - 0.09) * 60) - 0.8 * im

    props = pd.DataFrame({
        "mfi": mfi, "density": density, "tensile_strength": ts, "elongation": elong,
        "flexural_modulus": flex_mod, "flexural_strength": flex_str,
        "izod_impact": izod, "hdt": hdt, "hdt_18": hdt_18, "ul94": ul_cont,
    })
    if rng is not None and noise > 0:
        rel = {"mfi": 0.08, "density": 0.003, "tensile_strength": 0.04, "elongation": 0.15,
               "flexural_modulus": 0.04, "flexural_strength": 0.04, "izod_impact": 0.10,
               "hdt": 0.015, "hdt_18": 0.02}
        for k, s in rel.items():
            props[k] *= np.exp(rng.normal(0, s * noise, len(props)))
        props["ul94"] += rng.normal(0, 0.25 * noise, len(props))
    props["ul94"] = np.clip(np.round(props["ul94"]), 0, 3) + 0.0  # 등급은 이산값
    dec = {"mfi": 2, "density": 3, "tensile_strength": 1, "elongation": 1,
           "flexural_modulus": 0, "flexural_strength": 1, "izod_impact": 1,
           "hdt": 0, "hdt_18": 0, "ul94": 0}
    return props.round(dec)


def make_dataset(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    comp = sample_compositions(n, rng)
    props = forward_properties(comp, rng)
    df = pd.concat([props, comp], axis=1)
    df.insert(0, "grade", [f"PES-DEMO-{i:04d}" for i in range(n)])
    return df
