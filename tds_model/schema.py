"""물성 항목 / 원부원료 정의.

TDS 에 흔히 기재되는 고분자(PP 컴파운드 기준) 물성 항목과 표준 단위를 정의한다.
학습 데이터 CSV 의 컬럼명은 아래 PROPERTIES 의 key 를 그대로 사용하고,
원부원료 배합비 컬럼은 ``comp_`` 접두사를 붙인다 (wt%, 합계 100).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PropertySpec:
    key: str
    label: str
    unit: str
    log_scale: bool = False          # MFI 처럼 로그 분포인 물성
    keywords: tuple[str, ...] = field(default_factory=tuple)  # TDS 파싱용 (소문자)


PROPERTIES: dict[str, PropertySpec] = {p.key: p for p in [
    PropertySpec("mfi", "Melt Flow Index (230°C/2.16kg)", "g/10min", True,
                 ("melt flow", "melt index", "mfi", "mfr", "용융지수", "용융흐름지수", "흐름지수")),
    PropertySpec("density", "Density / Specific gravity", "g/cm3", False,
                 ("density", "specific gravity", "밀도", "비중")),
    PropertySpec("tensile_strength", "Tensile strength", "MPa", False,
                 ("tensile strength", "tensile stress", "인장강도", "인장 강도")),
    PropertySpec("elongation", "Elongation at break", "%", True,
                 ("elongation", "strain at break", "신율", "연신율")),
    PropertySpec("flexural_modulus", "Flexural modulus", "MPa", False,
                 ("flexural modulus", "flex modulus", "굴곡탄성률", "굴곡 탄성률", "굴곡탄성율")),
    PropertySpec("flexural_strength", "Flexural strength", "MPa", False,
                 ("flexural strength", "굴곡강도", "굴곡 강도")),
    PropertySpec("izod_impact", "Notched Izod impact (23°C)", "kJ/m2", True,
                 ("izod", "notched impact", "충격강도", "아이조드")),
    PropertySpec("hdt", "Heat deflection temp. (0.45MPa)", "°C", False,
                 ("heat deflection", "heat distortion", "hdt", "열변형온도", "열변형 온도")),
]}

COMP_PREFIX = "comp_"

# 데모(합성 데이터)용 원부원료: (key, 설명, 밀도 g/cm3)
DEMO_COMPONENTS = {
    "comp_hpp": ("PP Homopolymer", 0.905),
    "comp_bpp": ("PP Block copolymer (Impact)", 0.900),
    "comp_poe": ("POE elastomer (impact modifier)", 0.870),
    "comp_talc": ("Talc", 2.75),
    "comp_gf": ("Glass fiber", 2.55),
    "comp_caco3": ("CaCO3", 2.70),
    "comp_additive": ("Additives (AO / nucleating / pigment)", 1.10),
}
