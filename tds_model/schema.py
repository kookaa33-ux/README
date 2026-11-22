"""물성 항목 / 원부원료 정의.

TDS 에 흔히 기재되는 고분자 컴파운드(PP / 폴리에스터) 물성 항목과 표준 단위를 정의한다.
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
    PropertySpec("mfi", "Melt Flow Index (TDS 시험조건)", "g/10min", True,
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
    # 1.8 MPa HDT 는 파서가 HDT 행의 시험조건을 보고 hdt -> hdt_18 로 분류한다
    PropertySpec("hdt_18", "Heat deflection temp. (1.8MPa)", "°C", False, ()),
    PropertySpec("ul94", "UL94 flammability (HB=0,V-2=1,V-1=2,V-0=3)", "level", False,
                 ("ul94", "ul 94", "ul-94", "flammability", "난연성", "연소성", "난연등급")),
]}

UL94_LEVELS = {"HB": 0, "V-2": 1, "V-1": 2, "V-0": 3, "5VB": 3, "5VA": 3}
UL94_NAMES = {0: "HB", 1: "V-2", 2: "V-1", 3: "V-0"}


def encode_value(key: str, value) -> float:
    """사용자 입력값 정규화 (예: ul94 "V-0" -> 3)."""
    if key == "ul94" and isinstance(value, str):
        v = value.upper().replace(" ", "").replace("V0", "V-0").replace("V1", "V-1").replace("V2", "V-2")
        return float(UL94_LEVELS[v])
    return float(value)


def format_value(key: str, value: float) -> str:
    if key == "ul94":
        lv = int(min(3, max(0, round(value))))
        return f"{UL94_NAMES[lv]} ({value:.2f})"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.3g}"

COMP_PREFIX = "comp_"

# 데모(합성 데이터)용 원부원료: (key, 설명, 밀도 g/cm3)
PP_COMPONENTS = {
    "comp_hpp": ("PP Homopolymer", 0.905),
    "comp_bpp": ("PP Block copolymer (Impact)", 0.900),
    "comp_poe": ("POE elastomer (impact modifier)", 0.870),
    "comp_talc": ("Talc", 2.75),
    "comp_gf": ("Glass fiber", 2.55),
    "comp_caco3": ("CaCO3", 2.70),
    "comp_additive": ("Additives (AO / nucleating / pigment)", 1.10),
}

POLYESTER_COMPONENTS = {
    "comp_pbt": ("PBT resin", 1.31),
    "comp_pet": ("PET resin", 1.38),
    "comp_gf": ("Glass fiber", 2.55),
    "comp_mineral": ("Mineral filler (Talc / Mica)", 2.80),
    "comp_im": ("Impact modifier (EMA-GMA / core-shell)", 0.95),
    "comp_fr": ("Brominated flame retardant", 2.10),
    "comp_sb2o3": ("Sb2O3 (FR synergist)", 5.20),
    "comp_additive": ("Additives (AO / nucleating / lubricant)", 1.10),
}

SYSTEMS = {"pp": PP_COMPONENTS, "polyester": POLYESTER_COMPONENTS}
DEMO_COMPONENTS = PP_COMPONENTS  # 하위 호환
COMPONENT_LABELS = {k: v[0] for comps in SYSTEMS.values() for k, v in comps.items()}
