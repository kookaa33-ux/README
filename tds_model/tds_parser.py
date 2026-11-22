"""TDS / 브로셔(PDF·텍스트)에서 물성값을 추출하고 표준 단위로 환산한다.

규칙 기반 휴리스틱이므로 추출 결과(``warnings`` 포함)를 반드시 확인하고,
필요하면 JSON 으로 직접 물성을 입력해 사용한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .schema import PROPERTIES, UL94_LEVELS

NUM = r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?"

# 시험 규격 / 시험 조건 토큰 — 값으로 오인하지 않도록 제거
STRIP_PATTERNS = [
    r"\b(?:ASTM|ISO|KS\s*M|JIS\s*K|DIN|IEC|UL)\s*[A-Z]?\s*\d+(?:[-/][\dA-Z]+)*",
    r"g\s*/\s*10\s*(?:min|분)",                                       # MFI 단위
    r"\([^)]*\)", r"\[[^\]]*\]",
    r"-?\d+(?:\.\d+)?\s*°?\s*C\s*[/,]\s*\d+(?:\.\d+)?\s*kg[f]?",   # 230°C/2.16kg
    r"\d+(?:\.\d+)?\s*kg[f]?\b(?!\s*[-/·]?\s*cm)",                   # 2.16 kg
    r"\b(?:0\.45|1\.8|1\.82|0\.455|1\.82)\s*MPa\b", r"\b(?:66|264)\s*psi\b",
    r"\b(?:4\.6|18\.6)\s*kgf?\s*/\s*cm2",                            # HDT 하중
    r"\b\d+(?:\.\d+)?\s*mm(?:/min)?\b",                                # 시편두께·속도
    r"@\s*-?\d+\s*°?\s*C", r"\b23\s*°\s*C\b",
]

HDT_18 = re.compile(r"1\.8\d?\s*mpa|264\s*psi|18\.6\s*kg", re.I)
UL_RATING = re.compile(r"(?<![\w])(5\s*V\s*A|5\s*V\s*B|V\s*-?\s*[012]|HB)(?![\w])", re.I)
LOW_TEMP = re.compile(r"(?:-\s?(?:10|18|20|30|40)\s*°?\s*C)|저온")


@dataclass
class ParseResult:
    properties: dict[str, float] = field(default_factory=dict)
    raw_lines: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _normalize(text: str) -> str:
    return (text.replace("²", "2").replace("³", "3").replace("·", "-").replace("•", "-")
            .replace("℃", "°C").replace("º", "°").replace(" ", " "))


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def _convert(key: str, value: float, line: str, res: ParseResult) -> float:
    l = line.lower()
    if key in ("tensile_strength", "flexural_modulus", "flexural_strength"):
        if re.search(r"kg[f]?\s*/\s*cm2", l):
            return value * 0.0980665
        if re.search(r"\bksi\b", l):
            return value * 6.89476
        if re.search(r"\bpsi\b", l):
            return value * 0.00689476
        if re.search(r"\bgpa\b", l):
            return value * 1000
    if key == "izod_impact":
        if re.search(r"kg[f]?\s*-?\s*cm\s*/\s*cm", l):
            value *= 9.80665          # -> J/m
            res.warnings.append("Izod kgf·cm/cm -> J/m -> kJ/m2 (4 mm 시편 가정, ÷4)")
            return value / 4
        if re.search(r"ft\s*-?\s*lb", l):
            res.warnings.append("Izod ft·lb/in -> kJ/m2 (4 mm 시편 가정)")
            return value * 53.38 / 4
        if re.search(r"\bj\s*/\s*m\b(?!2)", l):
            res.warnings.append("Izod J/m -> kJ/m2 (4 mm 시편 가정, ÷4)")
            return value / 4
    if key in ("hdt", "hdt_18") and re.search(r"°\s*f\b", l):
        return (value - 32) * 5 / 9
    if key == "density" and re.search(r"kg\s*/\s*m3", l):
        return value / 1000
    return value


def _score(key: str, line: str) -> int:
    l = line.lower()
    s = 0
    if key == "elongation":
        s += 2 if "break" in l or "파단" in l else (-2 if "yield" in l or "항복" in l else 0)
    if key == "tensile_strength":
        s += 1 if "yield" in l or "항복" in l else 0
    if key == "hdt":
        s += 2 if re.search(r"0\.45\s*mpa|66\s*psi|4\.6\s*kg", l) else 0
    if key == "izod_impact":
        s += 1 if "notch" in l else 0
        s -= 3 if LOW_TEMP.search(line) else 0
    return s


def parse_tds_text(text: str) -> ParseResult:
    res = ParseResult()
    best: dict[str, tuple[int, float, str]] = {}
    for line in _normalize(text).splitlines():
        low = line.lower()
        for key, spec in PROPERTIES.items():
            if not any(k in low for k in spec.keywords):
                continue
            if key == "flexural_strength" and "modulus" in low:
                continue
            if key == "tensile_strength" and ("modulus" in low or "elongation" in low):
                continue
            if key == "ul94":
                m = UL_RATING.search(line)
                if m and key not in best:
                    tok = re.sub(r"\s", "", m.group(1).upper())
                    tok = tok if tok in UL94_LEVELS else tok.replace("V", "V-", 1)
                    best[key] = (0, float(UL94_LEVELS[tok]), line.strip())
                continue
            if key == "hdt" and HDT_18.search(line):
                key = "hdt_18"  # 1.8 MPa 조건 HDT 는 별도 물성
            # 키워드 이후 문자열에서 값 탐색
            pos = min(low.find(k) for k in spec.keywords if k in low)
            tail = line[pos:]
            for p in STRIP_PATTERNS:
                tail = re.sub(p, " ", tail, flags=re.I)
            nums = re.findall(NUM, tail)
            if not nums:
                continue
            val = _to_float(nums[0])
            val = _convert(key, val, line, res)
            sc = _score(key, line)
            if key not in best or sc > best[key][0]:
                best[key] = (sc, val, line.strip())
    for key, (sc, val, line) in best.items():
        res.properties[key] = round(val, 4)
        res.raw_lines[key] = line
    missing = [k for k in PROPERTIES if k not in res.properties]
    if missing:
        res.warnings.append(f"추출되지 않은 물성: {missing}")
    return res


def parse_tds(path: str | Path) -> ParseResult:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        try:
            import pdfplumber
        except ImportError as e:  # pragma: no cover
            raise ImportError("PDF 파싱에는 pdfplumber 가 필요합니다: pip install pdfplumber") from e
        with pdfplumber.open(path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    else:
        text = path.read_text(encoding="utf-8")
    return parse_tds_text(text)
