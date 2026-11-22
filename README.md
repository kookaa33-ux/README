# TDS 물성 → 원부원료 배합 예측 모델

TDS(Technical Data Sheet)·브로셔에 기재된 고분자 물성값으로 원부원료 배합비(wt%)를
역으로 추정하는 모델입니다. PP 컴파운드(Homo/Block PP, POE, Talc, 유리섬유, CaCO3, 첨가제)를
기본 예시로 구성했으며, 학습 데이터의 컬럼만 바꾸면 다른 수지·원료 체계에도 그대로 쓸 수 있습니다.

> ⚠️ 저장소에 포함된 `data/demo_training_data.csv` 는 일반적인 경향을 단순화한 경험식으로 만든
> **합성 데모 데이터**입니다. 실제 배합 설계에 쓰려면 사내 실측 배합-물성 데이터로 재학습해야 합니다.

## 구성

```
tds_model/
  schema.py      물성 항목·표준 단위·TDS 키워드(영/한), 데모 원료 정의
  tds_parser.py  TDS(PDF/텍스트) 물성 추출 + 단위 환산 (kgf/cm², psi, J/m, kgf·cm/cm, °F …)
  model.py       역예측 모델 (CompositionPredictor)
  synthetic.py   합성 데모 데이터 생성기
  cli.py         명령행 도구
examples/        데모 TDS (영문·국문)
tests/           pytest 테스트
```

## 동작 방식

물성에서 배합을 구하는 역문제는 같은 물성을 내는 배합이 여러 개일 수 있어서, 두 가지 방법을 함께 씁니다.

1. **Direct inverse model**: 물성에서 배합비를 바로 회귀합니다 (HistGradientBoosting).
   학습할 때 물성 일부를 무작위로 지워서 학습하므로, TDS에 없는 항목이 있어도 예측할 수 있습니다.
2. **Forward 대리모델 + 최적화**: 배합에서 물성을 계산하는 Gaussian Process 모델을 학습합니다.
   그다음 입력 물성과의 오차(σ 단위)가 가장 작은 배합을 찾습니다.
   이 탐색은 0 ≤ wt%, 합계 = 100% 조건과 학습 데이터 범위 안에서 SLSQP로 하며,
   시작점은 Direct 예측값, 가까운 학습 배합, 무작위 배합 여러 개입니다.
   결과로 **서로 다른 후보 배합 Top‑K** 를 냅니다. 각 후보에는 **역산한 예측 물성과 적합오차**가 함께 나옵니다.

## 사용법

```bash
pip install -r requirements.txt

# 1) (데모) 합성 학습 데이터 생성 — 실데이터가 있으면 생략
python -m tds_model.cli demo-data --out data/demo_training_data.csv

# 2) 학습
python -m tds_model.cli train --data data/demo_training_data.csv --model models/model.joblib

# 3) 예측 — TDS 파일(PDF/TXT) 또는 물성 JSON
python -m tds_model.cli predict --tds examples/sample_tds_ko.txt
python -m tds_model.cli predict --props '{"mfi": 14, "density": 1.12, "flexural_modulus": 4800}' --top-k 3
python -m tds_model.cli predict --tds my_tds.pdf --json      # JSON 출력

# 4) 홀드아웃 평가 (--drop N : 물성 N개 누락 시나리오)
python -m tds_model.cli evaluate --data data/demo_training_data.csv --drop 3
```

Python API:

```python
from tds_model import CompositionPredictor, parse_tds
m = CompositionPredictor.load("models/model.joblib")
props = parse_tds("tds.pdf").properties
for c in m.predict(props, top_k=3, weights={"flexural_modulus": 2.0}):
    print(c.composition, c.error)
```

## 실데이터로 학습하기

CSV 한 행이 한 grade/배합입니다. 컬럼 구성은 다음과 같습니다.

| 컬럼 | 설명 |
|---|---|
| `grade` (선택) | 제품명 |
| `mfi`, `density`, `tensile_strength`, `elongation`, `flexural_modulus`, `flexural_strength`, `izod_impact`, `hdt` | 물성. 일부만 있어도 되고 빈칸도 허용합니다. 단위는 `schema.py` 기준입니다 (MPa, kJ/m², °C@0.45MPa 등) |
| `comp_<원료명>` | 원료별 wt%. 개수와 이름은 자유이며 합계는 100입니다 |

물성 항목을 추가하려면 `schema.py` 의 `PROPERTIES` 에 항목과 TDS 키워드를 추가하면 됩니다.

## 데모 성능 (합성 데이터 1,200건 중 60건 홀드아웃)

| 원료 | Direct MAE | 최적화 Top‑1 MAE |
|---|---|---|
| PP Homo | 6.2 | 6.5 |
| PP Block | 7.4 | 7.6 |
| POE | 1.3 | 1.5 |
| Talc | 1.6 | 1.7 |
| Glass fiber | 0.5 | 0.4 |
| CaCO3 | 1.3 | 1.4 |
| Additive | 0.6 | 0.7 |
| **평균 (wt%)** | **2.7** | **2.8** |

물성 3개를 무작위로 뺀 경우 평균 MAE는 Direct 3.9 wt%, 최적화 4.6 wt% 입니다.
데모 TDS 2건(`examples/`)은 정답 배합을 약 1~3 wt% 이내로 복원합니다.

## 한계와 주의사항

- **식별 불가능성**: Homo/Block PP 비율, 소량 첨가제처럼 물성에 미치는 영향이 작거나 서로 비슷한
  원료는 본질적으로 구분이 어렵습니다. 하나의 답보다 Top‑K 후보와 예측 물성을 함께 보고 판단하세요.
- **학습 범위 밖 외삽 불가**: 학습 데이터에 없는 원료나 배합 범위는 예측할 수 없습니다.
  해당 원료·범위의 데이터를 먼저 추가해야 합니다.
- **TDS 파싱은 규칙 기반**입니다. 표 레이아웃에 따라 틀릴 수 있으므로 출력되는 추출값과 `[주의]` 메시지를 확인하세요.
  Izod J/m → kJ/m² 환산은 4 mm 시편 가정의 근사치입니다. HDT는 0.45 MPa 기준이고, 시험 규격(ISO/ASTM)
  차이는 보정하지 않습니다.
- TDS 값은 대표값(typical)이므로 실제 배합과 오차가 있을 수 있습니다.
