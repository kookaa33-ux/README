"""명령행 도구.

    python -m tds_model.cli demo-data  --out data/demo_training_data.csv
    python -m tds_model.cli train      --data data/demo_training_data.csv --model models/model.joblib
    python -m tds_model.cli evaluate   --data data/demo_training_data.csv
    python -m tds_model.cli predict    --model models/model.joblib --tds examples/sample_tds_en.txt
    python -m tds_model.cli predict    --model models/model.joblib --props '{"mfi": 14, "density": 1.05}'
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from .model import CompositionPredictor
from .schema import DEMO_COMPONENTS, PROPERTIES
from .tds_parser import parse_tds


def _label(c: str) -> str:
    return DEMO_COMPONENTS.get(c, (c.removeprefix("comp_"),))[0]


def cmd_demo_data(a):
    from .synthetic import make_dataset
    df = make_dataset(a.n, a.seed)
    df.to_csv(a.out, index=False)
    print(f"합성 데모 데이터 {len(df)}행 저장: {a.out}")


def cmd_train(a):
    df = pd.read_csv(a.data)
    m = CompositionPredictor().fit(df)
    m.save(a.model)
    print(f"학습 완료 ({len(df)}행) -> {a.model}")
    print(f"  물성: {m.prop_cols}\n  원료: {m.comp_cols}")


def cmd_evaluate(a):
    df = pd.read_csv(a.data).sample(frac=1, random_state=0).reset_index(drop=True)
    n_test = min(a.n_test, len(df) // 5)
    test, train = df.iloc[:n_test], df.iloc[n_test:]
    m = CompositionPredictor(n_starts=a.starts).fit(train)
    rng = np.random.default_rng(1)
    err_d, err_o = [], []
    for _, r in test.iterrows():
        props = {c: r[c] for c in m.prop_cols}
        if a.drop > 0:  # 일부 물성 누락 시나리오
            for c in rng.choice(m.prop_cols, a.drop, replace=False):
                props.pop(c)
        true = r[m.comp_cols].to_numpy(float)
        cands = m.predict(props, top_k=5)
        top1 = np.array([cands[0].composition[c] for c in m.comp_cols])
        direct = m.predict_direct(props)
        err_o.append(np.abs(top1 - true))
        err_d.append(np.abs(np.array([direct[c] for c in m.comp_cols]) - true))
    rep = pd.DataFrame({"direct_MAE(wt%)": np.mean(err_d, 0),
                        "optimized_top1_MAE(wt%)": np.mean(err_o, 0)}, index=m.comp_cols)
    rep.index = [f"{c} ({_label(c)})" for c in m.comp_cols]
    print(f"테스트 {n_test}건, 누락 물성 {a.drop}개")
    print(rep.round(2).to_string())
    print(f"평균 MAE: direct {np.mean(err_d):.2f} wt% / optimized {np.mean(err_o):.2f} wt%")


def cmd_predict(a):
    m = CompositionPredictor.load(a.model)
    if a.tds:
        pr = parse_tds(a.tds)
        props = pr.properties
        print("== TDS 추출 물성 ==")
        for k, v in props.items():
            print(f"  {PROPERTIES[k].label:38s} {v:>10g} {PROPERTIES[k].unit}")
        for w in pr.warnings:
            print(f"  [주의] {w}")
    else:
        props = json.loads(a.props)
    cands = m.predict(props, top_k=a.top_k)
    if a.json:
        print(json.dumps([c.__dict__ for c in cands], ensure_ascii=False, indent=2))
        return
    for i, c in enumerate(cands, 1):
        print(f"\n== 배합 후보 #{i}  (적합오차 {c.error:.3f}σ, 시작점: {c.source}) ==")
        for k, v in sorted(c.composition.items(), key=lambda t: -t[1]):
            if v >= 0.05:
                print(f"  {_label(k):40s} {v:6.2f} wt%")
        print("  -- 예측 물성 (입력값 대비) --")
        for k, v in c.predicted_properties.items():
            tgt = props.get(k)
            t = f"(입력 {tgt:g})" if tgt is not None else "(입력 없음)"
            print(f"  {PROPERTIES[k].label:38s} {v:>10.3g} {t}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="tds_model", description="TDS 물성 -> 원부원료 배합 예측")
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("demo-data", help="합성 데모 학습 데이터 생성")
    s.add_argument("--out", default="data/demo_training_data.csv")
    s.add_argument("--n", type=int, default=1200)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(f=cmd_demo_data)

    s = sp.add_parser("train", help="모델 학습")
    s.add_argument("--data", required=True)
    s.add_argument("--model", default="models/model.joblib")
    s.set_defaults(f=cmd_train)

    s = sp.add_parser("evaluate", help="홀드아웃 평가")
    s.add_argument("--data", required=True)
    s.add_argument("--n-test", type=int, default=100)
    s.add_argument("--drop", type=int, default=0, help="테스트 시 무작위로 누락시킬 물성 개수")
    s.add_argument("--starts", type=int, default=20)
    s.set_defaults(f=cmd_evaluate)

    s = sp.add_parser("predict", help="배합 예측")
    s.add_argument("--model", default="models/model.joblib")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--tds", help="TDS 파일 (.pdf / .txt)")
    g.add_argument("--props", help='물성 JSON, 예: \'{"mfi": 14, "density": 1.05}\'')
    s.add_argument("--top-k", type=int, default=3)
    s.add_argument("--json", action="store_true")
    s.set_defaults(f=cmd_predict)

    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    sys.exit(main())
