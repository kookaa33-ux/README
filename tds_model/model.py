"""TDS 물성 -> 원부원료 배합비 역예측 모델.

역문제(물성 -> 배합)는 해가 여러 개일 수 있으므로 두 단계로 푼다.

1. **Direct inverse model** : 물성 -> 배합비를 직접 회귀
   (HistGradientBoosting, 결측 물성 지원 / 학습 시 무작위 마스킹 증강).
2. **Forward model + 최적화** : 배합 -> 물성 순방향 대리모델(Gaussian Process)을
   학습한 뒤, 입력된 물성과의 정규화 오차가 최소가 되는 배합을 단체(simplex,
   합계 100 wt%) 위에서 다중 시작점 SLSQP 로 탐색한다. Direct 예측을 시작점에
   포함하고, 서로 다른 후보 배합들을 Top-K 로 반환한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor

from .schema import COMP_PREFIX, PROPERTIES, encode_value


def project_simplex(x: np.ndarray, total: float = 100.0) -> np.ndarray:
    """벡터를 {x>=0, sum=total} 로 유클리드 사영."""
    x = np.asarray(x, float)
    u = np.sort(x)[::-1]
    css = np.cumsum(u) - total
    idx = np.arange(1, len(x) + 1)
    rho = np.nonzero(u - css / idx > 0)[0][-1]
    theta = css[rho] / (rho + 1)
    return np.maximum(x - theta, 0.0)


@dataclass
class Candidate:
    composition: dict[str, float]
    predicted_properties: dict[str, float]
    error: float          # 입력 물성 대비 정규화 RMSE (σ 단위)
    source: str


class CompositionPredictor:
    def __init__(self, n_starts: int = 30, mask_prob: float = 0.3, random_state: int = 0):
        self.n_starts = n_starts
        self.mask_prob = mask_prob
        self.random_state = random_state

    # ------------------------------------------------------------------ utils
    def _tf(self, props: pd.DataFrame) -> np.ndarray:
        """물성 -> 모델 공간 (로그변환 + 표준화)."""
        a = props[self.prop_cols].to_numpy(float).copy()
        for j, c in enumerate(self.prop_cols):
            if self._log[j]:
                a[:, j] = np.log(np.clip(a[:, j], 1e-6, None))
        return (a - self._p_mu) / self._p_sd

    def _inv_tf(self, z: np.ndarray) -> np.ndarray:
        a = z * self._p_sd + self._p_mu
        for j in range(len(self.prop_cols)):
            if self._log[j]:
                a[..., j] = np.exp(a[..., j])
        return a

    # ------------------------------------------------------------------- fit
    def fit(self, df: pd.DataFrame) -> "CompositionPredictor":
        self.comp_cols = [c for c in df.columns if c.startswith(COMP_PREFIX)]
        self.prop_cols = [c for c in PROPERTIES if c in df.columns]
        if not self.comp_cols or not self.prop_cols:
            raise ValueError("학습 데이터에 물성 컬럼과 comp_* 배합 컬럼이 모두 필요합니다.")
        df = df.dropna(subset=self.comp_cols).reset_index(drop=True)
        self._log = np.array([PROPERTIES[c].log_scale for c in self.prop_cols])

        raw = df[self.prop_cols].to_numpy(float).copy()
        for j in range(raw.shape[1]):
            if self._log[j]:
                raw[:, j] = np.log(np.clip(raw[:, j], 1e-6, None))
        self._p_mu = np.nanmean(raw, 0)
        self._p_sd = np.nanstd(raw, 0) + 1e-9

        Z = self._tf(df)
        C = df[self.comp_cols].to_numpy(float)
        C = np.array([project_simplex(r) for r in C])
        self.comp_bounds_ = np.c_[C.min(0), C.max(0)]
        self.train_comp_ = C

        # 1) direct inverse (결측 마스킹 증강)
        rng = np.random.default_rng(self.random_state)
        Zs, Cs = [Z], [C]
        for _ in range(2):
            Zm = Z.copy()
            mask = rng.random(Zm.shape) < self.mask_prob
            mask[mask.all(1), 0] = False
            Zm[mask] = np.nan
            Zs.append(Zm)
            Cs.append(C)
        self.inverse_ = MultiOutputRegressor(HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, random_state=self.random_state))
        self.inverse_.fit(np.vstack(Zs), np.vstack(Cs))

        # 2) forward surrogate (GP, 결측 없는 행만)
        ok = ~np.isnan(Z).any(1)
        Xf, Yf = C[ok] / 100.0, Z[ok]
        if len(Xf) > 1500:  # GP 계산량 제한
            sel = rng.choice(len(Xf), 1500, replace=False)
            Xf, Yf = Xf[sel], Yf[sel]
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.full(C.shape[1], 0.3), nu=2.5) \
            + WhiteKernel(0.01)
        self.forward_ = GaussianProcessRegressor(kernel, normalize_y=True,
                                                 random_state=self.random_state)
        self.forward_.fit(Xf, Yf)
        return self

    # --------------------------------------------------------------- predict
    def predict_properties(self, comp: pd.DataFrame | dict) -> pd.DataFrame:
        """배합 -> 물성 (순방향 검증용)."""
        if isinstance(comp, dict):
            comp = pd.DataFrame([comp])
        X = comp.reindex(columns=self.comp_cols, fill_value=0.0).to_numpy(float) / 100.0
        return pd.DataFrame(self._inv_tf(self.forward_.predict(X)), columns=self.prop_cols)

    def _props_row(self, props: dict) -> pd.DataFrame:
        return pd.DataFrame([{c: encode_value(c, props[c]) if props.get(c) is not None else np.nan
                              for c in self.prop_cols}], dtype=float)

    def predict_direct(self, props: dict[str, float]) -> dict[str, float]:
        """Direct inverse model 만 사용한 빠른 예측 (최적화 없음)."""
        row = self._props_row(props)
        x = project_simplex(self.inverse_.predict(self._tf(row))[0])
        return {c: round(float(v), 2) for c, v in zip(self.comp_cols, x)}

    def predict(self, props: dict[str, float], top_k: int = 3,
                weights: dict[str, float] | None = None) -> list[Candidate]:
        """TDS 물성(dict, 일부 누락 가능) -> 배합 후보 Top-K."""
        row = self._props_row(props)
        z = self._tf(row)[0]
        known = ~np.isnan(z)
        if not known.any():
            raise ValueError(f"인식 가능한 물성이 없습니다. 지원 항목: {self.prop_cols}")
        w = np.array([(weights or {}).get(c, 1.0) for c in self.prop_cols])[known]
        zk = z[known]

        def loss(x):
            pred = self.forward_.predict(x[None, :] / 100.0)[0][known]
            return float(np.sum(w * (pred - zk) ** 2) / w.sum())

        direct = project_simplex(self.inverse_.predict(z[None, :])[0])
        # 시작점: direct 예측 + 물성 공간에서 가장 가까운 학습 배합들 + 랜덤
        tz = self.forward_.predict(self.train_comp_ / 100.0)[:, known]
        near = np.argsort(((tz - zk) ** 2).sum(1))[: self.n_starts // 2]
        rng = np.random.default_rng(self.random_state)
        rand = rng.choice(len(self.train_comp_), self.n_starts - len(near) - 1, replace=False)
        starts = [("direct", direct)] + [("nearest", self.train_comp_[i]) for i in near] \
            + [("random", self.train_comp_[i]) for i in rand]

        lo, hi = self.comp_bounds_[:, 0], self.comp_bounds_[:, 1]
        cons = [{"type": "eq", "fun": lambda x: x.sum() - 100.0}]
        results = []
        for src, x0 in starts:
            x0 = np.clip(x0, lo, hi)
            r = minimize(loss, x0, method="SLSQP", bounds=list(zip(lo, hi)),
                         constraints=cons, options={"maxiter": 200, "ftol": 1e-8})
            x = project_simplex(np.clip(r.x, lo, hi))
            results.append((loss(x), src, x))
        results.append((loss(direct), "direct-raw", direct))
        results.sort(key=lambda t: t[0])

        # 서로 충분히 다른 후보만 선택 (wt% 유클리드 거리 > 3)
        picked: list[tuple] = []
        for res in results:
            if all(np.linalg.norm(res[2] - p[2]) > 3.0 for p in picked):
                picked.append(res)
            if len(picked) == top_k:
                break

        out = []
        for err, src, x in picked:
            pp = self._inv_tf(self.forward_.predict(x[None, :] / 100.0))[0]
            out.append(Candidate(
                composition={c: round(float(v), 2) for c, v in zip(self.comp_cols, x)},
                predicted_properties={c: round(float(v), 3) for c, v in zip(self.prop_cols, pp)},
                error=round(float(np.sqrt(err)), 3), source=src))
        return out

    # ------------------------------------------------------------- persist
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "CompositionPredictor":
        return joblib.load(path)
