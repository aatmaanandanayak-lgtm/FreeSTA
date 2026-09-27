"""Surrogate models of the landscape.

DynamicsModel learns   state_after - state_before = g(state_before, job type, parameters, particle)
for three state variables (E_res, ln particle fraction, noise) plus the compute cost (log hours).
Each target uses an average of a Gaussian process (ARD Matern kernel: smooth, calibrated
uncertainty, per-parameter length-scales) and an extremely-randomised tree ensemble (robust to
discontinuities and interactions).  Uncertainty = mixture variance of the two.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold

from .landscape import F_from_state

TARGETS = ["d.E_res", "d.lnfrac", "d.noise", "log_hours"]
STATE_COLS = ["pre.E_res", "pre.lnfrac", "pre.noise", "pre.gold", "pre.depth",
              "pre.n_class_rounds", "pre.n_refine_rounds"]


class Surrogate:
    def __init__(self, seed=0, use_gp=True):
        self.seed = seed
        self.use_gp = use_gp

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1.0
        Z = (X - self.mu) / self.sd
        n, d = Z.shape
        self.et = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2 if n > 10 else 1,
                                      random_state=self.seed, n_jobs=-1).fit(Z, y)
        self.gp = None
        self.w_gp = 0.0
        if self.use_gp and n >= 6:
            k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(d) * 2.0,
                                                          length_scale_bounds=(5e-2, 1e3), nu=2.5) \
                + WhiteKernel(0.1, (1e-6, 1e1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)
                self.gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2,
                                                   random_state=self.seed).fit(Z, y)
            self.w_gp = 0.5
        self.y_sd = float(np.std(y)) if len(y) > 1 else 1.0
        return self

    def predict(self, X: np.ndarray):
        Z = (np.atleast_2d(X) - self.mu) / self.sd
        allt = np.stack([t.predict(Z) for t in self.et.estimators_])
        m_et, s_et = allt.mean(0), allt.std(0)
        if self.gp is None:
            return m_et, np.maximum(s_et, 1e-6)
        m_gp, s_gp = self.gp.predict(Z, return_std=True)
        w = self.w_gp
        m = w * m_gp + (1 - w) * m_et
        v = w * s_gp ** 2 + (1 - w) * s_et ** 2 + w * (1 - w) * (m_gp - m_et) ** 2
        return m, np.sqrt(np.maximum(v, 1e-12))

    def ard_relevance(self) -> Optional[np.ndarray]:
        """1/length-scale per (standardised) input from the GP, if fitted."""
        if self.gp is None:
            return None
        try:
            ls = self.gp.kernel_.k1.k2.length_scale
            return 1.0 / np.asarray(ls, float)
        except AttributeError:
            return None


class FeatureSpace:
    """Keeps the column list, fill values, types and ranges of the landscape coordinates."""

    def __init__(self, tr: pd.DataFrame, use_ctx: bool, switches=()):
        acols = [c for c in tr.columns if c.startswith("a.")]
        ctx = [c for c in tr.columns if c.startswith("ctx.")] if use_ctx else []
        self.cols = STATE_COLS + ["kind.classify", "kind.refine"] + ctx + acols
        self.fill: Dict[str, float] = {}
        self.bool_cols, self.int_cols = set(), set()
        self.lo, self.hi = {}, {}
        switches = set(switches)
        for c in self.cols:
            s = pd.to_numeric(tr[c], errors="coerce") if c in tr.columns else pd.Series([np.nan] * len(tr))
            if c.startswith("a.mod.") or (c.startswith("a.") and c[2:] in switches):
                s = s.fillna(0.0)          # an absent switch / intermediate job means 'off'
                self.fill[c] = 0.0
                if c[2:] in switches:
                    self.bool_cols.add(c)
            vals = s.dropna()
            if c not in self.fill:
                self.fill[c] = float(vals.median()) if len(vals) else 0.0
            if len(vals) and np.allclose(vals, np.round(vals)) and c not in self.bool_cols and c.startswith("a."):
                self.int_cols.add(c)
            self.lo[c] = float(vals.min()) if len(vals) else 0.0
            self.hi[c] = float(vals.max()) if len(vals) else 0.0
        # drop coordinates that never varied: they cannot be learned (reported as 'unexplored')
        self.constant = [c for c in self.cols if c.startswith(("a.", "ctx.")) and self.lo[c] == self.hi[c]]
        self.const_values = {c: self.lo[c] for c in self.constant}
        # coordinates fully determined by the job type (e.g. --firstiter_cc only in Class3D): confounded
        # with the job type, so they are not separate landscape directions
        self.kind_const: Dict[str, Dict[str, float]] = {}
        for c in self.cols:
            if not c.startswith("a.") or c in self.constant or c not in tr.columns:
                continue
            s = pd.to_numeric(tr[c], errors="coerce")
            if c in self.bool_cols or c.startswith("a.mod."):
                s = s.fillna(0.0)
            per = {}
            for k, g in s.groupby(tr["kind"]):
                g = g.dropna()
                if len(g) == 0 or g.nunique() > 1:
                    per = None
                    break
                per[k] = float(g.iloc[0])
            if per:
                self.kind_const[c] = per
        drop = set(self.constant) | set(self.kind_const)
        self.cols = [c for c in self.cols if c not in drop]
        # per job type: which coordinates were ever varied / exist, and over what range
        self.kind_fixed: Dict[str, Dict[str, float]] = {}
        self.kind_absent: Dict[str, set] = {}
        self.kind_lo: Dict[str, Dict[str, float]] = {}
        self.kind_hi: Dict[str, Dict[str, float]] = {}
        for c in self.action_cols():
            s = pd.to_numeric(tr[c], errors="coerce") if c in tr.columns else pd.Series([np.nan] * len(tr))
            if c in self.bool_cols or c.startswith("a.mod."):
                s = s.fillna(0.0)
            for k, g in s.groupby(tr["kind"]):
                g = g.dropna()
                if len(g) == 0:
                    self.kind_absent.setdefault(c, set()).add(k)
                    self.kind_fixed.setdefault(c, {})[k] = self.fill[c]
                    continue
                self.kind_lo.setdefault(c, {})[k] = float(g.min())
                self.kind_hi.setdefault(c, {})[k] = float(g.max())
                if g.nunique() == 1:
                    self.kind_fixed.setdefault(c, {})[k] = float(g.iloc[0])

    def range_for(self, c, kind):
        return (self.kind_lo.get(c, {}).get(kind, self.lo[c]), self.kind_hi.get(c, {}).get(kind, self.hi[c]))

    def matrix(self, df: pd.DataFrame) -> np.ndarray:
        X = np.zeros((len(df), len(self.cols)))
        for i, c in enumerate(self.cols):
            if c in df.columns:
                v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
                X[:, i] = np.where(np.isnan(v), self.fill[c], v)
            else:
                X[:, i] = self.fill[c]
        return X

    def action_cols(self):
        return [c for c in self.cols if c.startswith("a.")]

    def full_action(self, action: Dict[str, float], kind: str) -> Dict[str, float]:
        """Add back the coordinates that were constant / fixed by job type."""
        full = {c: v for c, v in self.const_values.items() if c.startswith("a.")}
        for c, per in self.kind_const.items():
            if kind in per:
                full[c] = per[kind]
        full.update(action)
        for c, kinds in self.kind_absent.items():  # e.g. --auto_local_healpix_order never exists for Class3D
            if kind in kinds:
                full.pop(c, None)
        return full


class DynamicsModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.seed = cfg["model"].get("seed", 0)

    def fit(self, tr: pd.DataFrame, switches=()):
        tr = tr.copy()
        tr = tr[np.isfinite(tr["d.E_res"].astype(float))].reset_index(drop=True)
        tr["log_hours"] = np.log1p(tr["hours"].astype(float).fillna(tr["hours"].median() if tr["hours"].notna().any() else 1.0))
        use_ctx = tr["particle"].nunique() > 1
        self.fs = FeatureSpace(tr, use_ctx, switches)
        for c in self.fs.cols:  # store switches / modifiers as explicit 0 so tables read naturally
            if c in tr.columns and self.fs.fill.get(c) == 0.0 and (c in self.fs.bool_cols or c.startswith("a.mod.")):
                tr[c] = tr[c].fillna(0.0)
        self.tr = tr
        # guard-rails against wild extrapolation: predicted states may go at most `extrapolation` x
        # (observed range) beyond anything I have measured
        ext = self.cfg["model"].get("extrapolation", 0.25)
        self.bounds = {}
        for v in ("E_res", "noise"):
            allv = pd.concat([tr["post." + v], tr["pre." + v]]).astype(float).dropna()
            post = tr["post." + v].astype(float).dropna()
            r = float(allv.max() - allv.min()) or 1.0
            self.bounds[v] = (float(post.min()) - ext * r, float(allv.max()) + ext * r)
        X = self.fs.matrix(tr)
        self.X = X
        self.models = {}
        for t in TARGETS:
            y = tr[t].astype(float).fillna(0.0).to_numpy()
            self.models[t] = Surrogate(self.seed).fit(X, y)
        return self

    def predict(self, df: pd.DataFrame):
        X = self.fs.matrix(df)
        return {t: self.models[t].predict(X) for t in TARGETS}

    def predict_F(self, df: pd.DataFrame, n_samples=64, rng=None):
        """Monte-Carlo distribution of F after the action (plus predicted means)."""
        rng = rng or np.random.default_rng(self.seed)
        p = self.predict(df)
        pre = {k: pd.to_numeric(df[k], errors="coerce").fillna(self.fs.fill.get(k, 0)).to_numpy(float)
               for k in ("pre.E_res", "pre.lnfrac", "pre.noise")}
        n = len(df)
        be, bz = self.bounds["E_res"], self.bounds["noise"]
        eps = rng.standard_normal((3, n_samples, n))
        e = np.clip(pre["pre.E_res"] + p["d.E_res"][0] + eps[0] * p["d.E_res"][1], *be)
        l = np.minimum(pre["pre.lnfrac"] + p["d.lnfrac"][0] + eps[1] * p["d.lnfrac"][1], 0.0)
        z = np.clip(pre["pre.noise"] + p["d.noise"][0] + eps[2] * p["d.noise"][1], max(0.0, bz[0]), min(1.0, bz[1]))
        F = F_from_state(e, l, z, self.cfg)
        mean_state = dict(E_res=np.clip(pre["pre.E_res"] + p["d.E_res"][0], *be),
                          lnfrac=np.minimum(pre["pre.lnfrac"] + p["d.lnfrac"][0], 0),
                          noise=np.clip(pre["pre.noise"] + p["d.noise"][0], max(0.0, bz[0]), min(1.0, bz[1])),
                          hours=np.expm1(p["log_hours"][0]),
                          sd_E_res=p["d.E_res"][1], sd_lnfrac=p["d.lnfrac"][1], sd_noise=p["d.noise"][1])
        return F, mean_state

    def composed_F(self, X: np.ndarray) -> np.ndarray:
        """Mean predicted F for raw feature matrix rows (used for sensitivity analysis)."""
        cols = self.fs.cols
        idx = {c: i for i, c in enumerate(cols)}

        def col(name):
            return X[:, idx[name]] if name in idx else np.full(len(X), self.fs.fill.get(name, self.fs.const_values.get(name, 0.0)))
        be, bz = self.bounds["E_res"], self.bounds["noise"]
        e = np.clip(col("pre.E_res") + self.models["d.E_res"].predict(X)[0], *be)
        l = np.minimum(col("pre.lnfrac") + self.models["d.lnfrac"].predict(X)[0], 0)
        z = np.clip(col("pre.noise") + self.models["d.noise"].predict(X)[0], max(0.0, bz[0]), min(1.0, bz[1]))
        return F_from_state(e, l, z, self.cfg)

    def cv_scores(self, k=5) -> pd.DataFrame:
        """Cross-validated predictive skill for each target (honesty check)."""
        n = len(self.X)
        rows = []
        if n < 8:
            return pd.DataFrame([{"target": t, "R2_cv": np.nan, "RMSE_cv": np.nan, "n": n} for t in TARGETS])
        kf = KFold(n_splits=min(k, n), shuffle=True, random_state=self.seed)
        for t in TARGETS:
            y = self.tr[t].astype(float).fillna(0.0).to_numpy()
            pred = np.zeros(n)
            for a, b in kf.split(self.X):
                s = Surrogate(self.seed, use_gp=True).fit(self.X[a], y[a])
                pred[b] = s.predict(self.X[b])[0]
            ss = np.sum((y - y.mean()) ** 2)
            rows.append({"target": t, "R2_cv": 1 - np.sum((y - pred) ** 2) / ss if ss > 0 else np.nan,
                         "RMSE_cv": float(np.sqrt(np.mean((y - pred) ** 2))), "n": n})
        return pd.DataFrame(rows)
