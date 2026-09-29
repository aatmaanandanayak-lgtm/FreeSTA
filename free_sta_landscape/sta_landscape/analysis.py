from __future__ import annotations

import math
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .models import STATE_COLS, DynamicsModel


# sensitivity
def importance(dm: DynamicsModel, n_repeats=10, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = dm.X
    y = dm.tr["post.F"].astype(float).to_numpy()
    ok = np.isfinite(y)
    base = dm.composed_F(X)
    base_err = np.mean((base[ok] - y[ok]) ** 2)
    rows = []
    ard = dm.models["d.E_res"].ard_relevance()
    for i, c in enumerate(dm.fs.cols):
        incs, spread = [], []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, i] = rng.permutation(Xp[:, i])
            fp = dm.composed_F(Xp)
            incs.append(np.mean((fp[ok] - y[ok]) ** 2) - base_err)
            spread.append(np.mean(np.abs(fp - base)))
        rows.append({"feature": c, "perm_importance": float(np.mean(incs)), "mean_abs_dF": float(np.mean(spread)),
                     "ard_relevance_Eres": float(ard[i]) if ard is not None else np.nan,
                     "min": dm.fs.lo[c], "max": dm.fs.hi[c]})
    df = pd.DataFrame(rows).sort_values("mean_abs_dF", ascending=False).reset_index(drop=True)
    return df


def interactions(dm: DynamicsModel, features: List[str], max_points=50, seed=0) -> pd.DataFrame:
    """Friedman's H^2 statistic for pairs (0 = no variance from interaction, just additive, 1 = variance from interaction)."""
    rng = np.random.default_rng(seed)
    X = dm.X 
    if len(X) > max_points:
        X = X[rng.choice(len(X), max_points, replace=False)]
    n = len(X)
    idx = {c: dm.fs.cols.index(c) for c in features if c in dm.fs.cols}

    def pd_fun(cols):
        out = np.zeros(n)
        big = np.tile(X, (n, 1))                      # row k*n + i : background i, anchor k
        anchors = np.repeat(np.arange(n), n)
        for c in cols:
            big[:, idx[c]] = X[anchors, idx[c]]
        f = dm.composed_F(big).reshape(n, n)
        out = f.mean(1)
        return out - out.mean()

    single = {c: pd_fun([c]) for c in idx}
    rows = []
    keys = list(idx)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ca, cb = keys[a], keys[b]
            pab = pd_fun([ca, cb])
            den = np.sum(pab ** 2)
            h2 = np.sum((pab - single[ca] - single[cb]) ** 2) / den if den > 1e-12 else 0.0
            rows.append({"feature_a": ca, "feature_b": cb, "H2": float(min(h2, 1.0)),
                         "joint_effect_sd": float(np.sqrt(den / n))})
    return pd.DataFrame(rows).sort_values("H2", ascending=False).reset_index(drop=True)


def local_slices(dm: DynamicsModel, row_idx: dict, features: List[str], n=25):
    """1-D cuts through the landscape at my best job of the relevant type.
    row_idx: {'classify': index, 'refine': index} into dm.tr."""
    out = {}
    for c in features:
        if c not in dm.fs.cols:
            continue
        kinds = [k for k in ("refine", "classify") if k in row_idx and k not in dm.fs.kind_fixed.get(c, {})]
        if not kinds:
            continue
        kind = kinds[0]
        x0 = dm.X[row_idx[kind]].copy()
        i = dm.fs.cols.index(c)
        lo, hi = dm.fs.range_for(c, kind)
        r = hi - lo
        grid = np.array([0.0, 1.0]) if c in dm.fs.bool_cols else np.linspace(lo - 0.25 * r, hi + 0.25 * r, n)
        Xg = np.repeat(x0[None, :], len(grid), 0)
        Xg[:, i] = grid
        df = pd.DataFrame(Xg, columns=dm.fs.cols)
        F, st = dm.predict_F(df, n_samples=400, rng=np.random.default_rng(0))
        out[f"{c} ({kind})"] = dict(grid=grid, mean=dm.composed_F(Xg), sd=F.std(0), x0=x0[i], lo=lo, hi=hi)
    return out


# candidates
def _state_vector(dm, state: Dict[str, float]):
    return np.array([state.get(c, dm.fs.fill.get(c, 0.0)) for c in STATE_COLS], float)


def generate_candidates(dm: DynamicsModel, state: Dict[str, float], kind: str, n: int,
                        rng, explore_frac=0.3, pool_filter=None) -> pd.DataFrame:
    cfg = dm.cfg
    T = max(cfg["free_energy"].get("temperature", 0.1), 1e-6)
    h = cfg["model"].get("bandwidth", 1.0)
    tr = dm.tr
    mask = (tr["kind"] == kind).to_numpy()
    if pool_filter is not None:
        mask &= pool_filter
    if mask.sum() == 0:
        return pd.DataFrame(columns=dm.fs.cols)
    pool = dm.X[mask]
    adv = (tr.loc[mask, "pre.F"].astype(float) - tr.loc[mask, "post.F"].astype(float)).fillna(-1).to_numpy()
    S = tr.loc[mask, STATE_COLS].astype(float).fillna(0).to_numpy()
    sd = tr[STATE_COLS].astype(float).std().replace(0, 1).fillna(1).to_numpy()
    s0 = _state_vector(dm, state)
    d2 = np.sum(((S - s0) / sd) ** 2, 1)
    logw = (adv - adv.max()) / T - d2 / (2 * h * h)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    acols = dm.fs.action_cols()
    aidx = [dm.fs.cols.index(c) for c in acols]
    n_exp = int(n * explore_frac)
    base = pool[rng.choice(len(pool), n - n_exp, p=w)][:, aidx] if n - n_exp > 0 else np.zeros((0, len(aidx)))
    lo = np.array([dm.fs.range_for(c, kind)[0] for c in acols])
    hi = np.array([dm.fs.range_for(c, kind)[1] for c in acols])
    rngw = np.maximum(hi - lo, 1e-9)
    fixed = np.array([kind in dm.fs.kind_fixed.get(c, {}) for c in acols])
    # local perturbations of my best moves (only along directions I varied for this job type)
    pert = base.copy()
    for j, c in enumerate(acols):
        if fixed[j]:
            continue
        if c in dm.fs.bool_cols:
            flip = rng.random(len(pert)) < 0.1
            pert[flip, j] = 1 - pert[flip, j]
        else:
            move = rng.random(len(pert)) < 0.5
            pert[move, j] += rng.normal(0, 0.2 * rngw[j], move.sum())
    # exploration: anywhere in the box I explored for this job type, extended by 25%
    ex = lo + (rng.random((n_exp, len(acols))) * 1.5 - 0.25) * rngw
    A = np.vstack([pert, ex])
    # keep exact historical moves too (top weights)
    top = pool[np.argsort(-w)[:min(10, len(pool))]][:, aidx]
    A = np.vstack([A, top])
    for j, c in enumerate(acols):
        if fixed[j]:
            A[:, j] = dm.fs.kind_fixed[c][kind]
    df = pd.DataFrame(A, columns=acols)
    _sanitise(df, dm, kind)
    for c in dm.fs.cols:
        if not c.startswith("a."):
            df[c] = state.get(c, dm.fs.fill.get(c, 0.0))
    df["kind.classify"] = float(kind == "classify")
    df["kind.refine"] = float(kind == "refine")
    return df[dm.fs.cols].drop_duplicates().reset_index(drop=True)


def _sanitise(df: pd.DataFrame, dm: DynamicsModel, kind: str):
    for c in df.columns:
        lo, hi = dm.fs.range_for(c, kind)
        r = hi - lo
        if c not in dm.fs.bool_cols and r > 0:
            df[c] = df[c].clip(lo - 0.25 * r, hi + 0.25 * r)
        if c in dm.fs.bool_cols:
            df[c] = (df[c] >= 0.5).astype(float)
            continue
        if lo >= 0:
            df[c] = df[c].clip(lower=0)
        if c in dm.fs.int_cols:
            df[c] = df[c].round()
        if c == "a.sel.frac":
            df[c] = df[c].clip(0.05, 1.0) if kind == "classify" else 1.0
        if c == "a.highres_limit_rel":
            df[c] = df[c].clip(0.2, 1.0)
        if c == "a.particle_diameter_rel":
            df[c] = df[c].clip(lower=0.8)
        if c == "a.K":
            df[c] = df[c].clip(lower=2) if kind == "classify" else 1.0
        if c == "a.sel.n_classes":
            df[c] = df[c].clip(lower=1) if kind == "classify" else 1.0
        if c.startswith("a.mod."):
            df[c] = df[c].round().clip(lower=0)
    if "a.K" in df.columns and "a.sel.n_classes" in df.columns:
        df["a.sel.n_classes"] = np.minimum(df["a.sel.n_classes"], df["a.K"])
    if "a.iter" in df.columns:
        df["a.iter"] = df["a.iter"].clip(lower=5)


def novelty(dm: DynamicsModel, X: np.ndarray) -> np.ndarray:
    """Distance to the nearest job I actually ran, in units of the typical spacing between my jobs
    (standardised coordinates). ~1 = interpolation, >2 = extrapolation."""
    sd = dm.X.std(0)
    sd[sd == 0] = 1
    Z0 = dm.X / sd
    Z = np.atleast_2d(X) / sd
    dmin = np.sqrt(((Z[:, None, :] - Z0[None, :, :]) ** 2).sum(-1)).min(1)
    if not hasattr(dm, "_typical_spacing"):
        nn = np.sqrt(((Z0[:, None, :] - Z0[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(nn, np.inf)
        dm._typical_spacing = float(np.median(nn.min(1))) if len(Z0) > 1 else 1.0
    return dmin / max(dm._typical_spacing, 1e-9)


def evaluate(dm: DynamicsModel, cands: pd.DataFrame, F_best: float, margin: float, rng=None):
    F, st = dm.predict_F(cands, rng=rng)
    m, s = F.mean(0), F.std(0)
    ei = np.mean(np.maximum(F_best - F, 0), 0)
    pi = np.mean(F < F_best - margin, 0)
    res = cands.copy()
    res["F_mean"], res["F_sd"], res["EI"], res["P_improve"] = m, s, ei, pi
    for k, v in st.items():
        res["pred." + k] = v
    res["novelty"] = novelty(dm, cands[dm.fs.cols].to_numpy(float))
    return res


# planning
def state_from_transition_post(dm: DynamicsModel, row: pd.Series) -> Dict[str, float]:
    """State *after* an observed transition, in the coordinates of the next transition."""
    s = {"pre.E_res": float(row["post.E_res"]), "pre.lnfrac": float(row["post.lnfrac"]),
         "pre.noise": float(row["post.noise"]), "pre.gold": float(row["kind"] == "refine"),
         "pre.depth": float(row["pre.depth"]) + 1,
         "pre.n_class_rounds": float(row["pre.n_class_rounds"]) + (row["kind"] == "classify"),
         "pre.n_refine_rounds": float(row["pre.n_refine_rounds"]) + (row["kind"] == "refine")}
    for c in row.index:
        if c.startswith("ctx."):
            s[c] = float(row[c])
    s["F"] = float(row["post.F"])
    return s


def plan_beam(dm: DynamicsModel, start: Dict[str, float], mode="explore", horizon=None, beam=None,
              n_cand=None, rng=None, kappa=None):
    """Beam search over sequences of Class3D / Refine3D actions. Plans must end with Refine3D
    (gold-standard resolution).  mode: explore (mean - kappa*sd), expected (mean), safe (mean + kappa*sd)."""
    cfg = dm.cfg["model"]
    horizon = horizon or cfg.get("horizon", 3)
    beam = beam or cfg.get("beam_width", 6)
    n_cand = n_cand or cfg.get("n_candidates", 250)
    kappa = cfg.get("kappa", 1.0) if kappa is None else kappa
    sign = {"explore": -1.0, "expected": 0.0, "safe": 1.0}[mode]
    rng = rng or np.random.default_rng(cfg.get("seed", 0))
    beams = [dict(state=dict(start), steps=[], var=0.0)]
    finished = []
    kinds = [k for k in ("classify", "refine") if (dm.tr["kind"] == k).any()]
    for depth in range(horizon):
        new = []
        for b in beams:
            for kind in kinds:
                cands = generate_candidates(dm, b["state"], kind, n_cand // len(kinds), rng)
                if cands.empty:
                    continue
                F, st = dm.predict_F(cands, rng=rng)
                m, s = F.mean(0), F.std(0)
                tot_sd = np.sqrt(b["var"] + s ** 2)
                score = m + sign * kappa * tot_sd
                for i in np.argsort(score)[:3]:
                    ns = dict(b["state"])
                    ns.update({"pre.E_res": float(st["E_res"][i]), "pre.lnfrac": float(st["lnfrac"][i]),
                               "pre.noise": float(st["noise"][i]), "pre.gold": float(kind == "refine"),
                               "pre.depth": b["state"].get("pre.depth", 0) + 1,
                               "pre.n_class_rounds": b["state"].get("pre.n_class_rounds", 0) + (kind == "classify"),
                               "pre.n_refine_rounds": b["state"].get("pre.n_refine_rounds", 0) + (kind == "refine")})
                    step = dict(kind=kind, action={c: float(cands.iloc[i][c]) for c in dm.fs.action_cols()},
                                novelty=float(novelty(dm, cands.iloc[[i]][dm.fs.cols].to_numpy(float))[0]),
                                F_mean=float(m[i]), F_sd=float(s[i]), hours=float(st["hours"][i]),
                                res_E=float(st["E_res"][i]), lnfrac=float(st["lnfrac"][i]), noise=float(st["noise"][i]))
                    nb = dict(state=ns, steps=b["steps"] + [step], var=float(tot_sd[i] ** 2), score=float(score[i]))
                    new.append(nb)
                    if kind == "refine":
                        finished.append(nb)
        if not new:
            break
        new.sort(key=lambda x: x["score"])
        # diversity: do not keep two beams with identical kind sequences and near-identical score
        kept, seen = [], set()
        for nb in new:
            key = (tuple(s["kind"] for s in nb["steps"]), round(nb["score"], 3))
            if key in seen:
                continue
            seen.add(key)
            kept.append(nb)
            if len(kept) >= beam:
                break
        beams = kept
    finished.sort(key=lambda x: x["score"])
    return finished[:beam]


# paths
def critical_path(tr: pd.DataFrame, outcome: str) -> List[str]:
    idx = tr.set_index("outcome")
    path, cur, seen = [], outcome, set()
    while cur is not None and cur in idx.index and cur not in seen:
        seen.add(cur)
        path.append(cur)
        p = idx.loc[cur, "pre_outcome"]
        if isinstance(p, pd.Series):
            p = p.iloc[0]
        cur = p if isinstance(p, str) else None
    return path[::-1]


# Q1: efficiency
def _replay(dm, policy, avail_pre, outcomes, y, hours, rng=None, order_user=None, n_init=3):
    """Replay the *jobs I actually ran* in the order a policy would have chosen them, respecting that
    a job only becomes available once its input state exists.  Returns (best-so-far per step, hours per step).
    policy: 'user' (my chronological order) | 'bo' (expected improvement, GP on job coordinates) | 'random'."""
    n = len(y)
    if policy == "user":
        order = list(order_user)
        return np.minimum.accumulate(y[order]), np.cumsum(hours[order])
    sd = dm.X.std(0)
    sd[sd == 0] = 1
    Z = (dm.X - dm.X.mean(0)) / sd
    roots = [i for i in range(n) if avail_pre[i] is None]
    if policy == "bo" and rng is None:
        tried = list(order_user[:n_init])
    else:
        tried = list(rng.choice(roots, min(n_init if policy == "bo" else 1, len(roots)), replace=False))
    have = {outcomes[i] for i in tried}
    while len(tried) < n:
        cand = [i for i in range(n) if i not in tried and (avail_pre[i] is None or avail_pre[i] in have)]
        if not cand:
            break
        if policy == "random":
            pick = cand[rng.integers(len(cand))]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp = GaussianProcessRegressor(ConstantKernel(1.0, (1e-2, 1e2)) * Matern(3.0, (0.3, 100.0), nu=2.5)
                                              + WhiteKernel(0.05, (1e-4, 1.0)), normalize_y=True).fit(Z[tried], y[tried])
            m, s_ = gp.predict(Z[cand], return_std=True)
            s_ = s_ + 1e-6
            best = np.min(y[tried])
            z = (best - m) / s_
            ei = (best - m) * norm.cdf(z) + s_ * norm.pdf(z)
            pick = cand[int(np.argmax(ei))]
        tried.append(pick)
        have.add(outcomes[pick])
    return np.minimum.accumulate(y[tried]), np.cumsum(hours[tried])


def _replay_stats(traj, target, f_min):
    bsf, hrs = traj
    hit = np.where(bsf <= target)[0]
    return dict(steps=float(hit[0] + 1) if len(hit) else np.nan, hours=float(hrs[hit[0]]) if len(hit) else np.nan,
                regret=float(np.mean(bsf - f_min)))


def efficiency(out: pd.DataFrame, tr: pd.DataFrame, jobs: dict, dm: DynamicsModel, imp: pd.DataFrame, cfg) -> dict:
    margin = cfg["model"].get("improvement_margin", 0.02)
    o = out.dropna(subset=["F"]).sort_values("number").reset_index(drop=True)
    best = o.loc[o["F"].idxmin()]
    path = critical_path(tr, best["outcome"])
    kinds = {"Class3D", "Refine3D"}
    all_jobs = sorted([j for j in jobs.values() if j.jtype in kinds], key=lambda j: j.number)
    o["best_so_far"] = o["F"].cummin()
    o["cum_jobs"] = [sum(1 for j in all_jobs if j.number <= n) for n in o["number"]]
    hrs = {j.number: (0 if np.isnan(j.hours) else j.hours) for j in jobs.values()}
    o["cum_hours"] = [sum(h for num, h in hrs.items() if num <= n) for n in o["number"]]
    reach = o[o["F"] <= best["F"] + margin].iloc[0]
    path_jobs = set(tr.set_index("outcome").loc[path, "job"])
    total_hours = sum(hrs.values())
    path_hours = sum(hrs.get(jobs[j].number, 0) for j in path_jobs if j in jobs)
    failed = [j.name for j in all_jobs if j.status in ("failed", "aborted")]

    # skippable steps on the critical path: apply step i+1's action directly to state i-1
    skip = []
    trx = tr.set_index("outcome")
    for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
        row_c = trx.loc[[c]].iloc[0]
        s_a = state_from_transition_post(dm, trx.loc[[a]].iloc[0])
        df = pd.DataFrame([row_c.to_dict()])
        for k, v in s_a.items():
            if k in df.columns:
                df[k] = v
        F, _ = dm.predict_F(df)
        pm, ps = float(F.mean()), float(F.std())
        skip.append({"skipped_step": b, "from": a, "then": c, "actual_F_after_then": float(row_c["post.F"]),
                     "pred_F_if_skipped": pm, "pred_sd": ps,
                     "verdict": "possibly skippable" if pm <= float(row_c["post.F"]) + margin + 0.5 * ps else "needed"})

    # flat directions: sibling jobs from the same input whose F barely differed
    flat = []
    low_imp = set(imp.loc[imp["mean_abs_dF"] < imp["mean_abs_dF"].quantile(0.5), "feature"]) if len(imp) else set()
    for (pre, kind), g in tr.dropna(subset=["post.F"]).groupby(["pre_outcome", "kind"], dropna=False):
        if len(g) < 3:
            continue
        spread = g["post.F"].max() - g["post.F"].min()
        acols = [c for c in g.columns if c.startswith("a.")]
        varied = [c for c in acols if g[c].astype(float).nunique(dropna=True) > 1]
        flat.append({"input_state": pre, "kind": kind, "n_siblings": len(g), "F_spread": float(spread),
                     "parameters_varied": ", ".join(v[2:] for v in varied),
                     "low_importance_only": bool(varied) and all(v in low_imp for v in varied),
                     "verdict": "flat (scan could have been skipped)" if spread < 2 * margin else "informative"})

    # Replays over my own jobs: my order vs Bayesian optimisation vs random
    y = tr["post.F"].astype(float).to_numpy()
    okm = np.isfinite(y)
    yy = np.where(okm, y, np.nanmax(y) + 1)
    hh = tr["hours"].astype(float).fillna(tr["hours"].median() if tr["hours"].notna().any() else 1.0).to_numpy()
    order_user = list(np.argsort(tr["number"].to_numpy()))
    f_min = float(np.nanmin(y))
    target = f_min + margin
    avail = [p if isinstance(p, str) else None for p in tr["pre_outcome"]]
    outs = list(tr["outcome"])
    rng = np.random.default_rng(cfg["model"].get("seed", 0))
    user = _replay_stats(_replay(dm, "user", avail, outs, yy, hh, order_user=order_user), target, f_min)
    bo_you = _replay_stats(_replay(dm, "bo", avail, outs, yy, hh, None, order_user), target, f_min)
    bo = [_replay_stats(_replay(dm, "bo", avail, outs, yy, hh, rng, order_user), target, f_min)
          for _ in range(cfg["model"].get("bo_replays", 20))]
    rnd = [_replay_stats(_replay(dm, "random", avail, outs, yy, hh, rng), target, f_min) for _ in range(200)]

    def agg(lst, k):
        v = np.array([d[k] for d in lst], float)
        return float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
    replay = dict(user_steps=user["steps"], user_hours=user["hours"], user_regret=user["regret"],
                  bo_from_my_start=bo_you["steps"], bo_hours_from_my_start=bo_you["hours"], bo_regret_from_my_start=bo_you["regret"],
                  bo_median=agg(bo, "steps"), bo_hours_median=agg(bo, "hours"), bo_regret_median=agg(bo, "regret"),
                  bo_iqr=(float(np.nanpercentile([d["steps"] for d in bo], 25)), float(np.nanpercentile([d["steps"] for d in bo], 75))),
                  random_median=agg(rnd, "steps"), random_hours_median=agg(rnd, "hours"), random_regret_median=agg(rnd, "regret"),
                  n_pool=int(okm.sum()))
    return dict(
        timeline=o, best=best, path=path, failed=failed,
        n_jobs_total=len(all_jobs), n_jobs_on_path=len(path_jobs & {j.name for j in all_jobs}),
        jobs_until_best=int(reach["cum_jobs"]), hours_until_best=float(reach["cum_hours"]),
        total_hours=total_hours, path_hours=path_hours, skippable=pd.DataFrame(skip), flat=pd.DataFrame(flat),
        replay=replay,
    )


# Q2: deeper minimum
def rosenthal_henderson(out: pd.DataFrame, sym_order=1):
    r = out[(out["kind"] == "refine") & (out["gold"] == 1)].dropna(subset=["res_A", "n"])
    if len(r) < 3 or r["n"].max() / max(r["n"].min(), 1) < 1.5:
        return None
    x = 1.0 / r["res_A"].to_numpy(float) ** 2
    yv = np.log(r["n"].to_numpy(float) * sym_order)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    slope = coef[0]
    if slope <= 0:
        return dict(ok=False, x=x, y=yv, note="ln N does not increase with 1/d^2 across my refinements - "
                    "particle count is not what limits resolution (quality/heterogeneity/alignment does).")
    return dict(ok=True, B=2 * slope, intercept=coef[1], x=x, y=yv)


def deeper_minimum(dm: DynamicsModel, out: pd.DataFrame, tr: pd.DataFrame, eff: dict, cfg) -> dict:
    rng = np.random.default_rng(cfg["model"].get("seed", 0) + 1)
    margin = cfg["model"].get("improvement_margin", 0.02)
    F_best = float(eff["best"]["F"])
    trx = tr.set_index("outcome")
    starts = {}
    for oc in eff["path"]:
        starts[oc] = state_from_transition_post(dm, trx.loc[[oc]].iloc[0])
    # also the root state of the first job on the path
    first = trx.loc[[eff["path"][0]]].iloc[0]
    starts["<start>"] = {c: float(first[c]) for c in first.index if c.startswith(("pre.", "ctx."))}
    one_step = []
    for sname, st in starts.items():
        for kind in ("classify", "refine"):
            c = generate_candidates(dm, st, kind, cfg["model"].get("n_candidates", 250), rng)
            if c.empty:
                continue
            ev = evaluate(dm, c, F_best, margin, rng)
            ev["from_state"], ev["kind"] = sname, kind
            one_step.append(ev)
    one = pd.concat(one_step, ignore_index=True) if one_step else pd.DataFrame()
    # only refine end-points are comparable to a gold-standard best
    top = one[one["kind"] == "refine"].sort_values("EI", ascending=False).head(15) if len(one) else one
    plans = []
    for sname, st in starts.items():
        for mode in ("explore", "expected"):
            for p in plan_beam(dm, st, mode=mode, rng=rng)[:2]:
                fin = p["steps"][-1]
                z = (F_best - margin - fin["F_mean"]) / max(math.sqrt(p["var"]), 1e-6)
                plans.append(dict(from_state=sname, mode=mode, steps=p["steps"], F_final=fin["F_mean"],
                                  max_novelty=max(st["novelty"] for st in p["steps"]),
                                  F_sd=math.sqrt(p["var"]), P_improve=float(norm.cdf(z))))
    plans.sort(key=lambda p: (-p["P_improve"], p["F_final"]))
    unexplored = [dict(coordinate=c[2:], fixed_value=v, note="never varied") for c, v in dm.fs.const_values.items() if c.startswith("a.")]
    unexplored += [dict(coordinate=c[2:], fixed_value="; ".join(f"{k}: {v:g}" for k, v in per.items()), note="only ever changed together with the job type")
                   for c, per in dm.fs.kind_const.items()]
    best = eff["best"]
    apix = float(best["apix"]) if best["apix"] == best["apix"] else None
    nyq = 2 * apix if apix else None
    nyq_note = None
    if nyq and float(best["res_A"]) < 1.3 * nyq:
        nyq_note = (f"Best resolution {best['res_A']:.1f} A is within 30% of Nyquist ({nyq:.1f} A at {apix:.2f} A/px): "
                    "the deeper minimum is probably behind a smaller pixel size (less binning) and a re-extraction; "
                    "that move is outside what the model has seen and needs to be tested explicitly.")
    rh = rosenthal_henderson(out)
    p_deeper = max([float(top["P_improve"].max()) if len(top) else 0.0] + [p["P_improve"] for p in plans])
    return dict(one_step=top, plans=plans[:8], unexplored=pd.DataFrame(unexplored), nyquist_note=nyq_note,
                rh=rh, p_deeper=p_deeper, F_best=F_best)
