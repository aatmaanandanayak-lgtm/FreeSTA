from __future__ import annotations

import base64
import html
import io
import os
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from .codec import decode_features  # noqa: E402

KIND_COL = {"classify": "#2a78c7", "refine": "#d1495b"}
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


FIG_DIR = None  # when set, every figure is also written here as a PNG


def _png(fig, name=None):
    if FIG_DIR and name:
        os.makedirs(FIG_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIG_DIR, name + ".png"), bbox_inches="tight", dpi=150)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img(b64, alt=""):
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{b64}"/>'


def _tbl(df, n=None, floatfmt="{:.3g}"):
    if df is None or len(df) == 0:
        return "<p class='muted'>(none)</p>"
    d = df.head(n) if n else df
    return d.to_html(index=False, escape=True, float_format=lambda x: floatfmt.format(x), border=0, classes="t")


# flag descriptions
def action_to_flags(action: dict, dm, apix, diam):
    feat = {c[2:] if c.startswith("a.") else c: v for c, v in action.items()}
    return decode_features(feat, apix, diam, bool_cols={c[2:] for c in dm.fs.bool_cols},
                           int_cols={c[2:] for c in dm.fs.int_cols} | {"K", "iter"})


def describe_vs_nearest(action: dict, kind: str, dm, apix, diam, jobs):
    """Describe a proposed action as a diff against my most similar real job of the same type."""
    acols = dm.fs.action_cols()
    tr = dm.tr
    m = (tr["kind"] == kind).to_numpy()
    if not m.any():
        return "", ""
    idx = [dm.fs.cols.index(c) for c in acols]
    sd = dm.X[:, idx].std(0)
    sd[sd == 0] = 1
    v = np.array([action.get(c, dm.fs.fill[c]) for c in acols])
    Xk = dm.X[m][:, idx]
    j = int(np.argmin((((Xk - v) / sd) ** 2).sum(1)))
    row = tr[m].iloc[j]
    ref = {c: float(row[c]) if row[c] == row[c] else dm.fs.fill[c] for c in acols}
    new_f = action_to_flags(dm.fs.full_action(action, kind), dm, apix, diam)
    old_f = action_to_flags(dm.fs.full_action(ref, kind), dm, apix, diam)
    diffs = []
    for k in sorted(set(new_f) | set(old_f)):
        a, b = old_f.get(k, "off"), new_f.get(k, "off")
        if str(a) != str(b):
            diffs.append(f"--{k}: {a} → {b}")
    for c in acols:
        if c.startswith(("a.sel.", "a.mod.", "a.mask.")) and abs(ref[c] - action.get(c, ref[c])) > 1e-3 and kind not in dm.fs.kind_fixed.get(c, {}):
            diffs.append(f"{c[2:]}: {ref[c]:.3g} → {action.get(c):.3g}")
    return row["job"], "; ".join(diffs) if diffs else "(same parameters)"


# figures
def fig_tree(out, tr, path):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    o = out.dropna(subset=["F"]).set_index("outcome")
    for r in tr.itertuples():
        if isinstance(r.pre_outcome, str) and r.pre_outcome in o.index and r.outcome in o.index:
            a, b = o.loc[r.pre_outcome], o.loc[r.outcome]
            on = r.outcome in path and r.pre_outcome in path
            ax.plot([a["number"], b["number"]], [a["F"], b["F"]], color="#333" if on else "#bbb",
                    lw=2 if on else 0.8, zorder=1)
    for kind, g in o.groupby("kind"):
        ax.scatter(g["number"], g["F"], s=28, c=KIND_COL.get(kind, "gray"), label=kind, zorder=2, edgecolor="white", lw=0.5)
    b = o["F"].idxmin()
    ax.scatter([o.loc[b, "number"]], [o.loc[b, "F"]], s=160, facecolor="none", edgecolor="black", lw=1.5, zorder=3)
    ax.set_xlabel("job number (≈ time)")
    ax.set_ylabel("free energy F (lower = better)")
    ax.set_title("My exploration of the landscape (dark line = path to my minimum)")
    ax.legend(frameon=False)
    return _png(fig, "01_exploration_tree")


def fig_pca(dm):
    y = dm.tr["post.F"].astype(float).to_numpy()
    sd = dm.X.std(0)
    sd[sd == 0] = 1
    mu = dm.X.mean(0)
    Z = (dm.X - mu) / sd
    if Z.shape[1] < 2 or len(Z) < 4:
        return None
    pca = PCA(2).fit(Z)
    P = pca.transform(Z)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    gx = np.linspace(P[:, 0].min() - 1, P[:, 0].max() + 1, 45)
    gy = np.linspace(P[:, 1].min() - 1, P[:, 1].max() + 1, 45)
    GX, GY = np.meshgrid(gx, gy)
    G = pca.inverse_transform(np.c_[GX.ravel(), GY.ravel()]) * sd + mu
    FG = dm.composed_F(G).reshape(GX.shape)
    cs = ax.contourf(GX, GY, FG, levels=18, cmap="viridis_r", alpha=0.85)
    fig.colorbar(cs, ax=ax, label="surrogate F (PCA plane)")
    ax.scatter(P[:, 0], P[:, 1], c=y, cmap="viridis_r", edgecolor="white", s=36, vmin=np.nanmin(FG), vmax=np.nanmax(FG))
    for kind, mk in (("classify", "o"), ("refine", "s")):
        m = (dm.tr["kind"] == kind).to_numpy()
        ax.scatter(P[m, 0], P[m, 1], facecolor="none", edgecolor=KIND_COL[kind], marker=mk, s=70, lw=1, label=kind)
    ax.legend(frameon=False, loc="best")
    load = pd.Series(pca.components_[0], index=dm.fs.cols).abs().sort_values(ascending=False).head(3)
    load2 = pd.Series(pca.components_[1], index=dm.fs.cols).abs().sort_values(ascending=False).head(3)
    ax.set_xlabel("PC1: " + ", ".join(load.index))
    ax.set_ylabel("PC2: " + ", ".join(load2.index))
    ax.set_title("Landscape projected on its 2 main axes (points = my jobs)")
    return _png(fig, "02_landscape_pca")


def fig_importance(imp):
    d = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(d) + 1))
    ax.barh(d["feature"], d["mean_abs_dF"], color="#4c72b0")
    ax.set_xlabel("mean |ΔF| when this coordinate is scrambled")
    ax.set_title("Which parameters move the free energy")
    return _png(fig, "03_parameter_importance")


def fig_interactions(inter, feats):
    if inter is None or inter.empty:
        return None
    f = [x for x in feats if x in set(inter["feature_a"]) | set(inter["feature_b"])]
    M = pd.DataFrame(0.0, index=f, columns=f)
    for r in inter.itertuples():
        if r.feature_a in f and r.feature_b in f:
            M.loc[r.feature_a, r.feature_b] = M.loc[r.feature_b, r.feature_a] = r.H2
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(M.to_numpy(), cmap="magma_r", vmin=0, vmax=max(0.2, M.to_numpy().max()))
    ax.set_xticks(range(len(f)))
    ax.set_xticklabels(f, rotation=60, ha="right")
    ax.set_yticks(range(len(f)))
    ax.set_yticklabels(f)
    fig.colorbar(im, ax=ax, label="Friedman H² (interaction strength)")
    ax.set_title("Parameters acting in concert")
    return _png(fig, "04_parameter_interactions")


def fig_slices(sl):
    if not sl:
        return None
    k = len(sl)
    cols = 3
    rows = math.ceil(k / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(9, 2.4 * rows), squeeze=False)
    for ax, (c, d) in zip(axs.ravel(), sl.items()):
        ax.plot(d["grid"], d["mean"], color="#333")
        ax.fill_between(d["grid"], d["mean"] - d["sd"], d["mean"] + d["sd"], alpha=0.25, color="#4c72b0")
        ax.axvspan(d["lo"], d["hi"], color="#eee", zorder=0)
        ax.axvline(d["x0"], color="#d1495b", lw=1)
        ax.set_title(c, fontsize=8)
    for ax in axs.ravel()[k:]:
        ax.axis("off")
    fig.suptitle("Landscape slices through my best job (red); grey band = range I explored", fontsize=9)
    fig.tight_layout()
    return _png(fig, "05_landscape_slices")


def fig_progress(eff):
    t = eff["timeline"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    axs[0].plot(t["cum_jobs"], t["F"], "o", color="#999", ms=3)
    axs[0].step(t["cum_jobs"], t["best_so_far"], where="post", color="#d1495b")
    rp = eff["replay"]
    if rp["bo_median"] == rp["bo_median"] and rp["user_steps"] == rp["user_steps"]:
        axs[0].axvline(rp["bo_median"], ls="--", color="#2a78c7", label=f"BO replay median ({rp['bo_median']:.0f} measured jobs)")
        axs[0].axvline(rp["user_steps"], ls=":", color="#d1495b", label=f"my order ({rp['user_steps']:.0f} measured jobs)")
        axs[0].legend(frameon=False, fontsize=7)
    axs[0].set_xlabel("Class3D + Refine3D jobs run")
    axs[0].set_ylabel("F")
    axs[1].step(t["cum_hours"], t["best_so_far"], where="post", color="#d1495b")
    axs[1].set_xlabel("cumulative wall-clock hours")
    axs[1].set_ylabel("best F so far")
    fig.suptitle("Rate of descent", fontsize=9)
    fig.tight_layout()
    return _png(fig, "06_rate_of_descent")


def fig_rh(rh):
    if not rh:
        return None
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(rh["x"], rh["y"], "o")
    if rh.get("ok"):
        xx = np.linspace(min(rh["x"]) * 0.9, max(rh["x"]) * 1.1, 20)
        ax.plot(xx, (rh["B"] / 2) * xx + rh["intercept"], "-", color="#d1495b", label=f"B ≈ {rh['B']:.0f} Å²")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "no positive slope:\nparticle number is not\nthe limiting factor", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="#666")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.set_xlabel("1/d² (Å⁻²)")
    ax.set_ylabel("ln N")
    ax.set_title("Rosenthal–Henderson (my refinements)")
    return _png(fig, "07_rosenthal_henderson")


# HTML
CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#666;--line:#e3e3e3;--acc:#2a78c7;--card:#f7f7f8}
@media (prefers-color-scheme: dark){:root{--bg:#161618;--fg:#e8e8ea;--muted:#9a9aa0;--line:#333;--card:#1f1f22}
img{background:#fff;border-radius:6px}}
body{background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1050px;margin:0 auto;padding:16px}
h1{font-size:24px;margin:.2em 0}h2{margin-top:2em;border-bottom:1px solid var(--line);padding-bottom:4px}
h3{margin-top:1.4em}.muted{color:var(--muted)}img{max-width:100%}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:10px 0}
.kpi{display:inline-block;margin:0 24px 8px 0}.kpi b{display:block;font-size:20px}
table.t{border-collapse:collapse;font-size:12px;margin:8px 0;display:block;overflow-x:auto}
table.t th,table.t td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;white-space:nowrap}
code,pre{font-family:ui-monospace,Menlo,monospace;font-size:12px}pre{white-space:pre-wrap;background:var(--card);padding:8px;border-radius:6px;border:1px solid var(--line)}
"""


def build_report(path, cfg, out, dm, cv, imp, inter, slices, eff, deep, recipe, meta, jobs, transfer_res=None, fig_dir=None):
    global FIG_DIR
    FIG_DIR = fig_dir
    apix = float(eff["best"]["apix"]) if eff["best"]["apix"] == eff["best"]["apix"] else 1.0
    diam = meta["diameter"]
    fe = cfg["free_energy"]
    best = eff["best"]
    rp = eff["replay"]
    H = []
    H.append(f"<h1>Subtomogram-averaging free-energy landscape</h1><p class='muted'>{len(out)} measured states from "
             f"{eff['n_jobs_total']} Class3D/Refine3D jobs · particle: {html.escape(str(meta['particle'].get('name', '')))}</p>")
    H.append("<div class='card'>"
             f"<span class='kpi'>best state<b>{html.escape(str(best['outcome']))}</b></span>"
             f"<span class='kpi'>resolution<b>{best['res_A']:.2f} Å</b></span>"
             f"<span class='kpi'>particles<b>{int(best['n']) if best['n'] == best['n'] else '?'}</b></span>"
             f"<span class='kpi'>noise index<b>{best['noise']:.2f}</b></span>"
             f"<span class='kpi'>F<b>{best['F']:.3f}</b></span>"
             f"<span class='kpi'>P(deeper minimum)<b>{deep['p_deeper']:.0%}</b></span></div>")
    w = fe["weights"]
    H.append("<h2>Free energy used</h2><p>F = "
             f"{w['resolution']}·ln(d/{meta['d_ref']:.2f} Å) + {w['particles']}·E<sub>particles</sub>({fe['particles_preference']}) "
             f"+ {w['noise']}·noise + {w.get('compute', 0)}·ln(1+hours). Noise = weighted mean of available components "
             f"{', '.join(f'{k}×{v}' for k, v in cfg['noise']['components'].items() if v)} (each 0 = clean … 1 = pure noise).</p>")
    H.append("<h3>How much to trust the model</h3><p>Cross-validated skill of the learned dynamics "
             "(R² near 1 = predictive; near 0 or negative = the landscape is under-sampled in that direction).</p>" + _tbl(cv))

    H.append("<h2>The landscape I explored</h2>" + _img(fig_tree(out, dm.tr, eff["path"]), "tree"))
    p = fig_pca(dm)
    if p:
        H.append(_img(p, "pca"))
    cols = ["outcome", "kind", "res_A", "n", "noise", "E_res", "E_particles", "E_noise", "F"]
    H.append("<h3>Lowest-F states</h3>" + _tbl(out.sort_values("F")[cols], 12))

    ia = imp[imp["feature"].str.startswith("a.")]
    st = imp[~imp["feature"].str.startswith("a.")].head(3)
    H.append("<h2>Which parameters matter</h2>" + _img(fig_importance(ia), "importance"))
    H.append("<p class='muted'>Coordinates prefixed <code>mask.</code> come from the MaskCreate job used, <code>sel.</code> from my class selection, "
             "<code>mod.</code> are jobs run in between (e.g. CtfRefineTomo). The state I start a job from matters too: most influential state variables were "
             + ", ".join(st["feature"]) + ".</p>")
    it = fig_interactions(inter, list(inter[["feature_a", "feature_b"]].stack().unique()) if len(inter) else [])
    if it:
        H.append(_img(it, "interactions"))
        H.append("<p>Strongest pairwise interactions (H² ≳ 0.1 means the effect of one depends on the other):</p>" + _tbl(inter, 8))
    s = fig_slices(slices)
    if s:
        H.append(_img(s, "slices"))

    H.append("<h2>Q1 · How efficiently did I reach the minimum?</h2>" + _img(fig_progress(eff), "progress"))
    H.append("<div class='card'>"
             f"<span class='kpi'>jobs until best<b>{eff['jobs_until_best']}</b></span>"
             f"<span class='kpi'>jobs on final path<b>{eff['n_jobs_on_path']}</b></span>"
             f"<span class='kpi'>hours until best<b>{eff['hours_until_best']:.0f}</b></span>"
             f"<span class='kpi'>path / total compute<b>{(eff['path_hours'] / eff['total_hours'] if eff['total_hours'] else float('nan')):.0%}</b></span>"
             f"<span class='kpi'>failed jobs<b>{len(eff['failed'])}</b></span></div>")
    rtab = pd.DataFrame([
        {"ordering": "mine (chronological)", "jobs to reach minimum": rp["user_steps"], "hours to reach minimum": rp["user_hours"], "mean regret (ΔF above minimum)": rp["user_regret"]},
        {"ordering": "Bayesian optimisation from my first 3 jobs", "jobs to reach minimum": rp["bo_from_my_start"], "hours to reach minimum": rp["bo_hours_from_my_start"], "mean regret (ΔF above minimum)": rp["bo_regret_from_my_start"]},
        {"ordering": "Bayesian optimisation, random starts (median)", "jobs to reach minimum": rp["bo_median"], "hours to reach minimum": rp["bo_hours_median"], "mean regret (ΔF above minimum)": rp["bo_regret_median"]},
        {"ordering": "random order (median)", "jobs to reach minimum": rp["random_median"], "hours to reach minimum": rp["random_hours_median"], "mean regret (ΔF above minimum)": rp["random_regret_median"]}])
    H.append(f"<p><b>Replay of my own jobs.</b> The {rp['n_pool']} measured jobs I ran are re-ordered by different strategies "
             "(a job only becomes available once the job producing its input has been 'run'). Lower regret = the strategy spent less "
             "time far from the minimum. When the minimum sits at the end of a long dependency chain, every strategy needs most of the chain; "
             "the regret column is then the more telling number.</p>" + _tbl(rtab))
    H.append("<h3>Steps on my path that the model thinks could have been skipped</h3>" + _tbl(eff["skippable"]))
    fl = eff["flat"].copy()
    if len(fl):
        fl["input_state"] = fl["input_state"].fillna("<start>")
    H.append("<h3>Parameter scans from the same input</h3>" + _tbl(fl))
    top = imp[imp["feature"].str.startswith("a.")]["feature"].head(6).str[2:].tolist()
    H.append(f"<p><b>Suggested tuning order next time</b> (most to least influential): {', '.join(top)}. "
             "Tune these jointly (small batches of 3–5 jobs chosen by the planner) rather than one at a time; "
             "leave the low-influence coordinates at the values on my best path.</p>")

    H.append("<h2>Q2 · Is there a deeper minimum?</h2>")
    H.append(f"<p>Estimated probability that some next step (or short sequence) beats my best F by more than "
             f"{cfg['model']['improvement_margin']}: <b>{deep['p_deeper']:.0%}</b>. "
             "Treat high-novelty proposals (novelty &gt; 2 = further from my data than my jobs are from each other) as experiments, not predictions.</p>")
    if deep["nyquist_note"]:
        H.append(f"<div class='card'>{html.escape(deep['nyquist_note'])}</div>")
    rows = []
    os_ = deep["one_step"]
    for i in range(len(os_)):
        r = os_.iloc[i]
        act = r[dm.fs.action_cols()].to_dict()
        near, diff = describe_vs_nearest(act, r["kind"], dm, apix, diam, jobs)
        rows.append({"from state": r["from_state"], "job": r["kind"], "like": near, "change": diff,
                     "pred res (Å)": math.exp(r["pred.E_res"]) * meta["d_ref"],
                     "pred particles": math.exp(r["pred.lnfrac"]) * meta["n0"],
                     "F": r["F_mean"], "±": r["F_sd"], "P(improve)": r["P_improve"], "novelty": r["novelty"]})
    df1 = pd.DataFrame(rows)
    H.append("<h3>Best single next jobs (by expected improvement)</h3>" + _tbl(df1, 10))
    H.append("<h3>Best multi-step plans</h3>")
    for k, pl in enumerate(deep["plans"][:5], 1):
        H.append(f"<div class='card'><b>Plan {k}</b> from <code>{html.escape(pl['from_state'])}</code> ({pl['mode']}): "
                 f"final F {pl['F_final']:.3f} ± {pl['F_sd']:.3f}, P(improve) {pl['P_improve']:.0%}, "
                 f"max novelty {pl['max_novelty']:.1f}{' (extrapolation: treat as an experiment)' if pl['max_novelty'] > 2 else ''}<ol>")
        for st in pl["steps"]:
            near, diff = describe_vs_nearest(st["action"], st["kind"], dm, apix, diam, jobs)
            H.append(f"<li>{st['kind']} like <code>{html.escape(near)}</code> with {html.escape(diff)} → "
                     f"pred. {math.exp(st['res_E']) * meta['d_ref']:.2f} Å, {math.exp(st['lnfrac']) * meta['n0']:.0f} particles, noise {st['noise']:.2f}, "
                     f"~{st['hours']:.1f} h, novelty {st['novelty']:.1f}</li>")
        H.append("</ol></div>")
    H.append("<h3>Directions I never varied (the model cannot see them)</h3>" + _tbl(deep["unexplored"]))
    rh = fig_rh(deep["rh"])
    if rh:
        H.append(_img(rh, "rh"))
        if deep["rh"].get("note"):
            H.append(f"<p>{html.escape(deep['rh']['note'])}</p>")

    H.append("<h2>My recipe (distilled critical path)</h2>" + _tbl(recipe))
    if transfer_res:
        H.append(transfer_html(transfer_res))
    H.append("<p class='muted'>Generated by sta_landscape. Predictions are statistical summaries of my own "
             "exploration and inherit its biases; validate every proposal with a real job.</p>")
    doc = f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>STA landscape</title><style>{CSS}</style></head><body>{''.join(H)}</body></html>"
    with open(path, "w") as fh:
        fh.write(doc)
    return path


def transfer_html(T):
    t = T["target"]
    H = [f"<h2>Transfer plan for {html.escape(str(t.get('name', 'target')))}</h2>"]
    ph = T["physics"]
    H.append(f"<p>Box: <b>{T['box_px']} px</b> at {t['pixel_size_A']} Å/px ({html.escape(str(T['box_rule']))}). "
             f"Physics prior on attainable resolution: {ph['d_target']:.2f} Å (source {ph['d_source']:.2f} Å, "
             f"N<sub>eff</sub> {ph['neff_source']:.0f} → {ph['neff_target']:.0f}, B = {ph['B']:.0f} Å²; Nyquist {ph['nyquist_target']:.2f} Å). "
             f"{html.escape(ph['note'])}</p>")
    for s in T["steps"]:
        H.append(f"<div class='card'><b>Step {s['step']}: {s['kind']}</b> (mirrors <code>{s['source_job']}</code>, "
                 f"which gave {s['source_res_A']:.2f} Å from {s['source_n']:.0f} particles)<br>")
        if s["run_before"]:
            H.append(f"Run before this step: {', '.join(s['run_before'])}<br>")
        if s["mask"]:
            H.append(f"<code>{html.escape(s['mask'])}</code><br>")
        if s["kind"] == "classify" and s["select_fraction"] is not None:
            H.append(f"Selection: keep ≈{s['select_fraction']:.0%} of particles ({s['n_select_classes']:.0f} class(es)).<br>")
        H.append(f"Model prediction: {s['pred_res_A']:.2f} Å, {s['pred_n']:.0f} particles, noise {s['pred_noise']:.2f}")
        H.append(_tbl(s["params"]))
        if s["command"]:
            H.append(f"<pre>{html.escape(s['command'])}</pre>")
        H.append("</div>")
    if T["alternatives"]:
        H.append("<h3>Model-optimised alternatives</h3>")
        for k, a in enumerate(T["alternatives"], 1):
            H.append(f"<div class='card'><b>Alternative {k}</b>: final F {a['F_final']:.3f} ± {a['F_sd']:.3f}<ol>")
            for s in a["steps"]:
                fl = " ".join(f"--{k2}" if v is True else f"--{k2} {v}" for k2, v in s["flags"].items())
                sel = f" keep ≈{s['sel_frac']:.0%}" if s["kind"] == "classify" and s["sel_frac"] else ""
                pre = f" [after {', '.join(s['run_before'])}]" if s["run_before"] else ""
                H.append(f"<li>{s['kind']}{pre}: <code>{html.escape(fl)}</code>{sel} → {s['pred_res_A']:.2f} Å, "
                         f"{s['pred_n']:.0f} particles</li>")
            H.append("</ol></div>")
    return "".join(H)
