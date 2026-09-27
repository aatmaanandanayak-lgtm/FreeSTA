"""End-to-end pipeline used by the command line."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .analysis import deeper_minimum, efficiency, importance, interactions, local_slices
from .codec import GUI_NAMES
from .landscape import build_landscape
from .models import DynamicsModel
from .relion_project import jobs_table, scan_project
from .report import action_to_flags, build_report
from .transfer import transfer


def _log(msg):
    print(f"[sta_landscape] {msg}", flush=True)


def scan_all(cfg):
    projects = cfg["projects"]
    if not projects:
        raise SystemExit("No projects in config (see `python -m sta_landscape init`).")
    multi = len(projects) > 1
    all_out, all_tr, jobsets, metas, tables = [], [], [], [], []
    for i, p in enumerate(projects):
        _log(f"scanning {p['dir']}")
        jobs = scan_project(p["dir"])
        out, tr, meta = build_landscape(jobs, p, cfg, i)
        _log(f"  {len(jobs)} jobs, {len(out)} measured states, {len(tr)} transitions")
        if multi and len(out):
            out["outcome"] = f"p{i}:" + out["outcome"]
            tr["outcome"] = f"p{i}:" + tr["outcome"]
            tr["pre_outcome"] = [f"p{i}:{x}" if isinstance(x, str) else None for x in tr["pre_outcome"]]
        all_out.append(out)
        all_tr.append(tr)
        jobsets.append(jobs)
        metas.append(meta)
        t = jobs_table(jobs)
        t["project_index"] = i
        tables.append(t)
    return (pd.concat(all_out, ignore_index=True), pd.concat(all_tr, ignore_index=True),
            jobsets, metas, pd.concat(tables, ignore_index=True))


def run(cfg, target=None, source_index=0, do_report=True):
    od = cfg["output_dir"]
    os.makedirs(od, exist_ok=True)
    out, tr, jobsets, metas, jt = scan_all(cfg)
    jt.to_csv(os.path.join(od, "jobs.csv"), index=False)
    out.to_csv(os.path.join(od, "states.csv"), index=False)
    tr.to_csv(os.path.join(od, "transitions.csv"), index=False)
    if len(tr) < 5:
        raise SystemExit("Fewer than 5 measured Class3D/Refine3D outcomes - not enough to learn a landscape.")
    meta = metas[source_index]
    cfg["free_energy"]["n0"] = meta["n0"]

    _log("fitting dynamics model (GP + extra-trees per state variable)")
    switches = set().union(*[m.get("switches", set()) for m in metas])
    dm = DynamicsModel(cfg).fit(tr, switches)
    cv = dm.cv_scores()
    _log("sensitivity analysis")
    imp = importance(dm, seed=cfg["model"]["seed"])
    imp["gui"] = [GUI_NAMES.get(f[2:], "") or "" for f in imp["feature"]]
    top8 = list(imp[imp["feature"].str.startswith("a.")]["feature"].head(6)) + list(imp[~imp["feature"].str.startswith("a.")]["feature"].head(2))
    inter = interactions(dm, top8, seed=cfg["model"]["seed"])

    pm = (dm.tr["project"] == source_index).to_numpy()
    trp = dm.tr[pm].reset_index(drop=True)
    outp = out[out["project"] == source_index].reset_index(drop=True)

    class _View:  # dm restricted to one project's rows for replay/efficiency
        pass
    view = _View()
    view.__dict__.update(dm.__dict__)
    view.X = dm.X[pm]
    view.tr = trp
    view.predict_F = dm.predict_F
    view.composed_F = dm.composed_F
    _log("efficiency analysis (Q1) incl. Bayesian-optimisation replay")
    eff = efficiency(outp, trp, jobsets[source_index], view, imp, cfg)
    _log("searching for deeper minima (Q2)")
    deep = deeper_minimum(dm, outp, trp, eff, cfg)
    if deep["rh"] and deep["rh"].get("ok"):
        eff["rh_B"] = deep["rh"]["B"]

    top_actions = [f for f in imp["feature"] if f.startswith("a.")][:6]
    row_idx = {}
    for oc in eff["path"]:
        i = int(np.where(dm.tr["outcome"].to_numpy() == oc)[0][0])
        row_idx[dm.tr["kind"].iloc[i]] = i   # last (deepest) job of each type on my path
    slices = local_slices(dm, row_idx, top_actions)

    # recipe: my critical path in RELION terms
    apix = float(eff["best"]["apix"])
    rows = []
    trx = trp.set_index("outcome")
    ox = outp.set_index("outcome")
    for k, oc in enumerate(eff["path"], 1):
        r = trx.loc[[oc]].iloc[0]
        o = ox.loc[[oc]].iloc[0]
        fl = action_to_flags({c: r[c] for c in dm.fs.action_cols() + [c for c in r.index if c.startswith("a.") and c not in dm.fs.cols]
                              if r.get(c) == r.get(c)}, dm, apix, meta["diameter"])
        key = [f[2:] for f in top_actions]
        shown = {k2: v for k2, v in fl.items() if k2 in key or any(k2 in s for s in key) or k2 in ("tau2_fudge", "K", "strict_highres_exp", "healpix_order", "blush")}
        if r["kind"] == "refine":
            shown.pop("K", None)
        rows.append({"step": k, "job": r["job"], "select": o.get("select_job") if isinstance(o.get("select_job"), str) else "", "kind": r["kind"],
                     "run_before": ", ".join(c[6:] for c in r.index if c.startswith("a.mod.") and r[c] == r[c] and r[c] >= 1),
                     "key parameters": " ".join(f"--{k2}" if v is True else f"--{k2} {v}" for k2, v in shown.items()),
                     "kept": f"{r['a.sel.frac']:.0%}" if r["kind"] == "classify" and "a.sel.frac" in r.index else "",
                     "res (Å)": o["res_A"], "particles": o["n"], "noise": o["noise"], "F": o["F"]})
    recipe = pd.DataFrame(rows)

    T = None
    if target:
        _log("building transfer plan for the target particle")
        T = transfer(dm, outp, trp, jobsets[source_index], meta, target, cfg, eff)
        with open(os.path.join(od, "transfer_commands.txt"), "w") as fh:
            for s in T["steps"]:
                fh.write(f"# Step {s['step']}: {s['kind']} (mirrors {s['source_job']})\n")
                if s["run_before"]:
                    fh.write(f"#   run before: {', '.join(s['run_before'])}\n")
                if s["mask"]:
                    fh.write(f"#   {s['mask']}\n")
                if s["command"]:
                    fh.write(s["command"] + "\n")
                fh.write("\n")

    # --------------------------------------------------------------- write outputs
    cv.to_csv(os.path.join(od, "model_skill.csv"), index=False)
    imp.to_csv(os.path.join(od, "sensitivity.csv"), index=False)
    inter.to_csv(os.path.join(od, "interactions.csv"), index=False)
    eff["timeline"].to_csv(os.path.join(od, "timeline.csv"), index=False)
    eff["skippable"].to_csv(os.path.join(od, "skippable_steps.csv"), index=False)
    eff["flat"].to_csv(os.path.join(od, "parameter_scans.csv"), index=False)
    deep["one_step"].to_csv(os.path.join(od, "next_job_candidates.csv"), index=False)
    recipe.to_csv(os.path.join(od, "recipe.csv"), index=False)

    def conv(o):
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="records")
        if isinstance(o, pd.Series):
            return o.to_dict()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    summary = dict(best=eff["best"].to_dict(), path=eff["path"], replay=eff["replay"],
                   jobs_until_best=eff["jobs_until_best"], hours_until_best=eff["hours_until_best"],
                   p_deeper=deep["p_deeper"], nyquist_note=deep["nyquist_note"],
                   plans=deep["plans"], unexplored=deep["unexplored"], model_skill=cv)
    with open(os.path.join(od, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=conv)
    if T:
        with open(os.path.join(od, "transfer_plan.json"), "w") as fh:
            json.dump(T, fh, indent=1, default=conv)
    rp = None
    if do_report:
        rp = build_report(os.path.join(od, "report.html"), cfg, out[out["project"] == source_index], dm, cv, imp, inter,
                          slices, eff, deep, recipe, meta, jobsets[source_index], T,
                          fig_dir=os.path.join(od, "figures"))
        _log(f"report written to {rp}")
    return dict(out=out, tr=tr, dm=dm, cv=cv, imp=imp, inter=inter, eff=eff, deep=deep, recipe=recipe, transfer=T, report=rp)
