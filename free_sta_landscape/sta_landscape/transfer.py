"""Transfer a learned recipe to a modified particle (extra subunit, dimer, bigger box, ...).

Three layers, from most to least conservative:
  1. Recipe transfer: replay the job sequence on my critical path, with every parameter
     re-expressed for the new particle through its dimensionless coordinate
     (mask diameter/particle diameter, angular step as arc length at the particle edge,
     E-step limit relative to Nyquist, offsets in A, mask extension in A) plus explicit rules
     for symmetry, number of classes and box size.
  2. Model prediction: the dynamics model walks the new particle through that recipe
     (if I trained on several particles, particle descriptors are model inputs).
  3. Physics prior: Rosenthal-Henderson scaling of the attainable resolution with the effective
     number of asymmetric units (particles x symmetry x mass ratio).
The model-optimised alternative (beam search from the new particle's starting state) is given too.
"""
from __future__ import annotations

import math
import re
from typing import Dict

import numpy as np
import pandas as pd

from .analysis import critical_path, plan_beam
from .codec import GUI_NAMES, decode_features, sym_order
from .relion_project import NON_SCIENTIFIC_FLAGS, _clean_tokens

def _smooth(b):
    for p in (2, 3, 5, 7):
        while b % p == 0:
            b //= p
    return b == 1


def good_box(n):
    """Smallest even box >= n whose prime factors are all in {2,3,5,7} (FFT friendly)."""
    b = max(2, int(math.ceil(n)))
    b += b % 2
    while not _smooth(b):
        b += 2
    return b


def _num(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def substitute_command(cmd: str, new_flags: Dict[str, object], removed=(), kind="refine"):
    """Rewrite one of my real command lines with new parameter values, keeping everything else
    verbatim and replacing project-specific paths by placeholders."""
    toks = _clean_tokens(cmd)
    prog_i = next((i for i, t in enumerate(toks) if re.match(r"^relion_\w+", t.split("/")[-1])), 0)
    prog = toks[prog_i]
    placeholders = {"o": "<NEW_JOB_DIR>/run", "i": "<INPUT_PARTICLES>", "ios": "<INPUT_OPTIMISATION_SET>",
                    "ref": "<REFERENCE_MAP>", "solvent_mask": "<MASK>", "pipeline_control": "<NEW_JOB_DIR>/",
                    "t": "<TOMOGRAMS>", "tomograms": "<TOMOGRAMS>", "traj": "<TRAJECTORIES>", "mot": "<MOTION>"}
    # group tokens: [(flag, [raw value tokens])]
    groups, cur = [], None
    for t in toks[prog_i + 1:]:
        if t.startswith("--") and len(t) > 2 and not re.match(r"^--?\d", t):
            cur = (t[2:], [])
            groups.append(cur)
        elif cur is not None:
            cur[1].append(t)
    parts, done = [prog], set()
    for k, vals in groups:
        done.add(k)
        if k in removed:
            continue
        if k in placeholders:
            parts.append(f"--{k} {placeholders[k]}")
        elif k in new_flags and k not in NON_SCIENTIFIC_FLAGS:
            nv = new_flags[k]
            parts.append(f"--{k}" if nv is True else f"--{k} {_num(nv)}")
        else:
            raw = " ".join('""' if v == "" else v for v in vals)
            parts.append(f"--{k} {raw}".rstrip())
    from .codec import FLAG_DEFAULTS
    for k, v in new_flags.items():
        if k in done:
            continue
        if kind == "refine" and k == "K":
            continue
        if k in FLAG_DEFAULTS and not isinstance(v, bool) and float(v) == float(FLAG_DEFAULTS[k]):
            continue  # RELION default, no need to spell it out
        parts.append(f"--{k}" if v is True else f"--{k} {_num(v)}")
    return " ".join(parts)


def _fmt_int(v):
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return v


def transfer(dm, out: pd.DataFrame, tr: pd.DataFrame, jobs: dict, meta: dict, target: dict, cfg, eff) -> dict:
    src = meta["particle"]
    t = target
    apix_t = float(t["pixel_size_A"])
    diam_t = float(t["diameter_A"])
    diam_s = float(meta["diameter"])
    sym_t = str(t.get("symmetry", "C1"))
    sym_s = str(meta.get("sym", "C1"))
    mass_ratio = (float(t["mass_kda"]) / float(src["mass_kda"])) if (t.get("mass_kda") and src.get("mass_kda")) else 1.0
    n0_t = float(t.get("n_particles_start") or meta["n0"])
    n0_s = float(meta["n0"])
    best = eff["best"]
    path = critical_path(tr, best["outcome"])
    trx = tr.set_index("outcome")

    # --- box size
    src_box_A = float(best["box_px"]) * float(best["apix"]) if best["box_px"] == best["box_px"] else None
    box_rule = None
    if t.get("box_px"):
        box_t = int(t["box_px"])
        box_rule = "given in target file"
    elif src_box_A:
        box_t = good_box(int(math.ceil(src_box_A / diam_s * diam_t / apix_t)))
        box_rule = f"keeps box/diameter = {src_box_A / diam_s:.2f} of the source, rounded to an FFT-friendly size"
    else:
        box_t = None

    ctx = {"ctx.apix": apix_t, "ctx.box_A": (box_t or 0) * apix_t, "ctx.sym_order": float(sym_order(sym_t)),
           "ctx.mass_kda": float(t.get("mass_kda") or np.nan), "ctx.diameter_A": diam_t,
           "ctx.n_subunits": float(t.get("n_subunits") or 1), "ctx.flexibility": float(t.get("flexibility") or np.nan)}

    steps = []
    first = trx.loc[[path[0]]].iloc[0]
    ini = float(np.exp(first["pre.E_res"]) * meta["d_ref"])
    state = {"pre.E_res": math.log(ini / meta["d_ref"]), "pre.lnfrac": 0.0, "pre.noise": 1.0, "pre.gold": 0.0,
             "pre.depth": 0.0, "pre.n_class_rounds": 0.0, "pre.n_refine_rounds": 0.0, **ctx}
    start_state = dict(state)
    extra_k = int(t.get("extra_heterogeneity", 0) or 0)
    for k, oc in enumerate(path):
        row = trx.loc[[oc]].iloc[0]
        job = jobs[row["job"]]
        feat = {c[2:]: float(row[c]) for c in row.index if c.startswith("a.") and row[c] == row[c]}
        notes = {}
        if row["kind"] == "classify" and "K" in feat:
            k_src = feat["K"]
            k_new = int(np.clip(round(k_src * (n0_t / n0_s) ** 0.5), max(2, k_src - 2), k_src + 3)) + extra_k
            if k_new != k_src:
                notes["K"] = f"particles-per-class scaling (sqrt of particle ratio {n0_t / n0_s:.2f}) + {extra_k} extra class(es) for new heterogeneity"
            feat["K"] = k_new
        bools = {c[2:] for c in dm.fs.bool_cols}
        ints = {c[2:] for c in dm.fs.int_cols} | {"K", "iter"}
        new_flags = decode_features(feat, apix_t, diam_t, bool_cols=bools, int_cols=ints)
        if row["kind"] == "refine":
            new_flags.pop("K", None)
        old_flags = {k2: v for k2, v in job.flags.items() if k2 not in NON_SCIENTIFIC_FLAGS}
        new_flags["sym"] = sym_t if row["kind"] == "refine" or sym_order(sym_t) == 1 else old_flags.get("sym", "C1")
        if row["kind"] == "classify" and sym_order(sym_t) > 1:
            notes["sym"] = (f"classification kept in {new_flags['sym']} (as in my source run); "
                            f"consider {sym_t} once classes are clean, or --relax_sym {sym_t} for pseudo-symmetric dimers")
        elif new_flags["sym"] != old_flags.get("sym", "C1"):
            notes["sym"] = f"target symmetry {sym_t}"
        removed = [k2 for k2, v in old_flags.items() if v is True and k2 not in new_flags]
        table = []
        from .codec import FLAG_DEFAULTS

        def _same(a, b, k):
            if a == "-" and k in FLAG_DEFAULTS:
                a = FLAG_DEFAULTS[k]
            try:
                return abs(float(a) - float(b)) < 1e-6
            except (TypeError, ValueError):
                return str(a) == str(b)
        rules = {"particle_diameter": "scaled with particle diameter",
                 "offset_range": "kept constant in Angstrom", "offset_step": "kept constant in Angstrom",
                 "strict_highres_exp": "kept constant relative to Nyquist",
                 "healpix_order": "same arc length at particle edge (bigger particle -> finer sampling)",
                 "auto_local_healpix_order": "same arc length at particle edge"}
        key = {"K", "tau2_fudge", "healpix_order", "auto_local_healpix_order", "strict_highres_exp", "particle_diameter", "sym"}
        for fk in sorted(set(old_flags) | set(new_flags)):
            if fk in NON_SCIENTIFIC_FLAGS or (row["kind"] == "refine" and fk == "K"):
                continue
            ov = old_flags.get(fk, "-")
            nv = new_flags.get(fk, "(off)" if old_flags.get(fk) is True else "-")
            if ov == "-" and nv == "-":
                continue
            ov_s, nv_s = (_num(ov) if isinstance(ov, float) else str(ov)), (_num(nv) if isinstance(nv, float) else str(nv))
            changed = not _same(ov, nv, fk)
            if ov == "-" and not changed:
                continue  # RELION default filled in by the model, nothing to do
            rule = notes.get(fk, rules.get(fk, "") if changed else "")
            if changed or fk in key:
                table.append({"flag": fk, "gui": GUI_NAMES.get(fk) or "", "source": ov_s, "target": nv_s,
                              "rule": rule or ("" if changed else "unchanged")})
        mask_txt = None
        if feat.get("mask.has_mask"):
            ext = feat.get("mask.extend_A")
            soft = feat.get("mask.soft_edge_A")
            mask_txt = ("New mask for the target: relion_mask_create"
                        + (f" --lowpass {feat['mask.lowpass_A']:.0f}" if "mask.lowpass_A" in feat else "")
                        + (f" --extend_inimask {max(1, round(ext / apix_t))}" if ext else "")
                        + (f" --width_soft_edge {max(2, round(soft / apix_t))}" if soft else "")
                        + (f" --ini_threshold <re-pick for new map, source used {feat['mask.ini_threshold']:g}>" if "mask.ini_threshold" in feat else "")
                        + f" --angpix {apix_t}")
        mods = [c.split(".", 1)[1] for c, v in feat.items() if c.startswith("mod.") and v >= 1]
        # model prediction for the target
        df = pd.DataFrame([{**state, **{"a." + k2: v for k2, v in feat.items()},
                            "kind.classify": float(row["kind"] == "classify"), "kind.refine": float(row["kind"] == "refine")}])
        if "a.K" in df and row["kind"] == "classify":
            df["a.K"] = feat["K"]
        F, st = dm.predict_F(df)
        res_pred = float(np.exp(st["E_res"][0]) * meta["d_ref"])
        cmd = substitute_command(job.commands[0], dict(new_flags), removed, row["kind"]) if job.commands else None
        steps.append(dict(step=k + 1, kind=row["kind"], source_job=row["job"], source_outcome=oc,
                          source_res_A=float(np.exp(row["post.E_res"]) * meta["d_ref"]),
                          source_n=float(np.exp(row["post.lnfrac"]) * n0_s),
                          params=pd.DataFrame(table), mask=mask_txt, run_before=mods,
                          select_fraction=feat.get("sel.frac"), n_select_classes=feat.get("sel.n_classes"),
                          pred_res_A=res_pred, pred_res_sd_logunits=float(st["sd_E_res"][0]),
                          pred_n=float(np.exp(st["lnfrac"][0]) * n0_t), pred_noise=float(st["noise"][0]),
                          pred_F_sourceref=float(F.mean()), pred_F_sd=float(F.std()), command=cmd))
        state.update({"pre.E_res": float(st["E_res"][0]), "pre.lnfrac": float(st["lnfrac"][0]),
                      "pre.noise": float(st["noise"][0]), "pre.gold": float(row["kind"] == "refine"),
                      "pre.depth": state["pre.depth"] + 1,
                      "pre.n_class_rounds": state["pre.n_class_rounds"] + (row["kind"] == "classify"),
                      "pre.n_refine_rounds": state["pre.n_refine_rounds"] + (row["kind"] == "refine")})

    # --- physics prior (Rosenthal-Henderson) on the final resolution
    B = eff.get("rh_B") or cfg["physics"]["default_bfactor_A2"]
    alpha = cfg["physics"].get("mass_exponent", 1.0)
    d_s = float(best["res_A"])
    n_s = float(best["n"])
    frac = n_s / n0_s
    neff_s = n_s * sym_order(sym_s)
    neff_t = frac * n0_t * sym_order(sym_t) * mass_ratio ** alpha
    inv2 = 1.0 / d_s ** 2 + (2.0 / B) * math.log(neff_t / neff_s)
    d_phys = max(1.0 / math.sqrt(inv2), 2 * apix_t) if inv2 > 0 else float("nan")
    phys = dict(B=B, neff_source=neff_s, neff_target=neff_t, d_source=d_s, d_target=d_phys,
                nyquist_target=2 * apix_t,
                note=("Effective asymmetric units = particles kept x symmetry order x (mass ratio)^%.1f; "
                      "1/d_t^2 = 1/d_s^2 + (2/B) ln(Neff_t/Neff_s)." % alpha))

    # --- model-optimised alternative from the target's starting state
    alt = plan_beam(dm, start_state, mode="expected")
    alt_out = []
    for p in alt[:3]:
        alt_steps = []
        for s in p["steps"]:
            feat = {c[2:]: v for c, v in dm.fs.full_action(s["action"], s["kind"]).items()}
            fl = decode_features(feat, apix_t, diam_t, bool_cols={c[2:] for c in dm.fs.bool_cols},
                                 int_cols={c[2:] for c in dm.fs.int_cols} | {"K", "iter"})
            alt_steps.append(dict(kind=s["kind"], flags=fl, sel_frac=feat.get("sel.frac"),
                                  pred_res_A=float(np.exp(s["res_E"]) * meta["d_ref"]),
                                  pred_n=float(np.exp(s["lnfrac"]) * n0_t), pred_noise=s["noise"], F=s["F_mean"], F_sd=s["F_sd"],
                                  run_before=[c.split(".", 1)[1] for c, v in feat.items() if c.startswith("mod.") and v >= 0.5]))
        alt_out.append(dict(steps=alt_steps, F_final=p["steps"][-1]["F_mean"], F_sd=math.sqrt(p["var"])))
    return dict(target=t, box_px=box_t, box_rule=box_rule, steps=steps, physics=phys, alternatives=alt_out,
                source_best=dict(outcome=best["outcome"], res_A=d_s, n=n_s, F=float(best["F"])))
