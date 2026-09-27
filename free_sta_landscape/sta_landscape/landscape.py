"""Turn a scanned RELION job graph into a 'free-energy landscape':

  outcome  = a measurable state (resolution, particle count, noise) produced by a
             Class3D (+ the Select job that chose classes from it) or a Refine3D
             (+ its PostProcess)
  transition = (state before) --[job type + parameters]--> (state after)

F = w_res*E_res + w_part*E_part + w_noise*E_noise + w_comp*E_comp          (lower = better)
  E_res   = ln(d / d_ref)               d = resolution in A, d_ref = Nyquist of finest pixel size
  E_part  = -ln(N/N0) | +ln(N/N0) | (ln N/N_target)^2 | 0     (more | fewer | target | none)
  E_noise = weighted mean of available noise components, each in [0,1]
  E_comp  = ln(1 + cumulative compute hours along the lineage)
"""
from __future__ import annotations

import itertools
import math
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .codec import FLAG_DEFAULTS, encode_flags, mask_features, sym_order
from .relion_project import Job, job_kind

try:
    import mrcfile  # optional
except ImportError:  # pragma: no cover
    mrcfile = None


# ----------------------------------------------------------------------------- noise
def spectral_noise_fraction(res_inv_A, ssnr, apix, band) -> Optional[float]:
    """Mean of 1/(1+SSNR) over a frequency band (fractions of Nyquist). 0 = all signal, 1 = all noise."""
    if res_inv_A is None or apix is None:
        return None
    f = np.asarray(res_inv_A, float)
    s = np.clip(np.asarray(ssnr, float), 0, None)
    fny = 1.0 / (2.0 * apix)
    sel = (f >= band[0] * fny) & (f <= band[1] * fny)
    if sel.sum() < 2:
        return None
    return float(np.mean(1.0 / (1.0 + s[sel])))


def _read_map(p):
    if mrcfile is None or not p or not os.path.isfile(p):
        return None
    with mrcfile.open(p, permissive=True) as m:
        return np.asarray(m.data, dtype=np.float32)


def _sphere(shape, radius_px):
    z, y, x = np.indices(shape)
    c = [(s - 1) / 2 for s in shape]
    r = np.sqrt((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2)
    return (r <= radius_px).astype(np.float32)


def map_noise(h1p, h2p, mask_p=None, radius_px=None):
    """Half-map noise fraction var(h1-h2)/var(h1+h2) inside mask and solvent/protein std ratio."""
    h1, h2 = _read_map(h1p), _read_map(h2p)
    if h1 is None or h2 is None:
        return {}
    mask = _read_map(mask_p) if mask_p else None
    if mask is None or mask.shape != h1.shape:
        mask = _sphere(h1.shape, radius_px or min(h1.shape) * 0.4)
    inside = mask > 0.5
    s, d = (h1 + h2)[inside], (h1 - h2)[inside]
    out = {}
    if s.var() > 0:
        out["halfmap"] = float(np.clip(d.var() / s.var(), 0, 1))
    avg = (h1 + h2) / 2
    outside = mask < 0.01
    if outside.sum() > 100 and avg[inside].std() > 0:
        out["solvent"] = float(np.clip(avg[outside].std() / avg[inside].std(), 0, 1))
    return out


def single_map_solvent(p, mask_p=None, radius_px=None):
    m = _read_map(p)
    if m is None:
        return None
    mask = _read_map(mask_p) if mask_p else None
    if mask is None or mask.shape != m.shape:
        mask = _sphere(m.shape, radius_px or min(m.shape) * 0.4)
    inside, outside = mask > 0.5, mask < 0.01
    if outside.sum() < 100 or m[inside].std() == 0:
        return None
    return float(np.clip(m[outside].std() / m[inside].std(), 0, 1))


def combine_noise(comps: Dict[str, float], weights: Dict[str, float]) -> float:
    num = den = 0.0
    for k, v in comps.items():
        w = weights.get(k, 0.0)
        if v is None or w <= 0 or (isinstance(v, float) and np.isnan(v)):
            continue
        num += w * float(v)
        den += w
    return num / den if den > 0 else np.nan


# ----------------------------------------------------------------------------- free energy
def free_energy_terms(res_A, n, n0, noise, hours, cfg, d_ref):
    fe = cfg["free_energy"]
    e_res = math.log(res_A / d_ref) if (res_A and res_A > 0) else np.nan
    pref = str(fe.get("particles_preference", "more")).lower()
    frac = (n / n0) if (n and n0) else np.nan
    if pref == "more":
        e_n = -math.log(frac) if frac and frac > 0 else np.nan
    elif pref == "fewer":
        e_n = math.log(frac) if frac and frac > 0 else np.nan
    elif pref == "target" and fe.get("particles_target"):
        e_n = math.log(n / float(fe["particles_target"])) ** 2 if n else np.nan
    else:
        e_n = 0.0
    e_c = math.log1p(hours) if hours is not None and not np.isnan(hours) else 0.0
    return {"E_res": e_res, "E_particles": e_n, "E_noise": noise, "E_compute": e_c}


def F_from_terms(t, cfg):
    w = cfg["free_energy"]["weights"]
    tot = 0.0
    for key, wk in (("E_res", "resolution"), ("E_particles", "particles"), ("E_noise", "noise"), ("E_compute", "compute")):
        v = t.get(key)
        if w.get(wk, 0) == 0:
            continue
        if v is None or (isinstance(v, float) and np.isnan(v)):
            if key == "E_noise":
                v = 1.0  # unknown noise is treated as maximal (pessimistic)
            else:
                return np.nan
        tot += w[wk] * v
    return tot


def F_from_state(e_res, lnfrac, noise, cfg, e_comp=0.0):
    """Same F but from the state variables used by the dynamics model."""
    fe = cfg["free_energy"]
    pref = str(fe.get("particles_preference", "more")).lower()
    e_n = {"more": -lnfrac, "fewer": lnfrac}.get(pref, 0.0)
    if pref == "target" and fe.get("particles_target") and "n0" in fe:
        e_n = (lnfrac + math.log(fe["n0"] / float(fe["particles_target"]))) ** 2
    w = fe["weights"]
    return (w.get("resolution", 0) * e_res + w.get("particles", 0) * e_n
            + w.get("noise", 0) * np.clip(noise, 0, 1) + w.get("compute", 0) * e_comp)


# ----------------------------------------------------------------------------- outcomes
def _infer_selected(class_counts: Dict[int, int], n_sel: int) -> Optional[List[int]]:
    if not class_counts or not n_sel:
        return None
    ks = sorted(class_counts)
    if len(ks) > 14:
        return None
    best, best_err = None, None
    for r in range(1, len(ks) + 1):
        for combo in itertools.combinations(ks, r):
            err = abs(sum(class_counts[k] for k in combo) - n_sel)
            if best_err is None or err < best_err:
                best, best_err = list(combo), err
    if best_err is not None and best_err <= max(2, 0.01 * n_sel):
        return best
    return None


def _annotations(path):
    if not path or not os.path.isfile(path):
        return {}
    df = pd.read_csv(path)
    return {str(r["job"]).rstrip("/"): r.to_dict() for _, r in df.iterrows()}


def build_landscape(jobs: Dict[str, Job], proj_cfg: dict, cfg: dict, project_index: int = 0):
    """Return (outcomes DataFrame, transitions DataFrame) for one project."""
    ncfg = cfg["noise"]
    band = ncfg.get("spectral_band", [0.2, 0.8])
    wn = ncfg["components"]
    ann = _annotations(proj_cfg.get("annotations"))
    excl = set(j.rstrip("/") for j in proj_cfg.get("exclude_jobs", []))
    for k, a in ann.items():
        if str(a.get("exclude", "")).lower() in ("1", "true", "yes"):
            excl.add(k)
    part = proj_cfg.get("particle", {})
    use_maps = ncfg.get("use_maps", True) and mrcfile is not None

    ok = {n: j for n, j in jobs.items() if j.status in ("succeeded", "unknown") and n not in excl}
    apixes = [j.metrics.get("apix") for j in ok.values() if j.metrics.get("apix")]
    d_ref = cfg["free_energy"].get("resolution_reference_A") or (2.0 * min(apixes) if apixes else 1.0)
    diameter = part.get("diameter_A")
    if not diameter:
        ds = [float(j.flags["particle_diameter"]) for j in ok.values() if "particle_diameter" in j.flags]
        diameter = float(np.median(ds)) if ds else 200.0
    counts = [j.metrics.get("n_particles") for j in ok.values() if job_kind(j) in ("classify", "refine") and j.metrics.get("n_particles")]
    n0 = part.get("n_particles_start") or (max(counts) if counts else None)
    sym = part.get("symmetry", "C1")

    outcomes = []

    def user_noise(jobname):
        a = ann.get(jobname, {})
        s = a.get("noise_score")
        try:
            s = float(s)
            return (s - 1.0) / 4.0 if not np.isnan(s) else None
        except (TypeError, ValueError):
            return None

    def lineage_hours(jobname):
        h, seen, cur = 0.0, set(), jobname
        while cur and cur in jobs and cur not in seen:
            seen.add(cur)
            hh = jobs[cur].hours
            h += 0 if np.isnan(hh) else hh
            cur = jobs[cur].parent
        return h

    for name, j in sorted(ok.items(), key=lambda kv: kv[1].number):
        kind = job_kind(j)
        m = j.metrics
        apix = m.get("apix")
        if kind == "classify" and m.get("class_res"):
            dist = m.get("class_dist") or []
            cres = m.get("class_res") or []
            counts_c = m.get("class_counts") or {}
            n_in = m.get("n_particles") or (sum(counts_c.values()) if counts_c else None)
            sels = [jobs[c] for c in j.children if c in ok and job_kind(jobs[c]) == "select"]
            variants = []
            for s in sels:
                selc = s.metrics.get("selected_classes") or _infer_selected(counts_c, s.metrics.get("n_particles"))
                variants.append((f"{name}|{s.name}", s, selc, s.metrics.get("n_particles")))
            if not variants:
                cand = [k for k in range(1, len(cres) + 1) if (dist[k - 1] if k - 1 < len(dist) else 0) >= 0.05] or list(range(1, len(cres) + 1))
                best = min(cand, key=lambda k: cres[k - 1] if cres[k - 1] > 0 else 1e9)
                nb = counts_c.get(best) if counts_c else (int(round(dist[best - 1] * n_in)) if n_in else None)
                variants.append((name, None, [best], nb))
            for oid, sjob, selc, n_sel in variants:
                if not selc:
                    selc = list(range(1, len(cres) + 1))
                w = np.array([counts_c.get(k, (dist[k - 1] if k - 1 < len(dist) else 0) * (n_in or 1)) for k in selc], float)
                w = w / w.sum() if w.sum() > 0 else np.ones(len(selc)) / len(selc)
                rr = np.array([cres[k - 1] for k in selc], float)
                res = float(np.sum(w * rr))
                comps = {}
                sp = [spectral_noise_fraction(*m["class_ssnr"][k], apix, band) for k in selc if k in m.get("class_ssnr", {})]
                sp = [x for x in sp if x is not None]
                comps["spectral"] = float(np.sum(w[:len(sp)] * np.array(sp)) / max(w[:len(sp)].sum(), 1e-9)) if sp else None
                comps["pmax"] = 1.0 - float(m["pmax"]) if m.get("pmax") is not None else None
                acc = m.get("class_acc_rot")
                if acc:
                    comps["angular_accuracy"] = float(np.clip(np.mean([acc[k - 1] for k in selc]) / ncfg["angular_accuracy_scale_deg"], 0, 1))
                if use_maps and m.get("last_iter") is not None:
                    vals = []
                    for k in selc:
                        mp = os.path.join(j.project, name, f"run_it{m['last_iter']:03d}_class{k:03d}.mrc")
                        maskp = os.path.join(j.project, j.flags["solvent_mask"]) if isinstance(j.flags.get("solvent_mask"), str) else None
                        v = single_map_solvent(mp, maskp, (float(j.flags.get("particle_diameter", diameter)) / 2 / apix) if apix else None)
                        if v is not None:
                            vals.append(v)
                    comps["solvent"] = float(np.mean(vals)) if vals else None
                comps["user"] = user_noise(sjob.name if sjob else name) or user_noise(name)
                a = ann.get(sjob.name if sjob else name, {})
                if a.get("override_res") == a.get("override_res") and a.get("override_res") not in (None, ""):
                    res = float(a["override_res"])
                if a.get("override_n") == a.get("override_n") and a.get("override_n") not in (None, ""):
                    n_sel = int(a["override_n"])
                outcomes.append(dict(
                    outcome=oid, project=project_index, job=name, select_job=sjob.name if sjob else None,
                    kind="classify", number=(sjob.number if sjob else j.number), res_A=res, gold=0,
                    n=n_sel, n_in=n_in, sel_frac=(n_sel / n_in) if (n_sel and n_in) else np.nan,
                    n_sel_classes=len(selc), selected_classes=",".join(map(str, selc)),
                    apix=apix, box_px=m.get("box_px"), noise_comps=comps, hours=j.hours,
                    lineage_hours=lineage_hours(name)))
        elif kind == "refine" and m.get("res"):
            res = float(m["res"])
            pps = [jobs[c] for c in j.children if c in ok and job_kind(jobs[c]) == "postprocess" and jobs[c].metrics.get("res")]
            comps = {}
            bfac = None
            if pps:
                pp = min(pps, key=lambda p: p.metrics["res"])
                res = float(pp.metrics["res"])
                bfac = pp.metrics.get("bfactor")
                fsc = pp.metrics.get("fsc_table")
                if isinstance(fsc, pd.DataFrame) and "rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps" in fsc.columns:
                    f = fsc["rlnResolution"].to_numpy(float)
                    pr = fsc["rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps"].to_numpy(float)
                    sel = f > (1.0 / res)
                    comps["mask_artefact"] = float(np.clip(np.nanmean(np.abs(pr[sel])), 0, 1)) if sel.sum() else None
            if m.get("ssnr"):
                comps["spectral"] = spectral_noise_fraction(*m["ssnr"], apix, band)
            comps["pmax"] = 1.0 - float(m["pmax"]) if m.get("pmax") is not None else None
            if m.get("acc_rot") is not None:
                comps["angular_accuracy"] = float(np.clip(m["acc_rot"] / ncfg["angular_accuracy_scale_deg"], 0, 1))
            if use_maps and m.get("half_maps"):
                maskp = os.path.join(j.project, j.flags["solvent_mask"]) if isinstance(j.flags.get("solvent_mask"), str) else None
                comps.update(map_noise(*m["half_maps"], maskp, (float(j.flags.get("particle_diameter", diameter)) / 2 / apix) if apix else None))
            comps["user"] = user_noise(name) if user_noise(name) is not None else (user_noise(pps[0].name) if pps else None)
            a = ann.get(name, {})
            if a.get("override_res") == a.get("override_res") and a.get("override_res") not in (None, ""):
                res = float(a["override_res"])
            n = m.get("n_particles")
            outcomes.append(dict(
                outcome=name, project=project_index, job=name, select_job=None, kind="refine",
                number=j.number, res_A=res, gold=int(bool(m.get("gold", True))), n=n, n_in=n,
                sel_frac=1.0, n_sel_classes=1, selected_classes="", apix=apix, box_px=m.get("box_px"),
                noise_comps=comps, bfactor=bfac, hours=j.hours + sum(p.hours for p in pps if not np.isnan(p.hours)),
                lineage_hours=lineage_hours(name)))

    out = pd.DataFrame(outcomes)
    if out.empty:
        return out, pd.DataFrame(), dict(d_ref=d_ref, n0=n0, diameter=diameter)
    for c in ("spectral", "pmax", "angular_accuracy", "halfmap", "solvent", "mask_artefact", "user"):
        out["noise_" + c] = [nc.get(c) for nc in out["noise_comps"]]
    out["noise"] = [combine_noise(nc, wn) for nc in out["noise_comps"]]
    out = out.drop(columns=["noise_comps"])
    terms = [free_energy_terms(r.res_A, r.n, n0, r.noise, r.lineage_hours, cfg, d_ref) for r in out.itertuples()]
    for k in ("E_res", "E_particles", "E_noise", "E_compute"):
        out[k] = [t[k] for t in terms]
    out["F"] = [F_from_terms(t, cfg) for t in terms]
    out["lnfrac"] = np.log(out["n"].astype(float) / float(n0)) if n0 else np.nan
    out["particle"] = part.get("name", f"project{project_index}")

    # ------------------------------------------------------------------ transitions
    oidx = {r.outcome: r for r in out.itertuples()}
    by_job = {}
    for r in out.itertuples():
        by_job.setdefault(r.job, []).append(r)

    def pre_outcome(job: Job):
        """Walk up the particle lineage to the nearest measured state; collect intermediate job types."""
        mods: Dict[str, int] = {}
        cur = job.parent
        seen = set()
        while cur and cur in jobs and cur not in seen:
            seen.add(cur)
            pj = jobs[cur]
            k = job_kind(pj)
            if k == "select" and pj.parent and pj.parent in jobs and job_kind(jobs[pj.parent]) == "classify":
                oid = f"{pj.parent}|{pj.name}"
                if oid in oidx:
                    return oidx[oid], mods
            if k in ("refine", "classify") and cur in by_job:
                return by_job[cur][0], mods
            if k not in ("postprocess",):
                mods[pj.jtype] = mods.get(pj.jtype, 0) + 1
            cur = pj.parent
        return None, mods

    trans = []
    depth_cache: Dict[str, tuple] = {}

    def stage(oid):
        if oid in depth_cache:
            return depth_cache[oid]
        r = oidx[oid]
        pre, _ = pre_outcome(jobs[r.job])
        if pre is None:
            val = (0, 0, 0)
        else:
            d, nc, nr = stage(pre.outcome)
            val = (d + 1, nc + (pre.kind == "classify"), nr + (pre.kind == "refine"))
        depth_cache[oid] = val
        return val

    sorder = sym_order(sym)
    for r in out.itertuples():
        j = jobs[r.job]
        pre, mods = pre_outcome(j)
        apix = r.apix or (pre.apix if pre is not None else None) or 1.0
        defaults = {k: v for k, v in FLAG_DEFAULTS.items()
                    if k in ("tau2_fudge", "K", "healpix_order", "offset_range", "offset_step", "sigma_ang", "pad", "oversampling")}
        if r.kind == "refine":
            defaults["auto_local_healpix_order"] = FLAG_DEFAULTS["auto_local_healpix_order"]
        feats = encode_flags({**defaults, **j.flags}, apix, diameter)
        mjob = None
        for mname in j.inputs.get("mask", []):
            if mname in jobs:
                mjob = jobs[mname]
        feats.update(mask_features(mjob, apix))
        if pre is not None:  # jobs run between two measured states (CtfRefine, re-extraction, ...)
            for mk, mv in mods.items():
                feats[f"mod.{mk}"] = float(mv)
        if r.kind == "classify":
            feats["sel.frac"] = float(r.sel_frac) if r.sel_frac == r.sel_frac else 1.0
            feats["sel.n_classes"] = float(r.n_sel_classes)
        else:
            feats["sel.frac"], feats["sel.n_classes"] = 1.0, 1.0
        d, nc, nr = stage(r.outcome)
        if pre is None:
            ini = j.flags.get("ini_high")
            try:
                ini = float(ini) if ini is not None and float(ini) > 0 else 60.0
            except (TypeError, ValueError):
                ini = 60.0
            p_eres, p_noise, p_gold = math.log(ini / d_ref), 1.0, 0
            p_lnfrac = math.log(r.n_in / n0) if (r.n_in and n0) else 0.0
            p_F = F_from_state(p_eres, p_lnfrac, p_noise, cfg)
            pre_id = None
        else:
            p_eres, p_noise, p_gold = pre.E_res, (pre.noise if pre.noise == pre.noise else 1.0), pre.gold
            p_lnfrac = math.log(r.n_in / n0) if (r.n_in and n0) else pre.lnfrac
            p_F, pre_id = pre.F, pre.outcome
        rec = dict(
            outcome=r.outcome, pre_outcome=pre_id, project=project_index, particle=r.particle, job=r.job,
            number=r.number, kind=r.kind, hours=r.hours,
            **{"pre.E_res": p_eres, "pre.lnfrac": p_lnfrac, "pre.noise": p_noise, "pre.gold": float(p_gold),
               "pre.depth": float(d), "pre.n_class_rounds": float(nc), "pre.n_refine_rounds": float(nr)},
            **{"ctx.apix": float(apix), "ctx.box_A": float((r.box_px or 0) * apix), "ctx.sym_order": float(sorder),
               "ctx.mass_kda": float(part.get("mass_kda") or np.nan), "ctx.diameter_A": float(diameter),
               "ctx.n_subunits": float(part.get("n_subunits") or 1), "ctx.flexibility": float(part.get("flexibility") or np.nan)},
            **{"kind.classify": float(r.kind == "classify"), "kind.refine": float(r.kind == "refine")},
            **{f"a.{k}": v for k, v in feats.items()},
            **{"post.E_res": r.E_res, "post.lnfrac": r.lnfrac, "post.noise": (r.noise if r.noise == r.noise else 1.0),
               "pre.F": p_F, "post.F": r.F})
        rec["d.E_res"] = rec["post.E_res"] - p_eres
        rec["d.lnfrac"] = rec["post.lnfrac"] - p_lnfrac
        rec["d.noise"] = rec["post.noise"] - p_noise
        trans.append(rec)
    tr = pd.DataFrame(trans)
    switches = {"mask.has_mask"}
    for j in ok.values():
        if job_kind(j) in ("classify", "refine"):
            switches |= {k for k, v in j.flags.items() if v is True}
    meta = dict(d_ref=d_ref, n0=n0, diameter=diameter, sym=sym, particle=part, switches=switches)
    return out, tr, meta
