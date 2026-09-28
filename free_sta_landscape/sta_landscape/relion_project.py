from __future__ import annotations

import datetime as _dt
import glob
import os
import re
import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .starfile_io import read_star, particle_count, optimisation_set_particles

JOB_RE = re.compile(r"([A-Za-z0-9_]+/job\d{3,})")

# Flags that never change the result (compute / IO / bookkeeping) -> not landscape coordinates
NON_SCIENTIFIC_FLAGS = {
    "o", "i", "ios", "ref", "solvent_mask", "solvent_mask2", "j", "pool", "gpu", "cpu",
    "dont_combine_weights_via_disc", "pipeline_control", "scratch_dir", "preread_images",
    "free_gpu_memory", "dont_check_norm", "continue", "keep_free_scratch", "reuse_scratch",
    "random_seed", "fn_parts", "mask", "t", "tomograms", "traj", "mot", "trajectories",
    "motion", "fn_out", "angpix", "log", "only_do_unfinished", "keep_scratch",
    "onthefly_shifts", "maxsig_gpu", "verb", "fn_ref", "fn_mask", "ref_angpix",
    "blush_skip_spectral_trailing", "skip_gridding_gpu", "tomo", "particles",
}


@dataclass
class Job:
    name: str                      # e.g. 'Class3D/job012'
    jtype: str                     # 'Class3D'
    number: int
    project: str = ""
    alias: str = ""
    status: str = "unknown"        # succeeded | failed | aborted | running | unknown
    program: str = ""
    flags: Dict[str, object] = field(default_factory=dict)
    joboptions: Dict[str, object] = field(default_factory=dict)
    commands: List[str] = field(default_factory=list)
    inputs: Dict[str, List[str]] = field(default_factory=dict)   # role -> [job names]
    parent: Optional[str] = None   # primary (particle / map) lineage parent
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    metrics: Dict[str, object] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)

    @property
    def hours(self):
        if self.t_start and self.t_end and self.t_end >= self.t_start:
            return (self.t_end - self.t_start) / 3600.0
        return np.nan


# command lines
def _clean_tokens(line: str) -> List[str]:
    line = line.replace("`which ", "").replace("`", " ")
    try:
        toks = shlex.split(line)
    except ValueError:
        toks = line.split()
    return toks


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_flags(tokens: List[str]) -> Dict[str, object]:
    flags: Dict[str, object] = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--") and len(t) > 2 and not _is_number(t):
            key = t[2:]
            vals = []
            j = i + 1
            while j < len(tokens) and not (tokens[j].startswith("--") and not _is_number(tokens[j])):
                vals.append(tokens[j])
                j += 1
            if not vals:
                flags[key] = True
            else:
                v = " ".join(vals)
                flags[key] = float(v) if _is_number(v) else v
            i = j
        else:
            i += 1
    return flags


def parse_note(path: str):
    """Return (commands, start_times) from a RELION note.txt."""
    cmds, starts = [], []
    if not os.path.isfile(path):
        return cmds, starts
    with open(path, errors="replace") as fh:
        for line in fh:
            s = line.strip()
            m = re.search(r"Executing new job on (.+)$", s)
            if m:
                txt = m.group(1).strip()
                for fmt in ("%a %b %d %H:%M:%S %Y", "%a %b  %d %H:%M:%S %Y"):
                    try:
                        starts.append(_dt.datetime.strptime(txt, fmt).timestamp())
                        break
                    except ValueError:
                        continue
                continue
            if "relion_" in s and not s.startswith("++++"):
                cmds.append(s)
    return cmds, starts


def _main_command(cmds: List[str]):
    """Pick the main program + merged flags (later '--continue' commands override)."""
    program, flags = "", {}
    for c in cmds:
        toks = _clean_tokens(c)
        prog = next((t for t in toks if re.match(r"^relion_\w+", os.path.basename(t))), None)
        if prog is None:
            continue
        prog = os.path.basename(prog).replace("_mpi", "")
        f = parse_flags(toks)
        if not program:
            program, flags = prog, f
        elif prog == program:
            if "continue" in f:  # continuation: only override what is given
                for k, v in f.items():
                    if k not in ("continue", "o"):
                        flags[k] = v
    return program, flags


def read_joboptions(path: str) -> Dict[str, object]:
    d = read_star(path)
    out = {}
    tab = d.get("joboptions_values")
    if isinstance(tab, pd.DataFrame) and "rlnJobOptionVariable" in tab.columns:
        for k, v in zip(tab["rlnJobOptionVariable"], tab["rlnJobOptionValue"]):
            out[str(k)] = v
    return out


# scanning
def _status(jobdir: str) -> str:
    if os.path.exists(os.path.join(jobdir, "RELION_JOB_EXIT_SUCCESS")):
        return "succeeded"
    if os.path.exists(os.path.join(jobdir, "RELION_JOB_EXIT_FAILURE")):
        return "failed"
    if os.path.exists(os.path.join(jobdir, "RELION_JOB_EXIT_ABORTED")):
        return "aborted"
    return "unknown"


def _end_time(jobdir: str) -> Optional[float]:
    for f in ("RELION_JOB_EXIT_SUCCESS", "RELION_JOB_EXIT_FAILURE", "RELION_JOB_EXIT_ABORTED"):
        p = os.path.join(jobdir, f)
        if os.path.exists(p):
            return os.path.getmtime(p)
    try:
        return max(os.path.getmtime(p) for p in glob.glob(os.path.join(jobdir, "*")))
    except ValueError:
        return None


def _read_pipeline(project: str):
    p = os.path.join(project, "default_pipeline.star")
    aliases, status, edges = {}, {}, []
    if not os.path.isfile(p):
        return aliases, status, edges
    d = read_star(p)
    proc = d.get("pipeline_processes")
    if isinstance(proc, pd.DataFrame) and "rlnPipeLineProcessName" in proc.columns:
        for _, r in proc.iterrows():
            name = str(r["rlnPipeLineProcessName"]).rstrip("/")
            al = r.get("rlnPipeLineProcessAlias", "None")
            aliases[name] = "" if str(al) in ("None", "nan") else str(al).rstrip("/")
            st = r.get("rlnPipeLineProcessStatusLabel", r.get("rlnPipeLineProcessStatus", ""))
            status[name] = str(st)
    nodes = d.get("pipeline_nodes")
    ntype = {}
    if isinstance(nodes, pd.DataFrame) and "rlnPipeLineNodeName" in nodes.columns:
        col = "rlnPipeLineNodeTypeLabel" if "rlnPipeLineNodeTypeLabel" in nodes.columns else "rlnPipeLineNodeType"
        ntype = dict(zip(nodes["rlnPipeLineNodeName"].astype(str), nodes[col].astype(str)))
    ein = d.get("pipeline_input_edges")
    if isinstance(ein, pd.DataFrame) and len(ein):
        for fr, pr in zip(ein["rlnPipeLineEdgeFromNode"].astype(str), ein["rlnPipeLineEdgeProcess"].astype(str)):
            edges.append((fr, pr.rstrip("/"), ntype.get(fr, "")))
    return aliases, status, edges


def _role_of(flag: str, program: str) -> str:
    if flag in ("solvent_mask", "solvent_mask2", "mask"):
        return "mask"
    if flag in ("ref",):
        return "reference"
    return "primary"


def scan_project(project: str) -> Dict[str, Job]:
    project = os.path.abspath(project)
    aliases, pstatus, edges = _read_pipeline(project)
    jobs: Dict[str, Job] = {}
    for jd in sorted(glob.glob(os.path.join(project, "*", "job[0-9][0-9][0-9]*"))):
        if not os.path.isdir(jd):
            continue
        name = os.path.relpath(jd, project).replace(os.sep, "/")
        jtype, jnum = name.split("/")
        job = Job(name=name, jtype=jtype, number=int(re.sub(r"\D", "", jnum)), project=project)
        job.alias = aliases.get(name, "")
        cmds, starts = parse_note(os.path.join(jd, "note.txt"))
        job.commands = cmds
        job.program, job.flags = _main_command(cmds)
        job.joboptions = read_joboptions(os.path.join(jd, "job.star"))
        job.status = _status(jd)
        if job.status == "unknown" and name in pstatus:
            s = pstatus[name].lower()
            job.status = {"succeeded": "succeeded", "2": "succeeded", "failed": "failed",
                          "3": "failed", "aborted": "aborted", "4": "aborted",
                          "running": "running", "0": "running"}.get(s, "unknown")
        note = os.path.join(jd, "note.txt")
        job.t_start = starts[0] if starts else (os.path.getmtime(note) if os.path.exists(note) else None)
        job.t_end = _end_time(jd)
        # inputs from the command line (typed by flag)
        for c in cmds:
            for flag, val in parse_flags(_clean_tokens(c)).items():
                if not isinstance(val, str) or flag in ("o", "pipeline_control"):
                    continue
                for m in JOB_RE.findall(val):
                    if m != name:
                        job.inputs.setdefault(_role_of(flag, job.program), [])
                        if m not in job.inputs[_role_of(flag, job.program)]:
                            job.inputs[_role_of(flag, job.program)].append(m)
        jobs[name] = job

    # fall back to pipeline edges when note.txt did not reveal inputs
    for fr, proc, ntype in edges:
        if proc not in jobs:
            continue
        m = JOB_RE.search(fr)
        if not m or m.group(1) == proc:
            continue
        role = "mask" if "mask" in ntype.lower() else ("reference" if "densitymap" in ntype.lower() and jobs[proc].jtype in ("Class3D", "Refine3D") else "primary")
        lst = jobs[proc].inputs.setdefault(role, [])
        if m.group(1) not in lst and not any(m.group(1) in v for v in jobs[proc].inputs.values()):
            lst.append(m.group(1))

    for j in jobs.values():
        prim = [p for p in j.inputs.get("primary", []) if p in jobs]
        if prim:
            # the most recent upstream job carrying particles/maps
            j.parent = sorted(prim, key=lambda n: jobs[n].number)[-1]
            jobs[j.parent].children.append(j.name)

    for j in jobs.values():
        extract_metrics(j, jobs)
    return jobs


# metrics
def job_kind(job: Job) -> str:
    """classify | refine | postprocess | select | mask | other"""
    t = job.jtype.lower()
    if t == "class3d" or (job.program == "relion_refine" and t == "external" and "K" in job.flags and "auto_refine" not in job.flags):
        return "classify"
    if t == "refine3d" or (job.program == "relion_refine" and "auto_refine" in job.flags):
        return "refine"
    if t == "postprocess":
        return "postprocess"
    if t in ("select", "subset"):
        return "select"
    if t == "maskcreate":
        return "mask"
    return "other"


def _last_iter_file(jobdir: str, pattern: str) -> Optional[str]:
    files = glob.glob(os.path.join(jobdir, pattern))
    if not files:
        return None

    def it(f):
        m = re.search(r"_it(\d+)", f)
        return int(m.group(1)) if m else -1
    return max(files, key=it)


def _spectral(model: dict, cls: int = 1) -> Optional[pd.DataFrame]:
    t = model.get(f"model_class_{cls}")
    if isinstance(t, pd.DataFrame) and "rlnResolution" in t.columns:
        return t
    return None


def _model_general(model: dict) -> dict:
    g = model.get("model_general")
    return g if isinstance(g, dict) else {}


def _particles_file(jobdir: str, candidates) -> Optional[str]:
    for c in candidates:
        p = os.path.join(jobdir, c)
        if os.path.isfile(p):
            return p
    return None


def extract_metrics(job: Job, jobs: Dict[str, Job]):
    jd = os.path.join(job.project, job.name)
    m = job.metrics
    kind = job_kind(job)
    if kind == "classify":
        mf = _last_iter_file(jd, "run_it*_model.star")
        if mf:
            model = read_star(mf)
            g = _model_general(model)
            m["apix"] = g.get("rlnPixelSize")
            m["box_px"] = g.get("rlnOriginalImageSize")
            m["current_res"] = g.get("rlnCurrentResolution")
            m["pmax"] = g.get("rlnAveragePmax")
            m["loglik"] = g.get("rlnLogLikelihood")
            mc = model.get("model_classes")
            if isinstance(mc, pd.DataFrame) and len(mc):
                m["class_dist"] = [float(x) for x in mc.get("rlnClassDistribution", pd.Series(dtype=float))]
                m["class_res"] = [float(x) for x in mc.get("rlnEstimatedResolution", pd.Series(dtype=float))]
                acc = mc.get("rlnAccuracyRotations")
                m["class_acc_rot"] = [float(x) for x in acc] if acc is not None else None
            m["class_ssnr"] = {}
            for k in range(1, len(m.get("class_dist", [])) + 1):
                sp = _spectral(model, k)
                if sp is not None and "rlnSsnrMap" in sp.columns:
                    m["class_ssnr"][k] = (sp["rlnResolution"].to_numpy(float), sp["rlnSsnrMap"].to_numpy(float))
            it = re.search(r"_it(\d+)", mf)
            m["last_iter"] = int(it.group(1)) if it else None
            df = mf.replace("_model.star", "_data.star")
            if os.path.isfile(df):
                d = read_star(df, blocks=("particles", "images", ""), columns={"particles": ["rlnClassNumber"], "images": ["rlnClassNumber"], "": ["rlnClassNumber"]})
                t = next((v for v in d.values() if isinstance(v, pd.DataFrame)), None)
                if t is not None and "rlnClassNumber" in t.columns:
                    vc = t["rlnClassNumber"].astype(int).value_counts()
                    m["n_particles"] = int(len(t))
                    m["class_counts"] = {int(k): int(v) for k, v in vc.items()}
            if "n_particles" not in m and m.get("class_dist"):
                pass
    elif kind == "refine":
        mf = os.path.join(jd, "run_model.star")
        final = os.path.isfile(mf)
        if not final:
            mf = _last_iter_file(jd, "run_it*_half1_model.star")
        if mf:
            model = read_star(mf)
            g = _model_general(model)
            m["apix"] = g.get("rlnPixelSize")
            m["box_px"] = g.get("rlnOriginalImageSize")
            m["res"] = g.get("rlnCurrentResolution")
            m["pmax"] = g.get("rlnAveragePmax")
            m["gold"] = bool(final)
            mc = model.get("model_classes")
            if isinstance(mc, pd.DataFrame) and "rlnAccuracyRotations" in mc.columns:
                m["acc_rot"] = float(mc["rlnAccuracyRotations"].iloc[0])
            sp = _spectral(model, 1)
            if sp is not None:
                if "rlnSsnrMap" in sp.columns:
                    m["ssnr"] = (sp["rlnResolution"].to_numpy(float), sp["rlnSsnrMap"].to_numpy(float))
                if "rlnGoldStandardFsc" in sp.columns:
                    m["fsc"] = (sp["rlnResolution"].to_numpy(float), sp["rlnGoldStandardFsc"].to_numpy(float))
        pf = _particles_file(jd, ("run_data.star",))
        if pf:
            m["n_particles"] = particle_count(pf)
        h1 = os.path.join(jd, "run_half1_class001_unfil.mrc")
        h2 = os.path.join(jd, "run_half2_class001_unfil.mrc")
        if os.path.isfile(h1) and os.path.isfile(h2):
            m["half_maps"] = (h1, h2)
    elif job.jtype == "PostProcess":
        pp = read_star(os.path.join(jd, "postprocess.star"))
        g = pp.get("general") if isinstance(pp.get("general"), dict) else {}
        m["res"] = g.get("rlnFinalResolution")
        m["bfactor"] = g.get("rlnBfactorUsedForSharpening")
        fsc = pp.get("fsc")
        if isinstance(fsc, pd.DataFrame) and "rlnResolution" in fsc.columns:
            m["fsc_table"] = fsc
        mask = job.flags.get("mask")
        if isinstance(mask, str):
            m["mask_path"] = os.path.join(job.project, mask)
    elif job.jtype == "Select" or job.jtype == "Subset":
        pf = _particles_file(jd, ("particles.star", "selected_particles.star"))
        if pf is None:
            os_ = _particles_file(jd, ("optimisation_set.star",))
            if os_:
                p = optimisation_set_particles(os_)
                pf = os.path.join(job.project, p) if p else None
        if pf:
            m["n_particles"] = particle_count(pf)
        bs = os.path.join(jd, "backup_selection.star")
        if os.path.isfile(bs):
            d = read_star(bs)
            t = next((v for v in d.values() if isinstance(v, pd.DataFrame) and "rlnSelected" in v.columns), None)
            if t is not None:
                m["selected_classes"] = [i + 1 for i, s in enumerate(t["rlnSelected"].astype(int)) if s == 1]
        if "selected_classes" not in m:
            # relion_star_handler --select rlnClassNumber --minval 2 --maxval 2 style selections
            f = job.flags
            if str(f.get("select", "")) == "rlnClassNumber" and "minval" in f and "maxval" in f:
                try:
                    m["selected_classes"] = list(range(int(float(f["minval"])), int(float(f["maxval"])) + 1))
                except (TypeError, ValueError):
                    pass
    else:
        # generic: try to count particles in common outputs
        for c in ("particles.star", "run_data.star", "optimisation_set.star"):
            p = os.path.join(jd, c)
            if os.path.isfile(p):
                if c == "optimisation_set.star":
                    q = optimisation_set_particles(p)
                    p = os.path.join(job.project, q) if q else None
                n = particle_count(p) if p else None
                if n:
                    m["n_particles"] = n
                    break


def jobs_table(jobs: Dict[str, Job]) -> pd.DataFrame:
    rows = []
    for j in sorted(jobs.values(), key=lambda x: (x.project, x.number)):
        rows.append({
            "project": j.project, "job": j.name, "type": j.jtype, "number": j.number, "alias": j.alias,
            "status": j.status, "program": j.program, "parent": j.parent,
            "mask": ";".join(j.inputs.get("mask", [])), "reference": ";".join(j.inputs.get("reference", [])),
            "hours": j.hours,
            "n_particles": j.metrics.get("n_particles"), "res": j.metrics.get("res", j.metrics.get("current_res")),
            "flags": " ".join(f"--{k} {v}" if v is not True else f"--{k}" for k, v in j.flags.items()),
        })
    return pd.DataFrame(rows)
