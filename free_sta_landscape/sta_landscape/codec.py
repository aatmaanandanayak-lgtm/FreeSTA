from __future__ import annotations

import math
import re
from typing import Dict, Optional

import numpy as np

from .relion_project import NON_SCIENTIFIC_FLAGS, Job

GUI_NAMES = {
    "tau2_fudge": "Regularisation parameter T", "K": "Number of classes", "iter": "Number of iterations",
    "particle_diameter": "Mask diameter (A)", "ini_high": "Initial low-pass filter (A)",
    "strict_highres_exp": "Limit resolution E-step to (A)", "healpix_order": "Angular sampling interval",
    "offset_range": "Offset search range (pix)", "offset_step": "Offset search step (pix)",
    "sigma_ang": "Local angular search range (sigma, deg)", "zero_mask": "Mask individual particles with zeros?",
    "flatten_solvent": "Flatten solvent", "ctf": "Do CTF-correction?",
    "ctf_intact_first_peak": "Ignore CTFs until first peak?", "fast_subsets": "Use fast subsets?",
    "skip_align": "Perform image alignment? (skip)", "blush": "Use Blush regularisation?",
    "allow_coarser_sampling": "Allow coarser sampling?", "auto_local_healpix_order": "Local searches from auto-sampling",
    "solvent_correct_fsc": "Use solvent-flattened FSCs?", "relax_sym": "Relax symmetry", "sym": "Symmetry",
    "pad": "Padding factor (1 = skip padding)", "skip_gridding": "Skip gridding?",
    "firstiter_cc": "Ref. map NOT on absolute greyscale (first-iteration CC)",
    "auto_ignore_angles": "Finer angular sampling faster", "auto_resol_angles": "Finer angular sampling faster",
    "maxsig": "Maximum number of significant poses", "norm": "Normalisation correction",
    "scale": "Intensity-scale correction", "oversampling": "Oversampling", "trust_ref_size": "Trust reference size",
    "low_resol_join_halves": "Join half-maps below (A)", "sigma_tilt": "Prior width on tilt",
    "sigma_psi": "Prior width on psi", "sigma_rot": "Prior width on rot", "helix": "Helical reconstruction",
    "iter_perturb": None,
}

DERIVED = {
    "particle_diameter_rel": "particle_diameter", "offset_range_A": "offset_range",
    "offset_step_A": "offset_step", "highres_limit_rel": "strict_highres_exp",
    "healpix_arc_A": "healpix_order", "auto_local_arc_A": "auto_local_healpix_order",
    "log_tau2_fudge": "tau2_fudge",
}

# Default values 
FLAG_DEFAULTS = {
    "tau2_fudge": 1.0, "K": 1.0, "iter": 50.0, "healpix_order": 2.0, "offset_range": 6.0,
    "offset_step": 2.0, "oversampling": 1.0, "pad": 2.0, "ini_high": -1.0, "maxsig": -1.0,
    "sigma_ang": 0.0, "sigma_tilt": 0.0, "sigma_psi": 0.0, "sigma_rot": 0.0,
    "auto_local_healpix_order": 4.0, "low_resol_join_halves": 40.0,
}


def healpix_deg(order: float) -> float:
    return 30.0 / (2.0 ** float(order))


def sym_order(sym: Optional[str]) -> int:
    if not sym or not isinstance(sym, str):
        return 1
    s = sym.strip().upper()
    m = re.match(r"^([CD])(\d+)", s)
    if m:
        n = int(m.group(2))
        return n if m.group(1) == "C" else 2 * n
    return {"T": 12, "O": 24, "I": 60}.get(s[:1], 1)


def _is_path(v) -> bool:
    return isinstance(v, str) and ("/" in v or v.endswith((".star", ".mrc", ".mrcs", ".txt")))


def encode_flags(flags: Dict[str, object], apix: float, diameter_A: float) -> Dict[str, float]:
    """relion_refine flags -> landscape coordinates."""
    f: Dict[str, float] = {}
    nyq = 2.0 * apix if apix else None
    for k, v in flags.items():
        if k in NON_SCIENTIFIC_FLAGS or _is_path(v):
            continue
        if k in ("sym", "relax_sym"):
            continue  # context
        if v is True:
            f[k] = 1.0
            continue
        if isinstance(v, str):
            try:
                v = float(v)
            except ValueError:
                continue  # other free-text options are ignored
        v = float(v)
        if k == "particle_diameter" and diameter_A:
            f["particle_diameter_rel"] = v / diameter_A
        elif k in ("offset_range", "offset_step") and apix:
            f[k + "_A"] = v * apix
        elif k == "strict_highres_exp":
            f["highres_limit_rel"] = min(1.0, nyq / v) if (v > 0 and nyq) else 1.0
        elif k in ("healpix_order", "auto_local_healpix_order") and diameter_A:
            name = "healpix_arc_A" if k == "healpix_order" else "auto_local_arc_A"
            f[name] = math.radians(healpix_deg(v)) * diameter_A / 2.0
        elif k == "tau2_fudge" and v > 0:
            f["log_tau2_fudge"] = math.log(v)
        else:
            f[k] = v
    # absent flags that have a meaningful 'off' value
    if "highres_limit_rel" not in f:
        f["highres_limit_rel"] = 1.0
    return f


def decode_features(feat: Dict[str, float], apix: float, diameter_A: float,
                    bool_cols=(), int_cols=()) -> Dict[str, object]:
    """landscape coordinates -> relion_refine flags (for a given particle / pixel size)."""
    flags: Dict[str, object] = {}
    nyq = 2.0 * apix
    for k, v in feat.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if k.startswith(("mask.", "sel.", "mod.", "pre.", "ctx.", "kind.")):
            continue
        if k == "particle_diameter_rel":
            flags["particle_diameter"] = int(round(v * diameter_A))
        elif k in ("offset_range_A", "offset_step_A"):
            flags[k[:-2]] = round(max(0.5, v / apix), 1)
        elif k == "highres_limit_rel":
            if v < 0.95:  # within 5% of Nyquist = effectively no limit
                flags["strict_highres_exp"] = round(nyq / max(v, 1e-3), 1)
        elif k in ("healpix_arc_A", "auto_local_arc_A"):
            deg = math.degrees(v / (diameter_A / 2.0))
            order = int(np.clip(round(math.log2(30.0 / max(deg, 1e-3))), 0, 8))
            flags["healpix_order" if k == "healpix_arc_A" else "auto_local_healpix_order"] = order
        elif k == "log_tau2_fudge":
            flags["tau2_fudge"] = round(float(math.exp(v)), 2)
        elif k in bool_cols:
            if v >= 0.5:
                flags[k] = True
        elif k in int_cols:
            flags[k] = int(round(v))
        else:
            flags[k] = round(float(v), 3)
    return flags


def mask_features(mask_job: Optional[Job], apix: float) -> Dict[str, float]:
    if mask_job is None:
        return {"mask.has_mask": 0.0}
    fl = mask_job.flags
    a = float(fl.get("angpix", apix) or apix or 1.0)
    out = {"mask.has_mask": 1.0}
    for k, name, scale in (("lowpass", "mask.lowpass_A", 1.0), ("ini_threshold", "mask.ini_threshold", None),
                           ("extend_inimask", "mask.extend_A", a), ("width_soft_edge", "mask.soft_edge_A", a)):
        if k in fl:
            try:
                out[name] = float(fl[k]) * (scale if scale else 1.0)
            except (TypeError, ValueError):
                pass
    return out
