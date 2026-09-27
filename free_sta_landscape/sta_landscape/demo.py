"""Generate a synthetic RELION-5-style subtomogram-averaging project with a hidden, known
landscape, explored by a simulated user doing one-factor-at-a-time tuning.
Used for testing and as a worked example.  No real data is involved."""
from __future__ import annotations

import datetime as dt
import math
import os

import numpy as np
import pandas as pd

from .starfile_io import write_star

APIX, BOX, DIAM, N0 = 3.4, 96, 180.0, 16000
NYQ = 2 * APIX


class Sim:
    def __init__(self, root, seed=1):
        self.root = root
        self.rng = np.random.default_rng(seed)
        self.num = 0
        self.t = dt.datetime(2026, 6, 1, 9, 0, 0).timestamp()
        os.makedirs(root, exist_ok=True)

    # ------------------------------------------------------------------ helpers
    def newjob(self, jtype, cmd, hours, status="succeeded"):
        self.num += 1
        name = f"{jtype}/job{self.num:03d}"
        d = os.path.join(self.root, name)
        os.makedirs(d, exist_ok=True)
        cmd = cmd.replace("JOB", name)
        start = self.t
        with open(os.path.join(d, "note.txt"), "w") as fh:
            fh.write(f" ++++ Executing new job on {dt.datetime.fromtimestamp(start).strftime('%a %b %d %H:%M:%S %Y')}\n")
            fh.write(" ++++ with the following command(s): \n")
            fh.write(cmd + "\n ++++ \n")
        self.t += hours * 3600
        flag = {"succeeded": "RELION_JOB_EXIT_SUCCESS", "failed": "RELION_JOB_EXIT_FAILURE"}[status]
        p = os.path.join(d, flag)
        open(p, "w").close()
        os.utime(p, (self.t, self.t))
        self.t += 600
        return name, d

    def particles(self, path, classes):
        df = pd.DataFrame({"rlnTomoName": ["TS_01"] * len(classes), "rlnClassNumber": classes})
        write_star(path, {"optics": {"rlnOpticsGroup": 1}, "particles": df})

    @staticmethod
    def ssnr_table(d, purity, gold=False):
        f = np.linspace(0, 1 / NYQ, BOX // 2 + 1)
        s0 = 30 * purity + 2
        bn = math.log(s0 / 0.333) * d * d
        ssnr = s0 * np.exp(-bn * f ** 2)
        fsc = ssnr / (ssnr + 2)
        t = pd.DataFrame({"rlnSpectralIndex": np.arange(len(f)), "rlnResolution": f,
                          "rlnAngstromResolution": np.where(f > 0, 1 / np.maximum(f, 1e-9), 999.0),
                          "rlnSsnrMap": ssnr})
        if gold:
            t["rlnGoldStandardFsc"] = fsc
        return t

    # ------------------------------------------------------------------ hidden physics
    @staticmethod
    def class_quality(p):
        tau, hl, K, order = p["tau2_fudge"], p["hl"], p["K"], p["healpix_order"]
        q = math.exp(-(math.log(tau / 3.0)) ** 2 / 0.6) * math.exp(-(hl - 0.6) ** 2 / 0.06)
        q *= max(0.3, 1 - 0.12 * abs(K - 4)) * max(0.4, 1 - 0.3 * abs(order - 2))
        q *= 0.85 + 0.15 * p.get("zero_mask", 1)
        if tau > 4 and hl > 0.8:          # interaction: strong regularisation + no E-step limit overfits
            q *= 0.55
        q *= 0.9 + 0.1 * math.exp(-abs(p.get("mask_ext", 10) - 17) / 8)
        return float(np.clip(q, 0.05, 1.0))

    @staticmethod
    def rh(neff, B):
        inv2 = (2.0 / B) * math.log(max(neff, 60) / 50.0)
        return max(1 / math.sqrt(max(inv2, 1e-6)), NYQ * 1.04)

    def refine_B(self, p, mods):
        B = 1100 * (1 + 0.5 * abs(math.log(p.get("tau2_fudge", 1.0)))) * (1 + 0.12 * abs(p.get("auto_local_healpix_order", 4) - 4))
        B *= 0.82 if p.get("blush") else 1.0
        B *= 1 + 0.015 * abs(p.get("mask_soft", 20) - 20)
        B *= 0.85 if mods.get("ctf") else 1.0
        return B

    # ------------------------------------------------------------------ job writers
    def mask(self, ext_px, soft_px, lowpass=20):
        name, d = self.newjob("MaskCreate", f"`which relion_mask_create` --i Import/job001/ref.mrc --o JOB/mask.mrc "
                              f"--lowpass {lowpass} --ini_threshold 0.01 --extend_inimask {ext_px} --width_soft_edge {soft_px} "
                              f"--angpix {APIX} --j 4", 0.05)
        open(os.path.join(d, "mask.mrc"), "w").close()
        return name, dict(mask_ext=ext_px * APIX, mask_soft=soft_px * APIX)

    def classify(self, inp, pset, p, mask):
        flags = (f"--tau2_fudge {p['tau2_fudge']} --K {p['K']} --healpix_order {p['healpix_order']} "
                 + (f"--strict_highres_exp {NYQ / p['hl']:.1f} " if p["hl"] < 0.99 else "")
                 + ("--zero_mask " if p.get("zero_mask", 1) else ""))
        cmd = (f"`which relion_refine_mpi` --o JOB/run --ios {inp} --ref Import/job001/ref.mrc --firstiter_cc "
               f"--trust_ref_size --ini_high 60 --dont_combine_weights_via_disc --pool 10 --pad 2 --ctf --iter 25 "
               f"{flags}--particle_diameter 200 --flatten_solvent --solvent_mask {mask[0]}/mask.mrc --oversampling 1 "
               f"--offset_range 5 --offset_step 2 --sym C1 --norm --scale --j 4 --gpu \"\" --pipeline_control JOB/")
        status = "failed" if self.rng.random() < 0.05 else "succeeded"
        name, d = self.newjob("Class3D", cmd, 2.5 + 0.4 * p["K"] + 0.8 * (p["healpix_order"] - 1), status)
        if status == "failed":
            return name, None
        pp = dict(p, **mask[1])
        q = self.class_quality(pp)
        N, pur = pset["N"], pset["purity"]
        K = p["K"]
        g = max(1, int(round(K * 0.3)))
        good_in, bad_in = N * pur, N * (1 - pur)
        gcnt = good_in * (0.45 + 0.5 * q)
        bcnt = bad_in * (1 - q) * 0.35
        counts = np.zeros(K)
        counts[:g] = (gcnt + bcnt) / g
        rest = N - counts[:g].sum()
        split = self.rng.dirichlet(np.ones(K - g) * 3) if K > g else np.array([])
        counts[g:] = rest * split
        counts = np.round(counts).astype(int)
        counts[-1] += N - counts.sum()
        purity_good = gcnt / (gcnt + bcnt)
        res = []
        for k in range(K):
            if k < g:
                neff = counts[k] * purity_good ** 2
                res.append(self.rh(neff, 1500 * (1.4 - 0.4 * q)) * 0.93 * (1 + 0.03 * self.rng.standard_normal()))
            else:
                res.append(float(self.rng.uniform(22, 40)))
        res = [max(r, NYQ / p["hl"] * 1.0) if p["hl"] < 0.99 else r for r in res]
        order = self.rng.permutation(K)            # RELION class numbers are arbitrary
        counts, res = counts[order], np.array(res)[order]
        good_ids = [int(np.where(order == k)[0][0]) + 1 for k in range(g)]
        it = 25
        mg = {"rlnReferenceDimensionality": 3, "rlnOriginalImageSize": BOX, "rlnCurrentResolution": float(min(res)),
              "rlnCurrentImageSize": BOX, "rlnPixelSize": APIX, "rlnNrClasses": K, "rlnTau2FudgeFactor": p["tau2_fudge"],
              "rlnAveragePmax": float(np.clip(0.15 + 0.55 * q * pur + 0.03 * self.rng.standard_normal(), 0.02, 0.98)),
              "rlnLogLikelihood": -1e8}
        mc = pd.DataFrame({"rlnReferenceImage": [f"{name}/run_it{it:03d}_class{k + 1:03d}.mrc" for k in range(K)],
                           "rlnClassDistribution": counts / N, "rlnAccuracyRotations": [3 + 10 * (1 - q)] * K,
                           "rlnAccuracyTranslationsAngst": [4.0] * K, "rlnEstimatedResolution": res,
                           "rlnOverallFourierCompleteness": [1.0] * K})
        blocks = {"model_general": mg, "model_classes": mc}
        for k in range(K):
            pk = purity_good if (k + 1) in good_ids else 0.2
            blocks[f"model_class_{k + 1}"] = self.ssnr_table(res[k], pk)
        write_star(os.path.join(d, f"run_it{it:03d}_model.star"), blocks)
        cls = np.repeat(np.arange(1, K + 1), counts)
        self.particles(os.path.join(d, f"run_it{it:03d}_data.star"), cls)
        return name, dict(counts=counts, res=res, good_ids=good_ids, purity_good=purity_good, q=q, g=g, pur_in=pur, N=N)

    def select(self, cname, cres, extra=0):
        # user keeps the good classes (+ optionally the next best by resolution)
        order = list(np.argsort(cres["res"]) + 1)
        keep = list(cres["good_ids"])
        for k in order:
            if extra <= 0:
                break
            if k not in keep:
                keep.append(int(k))
                extra -= 1
        name, d = self.newjob("Select", f"relion_display --gui --i {cname}/run_it025_optimiser.star --allow_save "
                              f"--fn_parts JOB/particles.star", 0.1)
        sel = pd.DataFrame({"rlnSelected": [1 if (k + 1) in keep else 0 for k in range(len(cres["counts"]))]})
        write_star(os.path.join(d, "backup_selection.star"), {"": sel})
        cls = np.concatenate([np.full(cres["counts"][k - 1], k) for k in keep])
        self.particles(os.path.join(d, "particles.star"), cls)
        n_keep = len(cls)
        good = sum(cres["counts"][k - 1] * (cres["purity_good"] if k in cres["good_ids"] else 0.25 * cres["pur_in"]) for k in keep)
        return name, dict(N=n_keep, purity=good / n_keep)

    def refine(self, inp, ref, pset, p, mask, mods=None):
        mods = mods or {}
        cmd = (f"`which relion_refine_mpi` --o JOB/run --auto_refine --split_random_halves --ios {inp} --ref {ref} "
               f"--ini_high 40 --dont_combine_weights_via_disc --pool 10 --pad 2 --ctf --particle_diameter 200 "
               f"--flatten_solvent --zero_mask --solvent_mask {mask[0]}/mask.mrc --solvent_correct_fsc --oversampling 1 "
               f"--healpix_order 2 --auto_local_healpix_order {p.get('auto_local_healpix_order', 4)} --offset_range 5 "
               f"--offset_step 2 --sym C1 --low_resol_join_halves 40 --norm --scale "
               + (f"--tau2_fudge {p['tau2_fudge']} " if p.get("tau2_fudge", 1) != 1 else "")
               + ("--blush " if p.get("blush") else "") + "--j 4 --gpu \"\" --pipeline_control JOB/")
        name, d = self.newjob("Refine3D", cmd, 4.0 + 1.5 * (4 - p.get("auto_local_healpix_order", 4) < 0) + (1.0 if p.get("blush") else 0))
        pp = dict(p, **mask[1])
        B = self.refine_B(pp, mods)
        neff = pset["N"] * pset["purity"] ** 2
        dres = self.rh(neff, B) * (1 + 0.02 * self.rng.standard_normal())
        mg = {"rlnReferenceDimensionality": 3, "rlnOriginalImageSize": BOX, "rlnCurrentResolution": dres * 1.12,
              "rlnCurrentImageSize": BOX, "rlnPixelSize": APIX, "rlnNrClasses": 1,
              "rlnAveragePmax": float(np.clip(0.2 + 0.5 * pset["purity"], 0, 1)), "rlnLogLikelihood": -1e8}
        mc = pd.DataFrame({"rlnReferenceImage": [f"{name}/run_class001.mrc"], "rlnClassDistribution": [1.0],
                           "rlnAccuracyRotations": [2 + 8 * (1 - pset["purity"])], "rlnAccuracyTranslationsAngst": [3.0],
                           "rlnEstimatedResolution": [dres * 1.12]})
        write_star(os.path.join(d, "run_model.star"), {"model_general": mg, "model_classes": mc,
                                                        "model_class_1": self.ssnr_table(dres * 1.12, pset["purity"], gold=True)})
        self.particles(os.path.join(d, "run_data.star"), np.ones(pset["N"], int))
        # post-processing
        pname, pd_ = self.newjob("PostProcess", f"`which relion_postprocess` --mask {mask[0]}/mask.mrc --i {name}/run_half1_class001_unfil.mrc "
                                 f"--o JOB/postprocess --angpix {APIX} --auto_bfac --autob_lowres 10", 0.05)
        f = np.linspace(0, 1 / NYQ, BOX // 2 + 1)
        fsc = 1 / (1 + np.exp((f - 1 / dres) * 60 * dres))
        art = 0.02 + 0.25 * math.exp(-pp["mask_soft"] / 8)
        pr = np.where(f > 1 / (dres * 1.4), art * (1 + 0.2 * self.rng.standard_normal(len(f))), fsc)
        write_star(os.path.join(pd_, "postprocess.star"), {
            "general": {"rlnFinalResolution": dres, "rlnBfactorUsedForSharpening": -B / 2, "rlnRandomiseFrom": dres * 1.4},
            "fsc": pd.DataFrame({"rlnSpectralIndex": np.arange(len(f)), "rlnResolution": f,
                                 "rlnAngstromResolution": np.where(f > 0, 1 / np.maximum(f, 1e-9), 999.0),
                                 "rlnFourierShellCorrelationCorrected": fsc,
                                 "rlnFourierShellCorrelationUnmaskedMaps": fsc * 0.9,
                                 "rlnFourierShellCorrelationMaskedMaps": fsc,
                                 "rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps": pr})})
        return name, dict(N=pset["N"], purity=pset["purity"], res=dres)

    def ctfrefine(self, inp):
        name, d = self.newjob("CtfRefineTomo", f"relion_tomo_refine_ctf --i {inp} --o JOB/ --do_defocus --j 8", 1.5)
        open(os.path.join(d, "optimisation_set.star"), "w").close()
        return name


def true_F(res, n, noise=0.3):
    return math.log(res / NYQ) - 0.3 * math.log(n / N0) + 0.5 * noise


def make_demo_project(root, seed=1):
    """Simulated user: greedy one-factor-at-a-time search in two rounds of classification + refinement."""
    s = Sim(root, seed)
    os.makedirs(os.path.join(root, "Import", "job001"), exist_ok=True)
    s.num = 1
    open(os.path.join(root, "Import", "job001", "ref.mrc"), "w").close()
    ext, d = s.newjob("PseudoSubtomo", f"relion_tomo_subtomo --i Import/job001/optimisation_set.star --o JOB/ --b {BOX} --crop {BOX} --bin 2 --j 8", 3.0)
    s.particles(os.path.join(d, "particles.star"), np.ones(N0, int))
    pset0 = dict(N=N0, purity=0.4)
    maskA = s.mask(3, 6)       # tight, sharp
    maskB = s.mask(5, 8)
    inp0 = f"{ext}/optimisation_set.star"

    def run_class_sweep(inp, pset, base, sweeps, mask):
        best = None
        for key, values in sweeps:
            results = []
            for v in values:
                p = dict(base, **{key: v})
                cname, cres = s.classify(inp, pset, p, mask)
                if cres is None:
                    continue
                extra = 1 if s.rng.random() < 0.25 else 0
                sname, sset = s.select(cname, cres, extra)
                score = cres["purity_good"] * math.sqrt(sset["N"])
                results.append((score, p, cname, sname, sset, cres))
            if results:
                bestr = max(results, key=lambda r: r[0])
                base = bestr[1]
                best = bestr
        return best

    # ---- round 1 classification: tau, then E-step limit, then K, then sampling, then mask
    base = dict(tau2_fudge=1, hl=1.0, K=4, healpix_order=2, zero_mask=1)
    b1 = run_class_sweep(inp0, pset0, base, [("tau2_fudge", [1, 2, 4, 8]), ("hl", [0.85, 0.6, 0.45]),
                                             ("K", [3, 6]), ("healpix_order", [1, 3])], maskA)
    _, p1, c1, s1, set1, _ = b1
    cmaskB, cresB = s.classify(inp0, pset0, p1, maskB)
    if cresB is not None:
        sB, setB = s.select(cmaskB, cresB)
        if cresB["purity_good"] * math.sqrt(setB["N"]) > b1[0]:
            c1, s1, set1, mask_best = cmaskB, sB, setB, maskB
        else:
            mask_best = maskA
    else:
        mask_best = maskA
    # ---- refinement sweep
    rbase = dict(tau2_fudge=1, auto_local_healpix_order=4, blush=0)
    r_results = []
    for key, vals in (("auto_local_healpix_order", [4, 5]), ("blush", [1]), ("tau2_fudge", [2])):
        for v in vals:
            p = dict(rbase, **{key: v})
            for m in ([maskA, maskB] if key == "auto_local_healpix_order" and v == 4 else [mask_best]):
                rname, rr = s.refine(f"{s1}/optimisation_set.star", f"{c1}/run_it025_class001.mrc", set1, p, m)
                r_results.append((rr["res"], p, rname, rr, m))
        rbase = min(r_results, key=lambda r: r[0])[1]
    _, rp, r1, rr1, rmask = min(r_results, key=lambda r: r[0])
    # ---- round 2: re-classify the refined particles (local), sweep tau and K
    base2 = dict(p1)
    b2 = run_class_sweep(f"{r1}/run_optimisation_set.star", dict(N=rr1["N"], purity=rr1["purity"]), base2,
                         [("tau2_fudge", [2, 6]), ("K", [3, 5])], rmask)
    _, p2, c2, s2, set2, _ = b2
    rname2, rr2 = s.refine(f"{s2}/optimisation_set.star", f"{r1}/run_class001.mrc", set2, rp, rmask)
    ctf = s.ctfrefine(f"{rname2}/run_optimisation_set.star")
    rname3, rr3 = s.refine(f"{ctf}/optimisation_set.star", f"{rname2}/run_class001.mrc", set2, rp, rmask, mods={"ctf": 1})
    # one extra exploratory refinement with blush on the ctf-refined set
    s.refine(f"{ctf}/optimisation_set.star", f"{rname2}/run_class001.mrc", set2, dict(rp, blush=1 - rp.get("blush", 0)), rmask, mods={"ctf": 1})
    return root


DEMO_CONFIG = """projects:
  - dir: {project}
    particle:
      name: demo_complex
      mass_kda: 300
      diameter_A: 180
      symmetry: C1
      n_subunits: 1
      oligomer: 1
      flexibility: 0.3
      n_particles_start: null
    exclude_jobs: []
    annotations: null
output_dir: {out}
free_energy:
  weights: {{resolution: 1.0, particles: 0.1, noise: 0.5, compute: 0.0}}
  particles_preference: more
model:
  bo_replays: 10
"""

DEMO_TARGET = """target:
  name: demo_complex_dimer
  mass_kda: 600
  diameter_A: 250
  symmetry: C2
  n_subunits: 2
  oligomer: 2
  flexibility: 0.4
  pixel_size_A: 3.4
  box_px: null
  n_particles_start: 12000
  extra_heterogeneity: 1
source_project: 0
"""
