from __future__ import annotations

import copy
import os

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

DEFAULTS = {
    "projects": [],
    "output_dir": "sta_landscape_out",
    "free_energy": {
        # F = w_res*E_res + w_particles*E_particles + w_noise*E_noise + w_compute*E_compute
        "weights": {"resolution": 1.0, "particles": 0.3, "noise": 0.5, "compute": 0.0},
        "particles_preference": "more",       # more | fewer | target | none
        "particles_target": None,             # absolute number, used with 'target'
        "resolution_reference_A": None,       # None -> 2 x smallest pixel size in the project
        "temperature": 0.1,                   # Boltzmann temperature (F units) for learning from my choices
    },
    "noise": {
        "components": {"spectral": 1.0, "pmax": 0.5, "angular_accuracy": 0.0, "halfmap": 1.0,
                       "solvent": 0.5, "mask_artefact": 0.5, "user": 1.0},
        "spectral_band": [0.2, 0.8],          # fraction of Nyquist frequency used for the SSNR noise fraction
        "angular_accuracy_scale_deg": 10.0,
        "use_maps": True,                     # needs 'mrcfile'; silently skipped if not installed
    },
    "model": {
        "kappa": 1.0,             # exploration weight in lower-confidence-bound planning
        "beam_width": 6,
        "horizon": 3,
        "n_candidates": 250,
        "improvement_margin": 0.02,
        "bandwidth": 1.0,         # state-similarity kernel width (standardised units)
        "seed": 0,
        "bo_replays": 20,
        "extrapolation": 0.25,    # predicted states may exceed the observed range by at most this fraction
    },
    "physics": {
        "default_bfactor_A2": 400.0,   # used for Rosenthal-Henderson extrapolation if it can't be fitted
        "mass_exponent": 1.0,          # N_eff scales as (mass ratio)^exponent
    },
}

TEMPLATE = """
# sta_landscape configuration
projects:
  - dir: /path/to/my/RELION/project
    particle:
      name: my_complex
      mass_kda: 250          # total mass of what I am averaging
      diameter_A: 180        # longest dimension
      symmetry: C1
      n_subunits: 1
      oligomer: 1
      flexibility: 0.3       # 0 = rigid, 1 = very flexible (my judgement)
      n_particles_start: null  # null -> largest particle set seen in the project
    exclude_jobs: []         # e.g. [Class3D/job031] (tests, mistakes)
    annotations: null        # optional CSV: job,noise_score(1-5),exclude,override_res,override_n,note

output_dir: sta_landscape_out

free_energy:
  weights: {resolution: 1.0, particles: 0.3, noise: 0.5, compute: 0.0}
  particles_preference: more    # more | fewer | target | none
  particles_target: null        # used when preference = target
  resolution_reference_A: null  # null -> 2 x smallest pixel size (Nyquist) in the project
  temperature: 0.1              # how sharply the model imitates my most successful moves

noise:
  # Noise index = weighted mean of whichever components are available for a job (each in [0,1])
  components: {spectral: 1.0, pmax: 0.5, angular_accuracy: 0.0, halfmap: 1.0, solvent: 0.5, mask_artefact: 0.5, user: 1.0}
  spectral_band: [0.2, 0.8]     # SSNR noise fraction averaged over this band (fraction of Nyquist)
  angular_accuracy_scale_deg: 10.0
  use_maps: true                # half-map / solvent statistics (needs `pip install mrcfile`)

model:
  kappa: 1.0            # >0 explores (optimism under uncertainty) when searching for deeper minima
  beam_width: 6
  horizon: 3            # max number of Class3D/Refine3D steps in a proposed plan
  n_candidates: 250
  improvement_margin: 0.02
  bandwidth: 1.0
  seed: 0
  bo_replays: 20
  extrapolation: 0.25   # predictions may exceed my observed range by at most this fraction

physics:
  default_bfactor_A2: 400.0
  mass_exponent: 1.0
"""

TARGET_TEMPLATE = """# Describe the NEW particle I want a plan for.
target:
  name: my_complex_dimer
  mass_kda: 500
  diameter_A: 260
  symmetry: C2
  n_subunits: 2
  oligomer: 2
  flexibility: 0.4
  pixel_size_A: 5.4          # pixel size of the subtomos I will start with
  box_px: 96                 # null -> recommended from source box/diameter ratio
  n_particles_start: 30000
  extra_heterogeneity: 1     # extra classes to add (e.g. partially occupied extra subunit)
source_project: 0            # index into 'projects' of the config to transfer from
"""


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path):
    if yaml is None:
        raise SystemExit("PyYAML is required: pip install pyyaml")
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def load_config(path=None, overrides=None):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        cfg = _merge(cfg, load_yaml(path))
        base = os.path.dirname(os.path.abspath(path))
        for p in cfg["projects"]:
            if not os.path.isabs(p["dir"]):
                p["dir"] = os.path.normpath(os.path.join(base, p["dir"]))
            if p.get("annotations") and not os.path.isabs(p["annotations"]):
                p["annotations"] = os.path.normpath(os.path.join(base, p["annotations"]))
    if overrides:
        cfg = _merge(cfg, overrides)
    for p in cfg["projects"]:
        p.setdefault("particle", {})
        p.setdefault("exclude_jobs", [])
        p.setdefault("annotations", None)
    return cfg
