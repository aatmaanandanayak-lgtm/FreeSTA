# Methods

Detailed description of how `sta_landscape` reads a RELION project, defines the free energy, learns the landscape and answers the three questions. See the [README](../README.md) for installation and usage.

## 1. How my RELION project is read

| Source | What is extracted |
|---|---|
| `default_pipeline.star` | aliases, status, input edges (fallback) |
| `*/jobNNN/note.txt` | the **exact command line**, including "additional arguments". Every `relion_refine` flag becomes a landscape coordinate automatically. |
| `job.star` | GUI options (kept for reference) |
| `Class3D`: `run_itNNN_model.star`, `run_itNNN_data.star` | per-class resolution, class distribution, per-class SSNR, pmax, angular accuracy, particles per class |
| `Select`: `particles.star`, `backup_selection.star` | which classes I kept and how many particles. If the selection file is missing, the kept classes are inferred from the particle counts. |
| `Refine3D`: `run_model.star`, `run_data.star`, half maps | gold-standard resolution, SSNR, FSC, pmax, particle count |
| `PostProcess`: `postprocess.star` | final masked resolution, B-factor, phase-randomised FSC |
| `MaskCreate` | lowpass, threshold, extension and soft edge of the mask each job used. These become coordinates too. |
| anything in between (CtfRefineTomo, FrameAlignTomo, re-extraction…) | recorded as "run before this step" modifiers |
| file times, `RELION_JOB_EXIT_*` | wall-clock hours per job and failed/aborted jobs |

**States and moves.** A *state* is something I can measure. It is either a Class3D together with the Select job that chose classes from it (one state per selection, so re-selecting the same classification is a different move), or a Refine3D together with its best PostProcess. A *transition* is state → [job type + all parameters + class selection + intermediate jobs] → state.

**Excluding jobs.** Use `exclude_jobs` in the config or `exclude=true` in `annotations.csv`.

---

## 2. The free energy

Lower is better:

```
F = w_res · E_res  +  w_particles · E_particles  +  w_noise · E_noise  (+ w_compute · E_compute)

E_res       = ln(d / d_ref)               d = resolution (Å); d_ref = Nyquist of my finest pixel size
E_particles = −ln(N/N0)   "more"          N0 = starting particle set
              +ln(N/N0)   "fewer"
              (ln N/N_target)²  "target"
              0           "none"
E_noise     ∈ [0, 1]  (below)
E_compute   = ln(1 + cumulative hours along the lineage)   (off by default)
```

The log scales make the terms dimensionless and comparable across particles and pixel sizes. A 10 % gain in resolution is worth the same everywhere. The trade-off between resolution and particle count is set explicitly by the weights. Because of Rosenthal–Henderson (ln N ∝ B/2 · 1/d²), E_res and E_particles are physically linked, which the report also uses.

**Which particle preference to choose.** "more" rewards keeping particles, for example when I want the fullest possible dataset. "fewer" rewards purity, for example when I want the most homogeneous sub-state. "target" is for when I know roughly how many particles belong to the state I am after.

### Quantifying "noise"

No single number captures noise, so the noise index is a weighted mean of whichever of these components exist for a job. Each component is scaled to 0 (clean) … 1 (pure noise):

| component | definition | available for |
|---|---|---|
| `spectral` | mean of 1/(1+SSNR) over a frequency band (default 20–80 % of Nyquist) = fraction of power in that band that is noise | Class3D (per selected class), Refine3D |
| `pmax` | 1 − ⟨Pmax⟩: how ambiguous the orientation/class assignments are | Class3D, Refine3D |
| `angular_accuracy` | RELION's rotational accuracy / 10° (off by default) | Class3D, Refine3D |
| `halfmap` | var(h1 − h2) / var(h1 + h2) inside the mask = noise fraction in real space | Refine3D (needs `mrcfile`) |
| `solvent` | std outside mask / std inside mask of the map | Class3D, Refine3D (needs `mrcfile`) |
| `mask_artefact` | mean \|phase-randomised masked FSC\| beyond the resolution: how much correlation the mask is creating | PostProcess |
| `user` | my own 1–5 score from `annotations.csv` (column `noise_score`) | anything I annotate |

Weights are in `noise.components`. If a map "looks noisy" to me in a way these numbers miss, add a `noise_score` for it. My judgement then becomes part of F and the model learns from it.

---

## 3. The model

**Coordinates.** Every scientific relion_refine flag is a coordinate: numbers as numbers, switches as 0/1. Paths and compute-only flags (`--j`, `--pool`, `--gpu`…) are excluded. Flags whose meaning depends on the particle are re-expressed so that they transfer between particles:

| flag | coordinate |
|---|---|
| `--particle_diameter` | mask diameter ÷ particle diameter |
| `--offset_range/step` (px) | Å |
| `--strict_highres_exp` | f_limit / f_Nyquist (no limit = 1) |
| `--healpix_order`, `--auto_local_healpix_order` | angular step × particle radius (arc length in Å at the particle edge; a bigger particle needs finer sampling) |
| `--tau2_fudge` | log T |
| mask job | lowpass (Å), threshold, extension (Å), soft edge (Å) |
| selection | fraction of particles kept, number of classes kept |

Some coordinates are handled separately. Those that never varied, or only ever changed together with the job type (e.g. `--firstiter_cc`), are removed from the learnable set and listed as unexplored directions. Candidates are only proposed within, or 25 % beyond, the range I explored *for that job type*.

**Dynamics model.** For each state variable (E_res, ln particle fraction, noise) plus log(compute hours), it learns the *change* caused by a job:

```
Δstate = g(state before, job type, parameters, [particle descriptors])
```

Each target is an average of two models:

- A Gaussian process with an ARD Matérn kernel. This gives smooth behaviour, calibrated uncertainty and a length-scale per parameter.
- An extra-trees ensemble. This copes with thresholds and interactions.

Their disagreement is part of the reported uncertainty. Predictions are Monte-Carlo propagated into a distribution of F and clipped to at most 25 % beyond the observed range (`model.extrapolation`). Cross-validated R² for each target is printed at the top of the report. **Read it first.** If R² is low, the landscape is under-sampled.

**Learning from how I got there.** When proposing moves, the planner samples from my own past moves. Each move is weighted by a Boltzmann factor exp(ΔF_improvement / T) (T = `free_energy.temperature`) and by how similar its starting state is to the current one. Proposals are local perturbations of my successful decisions, plus a fraction of exploratory ones.

**Planning.** A beam search looks over sequences of Class3D/Refine3D actions, up to `model.horizon` steps, and every plan must end in a Refine3D so it finishes on a gold-standard resolution. Plans are scored by the predicted F in one of three modes:

- `expected`: the mean.
- `explore`: mean − κ·sd, which looks for deeper minima.
- `safe`: mean + κ·sd, the conservative choice for a new particle.

Every proposed step carries a **novelty** score: its distance to the nearest job I ran, in units of the typical spacing between my jobs. Novelty > 2 means extrapolation.

**Sensitivity and interactions.** Each coordinate's influence on F is measured by permutation importance. Pairwise interactions ("in concert" effects) use Friedman's H² statistic: 0 means the two act additively, and ≥ 0.1 means the effect of one depends on the value of the other. The report also shows 1-D slices through my best Class3D and Refine3D.

---

## 4. Outputs (`output_dir`)

| file | content |
|---|---|
| `report.html` | everything below, with figures (self-contained) |
| `figures/*.png` | every report figure as a separate PNG |
| `jobs.csv`, `states.csv`, `transitions.csv` | parsed project, measured states with all F terms and noise components, and the learning table |
| `model_skill.csv` | cross-validated R² / RMSE of the dynamics model |
| `sensitivity.csv`, `interactions.csv` | which parameters matter, alone and together |
| `timeline.csv`, `skippable_steps.csv`, `parameter_scans.csv` | Q1 |
| `next_job_candidates.csv`, `summary.json` (plans) | Q2 |
| `recipe.csv` | my critical path to the minimum, in RELION terms |
| `transfer_plan.json`, `transfer_commands.txt` | the plan for the new particle, with ready-to-edit `relion_refine` command lines built from my own commands (paths replaced by `<PLACEHOLDERS>`) |

### Q1: rate

- A timeline of best-so-far F against jobs run and against wall-clock hours.
- The critical path: the lineage of my best state, and what fraction of compute was spent on it.
- **Skippable steps**: for each step on the path, the next move is applied directly to the state before it, to see whether the step could have been skipped.
- **Parameter scans**: sibling jobs from the same input whose F barely differed (a flat direction).
- **Replay**: my own measured jobs are re-ordered by (a) my chronology, (b) expected-improvement Bayesian optimisation, and (c) random choice. A job is only available once its input exists. It reports jobs and hours to reach the minimum and mean regret. Because it only uses jobs I really ran, it is an honest (if conservative) comparison.
- A suggested tuning order: the most influential parameters first, tuned jointly in small batches.

### Q2: depth

- P(deeper minimum): the highest probability, over proposed next jobs and plans, of beating my best F by more than `improvement_margin`.
- The best single next jobs, each described as a diff against my most similar real job (e.g. `--tau2_fudge: 1 → 2.2; mask.soft_edge_A: 27 → 21`).
- Multi-step plans from my best state and from each fork on my path. Sometimes the deeper basin branches off earlier.
- **Unexplored directions**: coordinates the model cannot see. These are often where a deeper minimum hides, for example `--sigma_ang`, `--offset_range`, mask threshold, `--pad`, or iterations.
- A Nyquist check: if the best resolution is within 30 % of Nyquist, the next minimum is behind less binning. That is a re-extraction step, which the model has not seen.
- A Rosenthal–Henderson fit over my refinements. If ln N does not rise with 1/d², particle number is not my limiting factor.

### Transfer to a modified particle

I describe the target in `target.yaml`: mass, diameter, symmetry, pixel size, starting particle number, and `extra_heterogeneity` (extra classes for, e.g., a partially occupied subunit). The plan has three layers:

1. **Recipe transfer.** My critical path is replayed with every parameter re-derived through its dimensionless coordinate, plus a few rules:
   - K is scaled by √(particle ratio), plus the extra classes.
   - Refinements use the target symmetry. Classification stays in my source symmetry, with a note suggesting C2 or `--relax_sym` for a pseudo-symmetric dimer.
   - The box keeps my box/diameter ratio and is rounded to an FFT-friendly size.
   - Mask extension and soft edge are kept in Å.
2. **Model prediction** of resolution, particles and noise after each step. With several source projects in the config, the particle descriptors (`ctx.*`) are model inputs, so this prediction becomes particle-aware.
3. **Physics prior (Rosenthal–Henderson):** N_eff = particles kept × symmetry order × (mass ratio)^α, and 1/d_t² = 1/d_s² + (2/B)·ln(N_eff,t/N_eff,s). B comes from my own R–H fit when possible, otherwise from `physics.default_bfactor_A2`.

It also gives model-optimised alternative sequences, planned from the target's starting state.

---

## 5. Getting the most out of it (and caveats)

- **The model is only as good as my exploration.** With a few dozen jobs it is a structured summary of my landscape, not an oracle. Check `model_skill.csv`, the novelty scores, and the unexplored-directions list. Validate every proposal with a real job, then re-run `analyse`: the loop is the point.
- One-factor-at-a-time scans are easy to interpret by eye but poor for learning interactions. Occasionally vary two or three parameters at once, for example with the planner's batch proposals. The H² estimates will improve quickly.
- **Class3D resolutions are not gold-standard.** They are treated as a separate kind of state, and the model learns how they translate into Refine3D results.
- **Transfer from one particle** relies on the dimensionless coordinates and the physics prior. The more particles (projects) I add, the more the model learns how the optimum *shifts* with mass, size, symmetry and flexibility.
- Use `annotations.csv` for things RELION can't see: a class that is "junk but high resolution", a map with streaks, a job run with a mistake (`exclude=true`), or a manual resolution override.
- Tested against synthetic RELION-5-style projects (the `demo` command). Formats from RELION 3.1 to 5.0 are handled, but check `python -m sta_landscape scan` on my project first, and look at `jobs.csv` to confirm every job's parent and flags were read as I expect.
