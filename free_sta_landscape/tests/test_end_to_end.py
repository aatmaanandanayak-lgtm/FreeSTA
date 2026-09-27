"""Builds a synthetic RELION project with a known landscape and runs the full pipeline."""
import json

from sta_landscape.config import load_config
from sta_landscape.demo import make_demo_project
from sta_landscape.pipeline import run


def test_full_pipeline(tmp_path):
    proj = tmp_path / "relion_project"
    make_demo_project(str(proj), seed=1)
    cfg = load_config(overrides={
        "projects": [{"dir": str(proj), "particle": {"name": "demo", "mass_kda": 300, "diameter_A": 180, "symmetry": "C1"}}],
        "output_dir": str(tmp_path / "out"),
        "free_energy": {"weights": {"particles": 0.1}},
        "model": {"bo_replays": 2, "n_candidates": 80, "horizon": 2, "beam_width": 3},
    })
    target = {"name": "dimer", "mass_kda": 600, "diameter_A": 250, "symmetry": "C2", "pixel_size_A": 3.4,
              "n_particles_start": 12000, "extra_heterogeneity": 1}
    res = run(cfg, target=target)
    out = tmp_path / "out"
    for f in ("report.html", "states.csv", "transitions.csv", "summary.json", "transfer_plan.json",
              "transfer_commands.txt", "figures/01_exploration_tree.png"):
        assert (out / f).exists(), f
    s = json.loads((out / "summary.json").read_text())
    assert 0.0 <= s["p_deeper"] <= 1.0
    assert len(s["path"]) >= 2
    assert res["transfer"]["steps"][-1]["kind"] == "refine"
    assert "--sym C2" in (out / "transfer_commands.txt").read_text()
