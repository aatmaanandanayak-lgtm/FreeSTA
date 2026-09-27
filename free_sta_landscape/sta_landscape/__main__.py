"""Command line:

  python -m sta_landscape init      [--dir DIR]              write config/target/annotation templates
  python -m sta_landscape scan      -c config.yaml           parse projects -> jobs.csv, states.csv, transitions.csv
  python -m sta_landscape analyse   -c config.yaml           learn landscape, Q1 efficiency, Q2 deeper minimum, report.html
  python -m sta_landscape transfer  -c config.yaml -t target.yaml   + plan for a modified particle
  python -m sta_landscape demo      [--dir DIR]              synthetic project -> full run (no real data needed)
"""
from __future__ import annotations

import argparse
import os
import sys

from .config import TARGET_TEMPLATE, TEMPLATE, load_config, load_yaml


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sta_landscape", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("init")
    a.add_argument("--dir", default=".")
    for name in ("scan", "analyse", "analyze", "transfer"):
        s = sub.add_parser(name)
        s.add_argument("-c", "--config", required=True)
        s.add_argument("-t", "--target", default=None)
        s.add_argument("-o", "--output-dir", default=None)
        s.add_argument("--source-project", type=int, default=None)
    d = sub.add_parser("demo")
    d.add_argument("--dir", default="sta_landscape_demo")
    d.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    if args.cmd == "init":
        os.makedirs(args.dir, exist_ok=True)
        for fn, txt in (("sta_config.yaml", TEMPLATE), ("target.yaml", TARGET_TEMPLATE),
                        ("annotations.csv", "job,noise_score,exclude,override_res,override_n,note\n")):
            p = os.path.join(args.dir, fn)
            if os.path.exists(p):
                print(f"exists, not overwritten: {p}")
                continue
            with open(p, "w") as fh:
                fh.write(txt)
            print(f"wrote {p}")
        return 0

    if args.cmd == "demo":
        from .demo import DEMO_CONFIG, DEMO_TARGET, make_demo_project
        root = os.path.abspath(args.dir)
        proj = os.path.join(root, "relion_project")
        if not os.path.isdir(proj):
            make_demo_project(proj, seed=args.seed)
        cfgp, tgtp = os.path.join(root, "sta_config.yaml"), os.path.join(root, "target.yaml")
        with open(cfgp, "w") as fh:
            fh.write(DEMO_CONFIG.format(project=proj, out=os.path.join(root, "output")))
        with open(tgtp, "w") as fh:
            fh.write(DEMO_TARGET)
        args.cmd, args.config, args.target, args.output_dir, args.source_project = "transfer", cfgp, tgtp, None, None

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    target, src = None, 0
    if args.target:
        t = load_yaml(args.target)
        target = t.get("target", t)
        src = t.get("source_project", 0) or 0
    if args.source_project is not None:
        src = args.source_project

    if args.cmd == "scan":
        from .pipeline import scan_all
        out, tr, _, _, jt = scan_all(cfg)
        os.makedirs(cfg["output_dir"], exist_ok=True)
        jt.to_csv(os.path.join(cfg["output_dir"], "jobs.csv"), index=False)
        out.to_csv(os.path.join(cfg["output_dir"], "states.csv"), index=False)
        tr.to_csv(os.path.join(cfg["output_dir"], "transitions.csv"), index=False)
        print(out[["outcome", "kind", "res_A", "n", "noise", "F"]].sort_values("F").to_string(index=False))
        return 0

    from .pipeline import run
    run(cfg, target=target if args.cmd == "transfer" or target else None, source_index=src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
