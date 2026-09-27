"""Minimal, dependency-light STAR file reader for RELION 3.1 / 4 / 5 files.

It is written to be tolerant rather than complete: it understands data_ blocks,
key/value pairs and loop_ tables, quoted strings and comments.  For very large
particle files I can ask for a row count only, or for a subset of columns,
so a 500k-particle run_data.star does not have to be fully materialised.
"""
from __future__ import annotations

import os
import shlex
from typing import Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd

Table = Union[pd.DataFrame, Dict[str, object]]


def _split(line: str):
    if '"' in line or "'" in line:
        try:
            return shlex.split(line)
        except ValueError:
            return line.split()
    return line.split()


def _to_num(v):
    try:
        if isinstance(v, str) and v.lstrip("-+").isdigit():
            return int(v)
        return float(v)
    except (TypeError, ValueError):
        return v


def read_star(path: str,
              blocks: Optional[Iterable[str]] = None,
              columns: Optional[Dict[str, Iterable[str]]] = None,
              count_only: Iterable[str] = ()) -> Dict[str, Table]:
    """Read a STAR file.

    Parameters
    ----------
    blocks      : only return these data blocks (names without 'data_'); None = all.
    columns     : {block: [labels]} restrict loop columns (labels without leading '_').
    count_only  : block names for which only the number of rows is returned
                  (as {'__nrows__': n, '__labels__': [...]}).

    Returns {block_name: DataFrame (loops) or dict (key/value blocks)}.
    """
    out: Dict[str, Table] = {}
    if not path or not os.path.isfile(path):
        return out
    blocks = set(blocks) if blocks is not None else None
    count_only = set(count_only)
    columns = {k: list(v) for k, v in (columns or {}).items()}

    block = None
    kv: Dict[str, object] = {}
    labels: list = []
    rows: list = []
    nrows = 0
    state = "none"  # none | kv | labels | rows

    def want(b):
        return blocks is None or b in blocks

    def flush():
        nonlocal kv, labels, rows, nrows, state
        if block is None:
            return
        if state in ("labels", "rows"):
            if block in count_only:
                out[block] = {"__nrows__": nrows, "__labels__": list(labels)}
            elif want(block):
                sel = columns.get(block)
                if sel is not None:
                    idx = [labels.index(c) for c in sel if c in labels]
                    labs = [labels[i] for i in idx]
                else:
                    labs = labels
                df = pd.DataFrame(rows, columns=labs) if rows else pd.DataFrame(columns=labs)
                for c in df.columns:
                    conv = pd.to_numeric(df[c], errors="coerce")
                    if conv.notna().sum() == df[c].notna().sum():
                        df[c] = conv
                out[block] = df
        elif state == "kv" and want(block):
            out[block] = dict(kv)
        kv, labels, rows, nrows, state = {}, [], [], 0, "none"

    with open(path, "r", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                if state == "rows":
                    flush()
                continue
            if line.startswith("#"):
                continue
            if line.startswith("data_"):
                flush()
                block = line[5:].strip()
                state = "kv"
                continue
            if line.startswith("loop_"):
                if state == "rows":
                    flush()
                kv = {}
                state = "labels"
                labels = []
                continue
            if line.startswith("_"):
                if state == "rows":  # a new key after a loop without blank line
                    flush()
                    state = "kv"
                toks = _split(line)
                lab = toks[0][1:]
                if state == "labels":
                    labels.append(lab)
                else:
                    state = "kv"
                    kv[lab] = _to_num(toks[1]) if len(toks) > 1 else None
                continue
            # data row
            if state in ("labels", "rows"):
                state = "rows"
                if block in count_only or not want(block):
                    nrows += 1
                    continue
                toks = _split(line)
                sel = columns.get(block)
                if sel is not None:
                    idx = [labels.index(c) for c in sel if c in labels]
                    toks = [toks[i] if i < len(toks) else None for i in idx]
                rows.append(toks)
                nrows += 1
    flush()
    return out


def particle_count(path: str) -> Optional[int]:
    """Number of particle rows in a RELION data/particles STAR file (None if unknown)."""
    if not path or not os.path.isfile(path):
        return None
    # RELION >=3.1: data_particles ; older: data_ (unnamed) or data_images
    res = read_star(path, count_only=("particles", "images", ""))
    for key in ("particles", "images", ""):
        if key in res and isinstance(res[key], dict) and "__nrows__" in res[key]:
            return int(res[key]["__nrows__"])
    return None


def optimisation_set_particles(path: str) -> Optional[str]:
    """RELION 4/5 tomo optimisation-set files point to the particle STAR file."""
    d = read_star(path)
    for blk in d.values():
        if isinstance(blk, dict):
            for k in ("rlnTomoParticlesFile",):
                if k in blk:
                    return str(blk[k])
        elif isinstance(blk, pd.DataFrame) and "rlnTomoParticlesFile" in blk.columns and len(blk):
            return str(blk["rlnTomoParticlesFile"].iloc[0])
    return None


def write_star(path: str, blocks: Dict[str, Table]):
    """Tiny writer (used by the synthetic demo generator)."""
    with open(path, "w") as fh:
        fh.write("\n# version 50001\n\n")
        for name, blk in blocks.items():
            fh.write(f"data_{name}\n\n")
            if isinstance(blk, pd.DataFrame):
                fh.write("loop_\n")
                for i, c in enumerate(blk.columns, 1):
                    fh.write(f"_{c} #{i}\n")
                for row in blk.itertuples(index=False):
                    fh.write(" ".join(_fmt(v) for v in row) + "\n")
                fh.write("\n")
            else:
                for k, v in blk.items():
                    fh.write(f"_{k:<40s} {_fmt(v)}\n")
                fh.write("\n")


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.6g}"
    s = str(v)
    return f'"{s}"' if (" " in s or s == "") else s
