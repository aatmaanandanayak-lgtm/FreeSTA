"""Regenerate docs/images/*.png and docs/example_report.html from the synthetic demo project.

    python scripts/make_docs_figures.py
"""
import glob
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sta_landscape.__main__ import main  # noqa: E402


def run():
    work = tempfile.mkdtemp(prefix="sta_demo_")
    main(["demo", "--dir", work])
    out = os.path.join(work, "output")
    img = os.path.join(ROOT, "docs", "images")
    os.makedirs(img, exist_ok=True)
    for p in glob.glob(os.path.join(out, "figures", "*.png")):
        shutil.copy(p, img)
    shutil.copy(os.path.join(out, "report.html"), os.path.join(ROOT, "docs", "example_report.html"))
    shutil.rmtree(work, ignore_errors=True)
    print("updated", img)


if __name__ == "__main__":
    run()
