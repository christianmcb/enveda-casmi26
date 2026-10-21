"""Assemble the Kaggle notebooks from work/kaggle_cells.py.   Run from the repo root:  python work/build_kaggle_notebook.py
One notebook version = one submission.csv, so there are two single-model notebooks (the ones to submit) and a combined one (dev/testing):
  kaggle/casmi_submission_baseline.ipynb  Ridge baseline only
  kaggle/casmi_submission_deep.ipynb      deep residual MLP only
  kaggle/casmi_submission_both.ipynb      both models (submission.csv = FINAL_CHOICE)"""
import os
import re
import sys

import nbformat as nbf

sys.path.insert(0, "work")
import kaggle_cells

VARIANTS = {
    "baseline": dict(RUN_BASELINE="True", RUN_DEEP="False", FINAL_CHOICE='"baseline"',
                     note="**This notebook trains only the Ridge baseline**, so that one notebook = one submission. The deep-MLP version is `casmi_submission_deep.ipynb`."),
    "deep": dict(RUN_BASELINE="False", RUN_DEEP="True", FINAL_CHOICE='"deep"',
                 note="**This notebook trains only the deep residual MLP**, so that one notebook = one submission. The Ridge version is `casmi_submission_baseline.ipynb`."),
    "both": dict(RUN_BASELINE="True", RUN_DEEP="True", FINAL_CHOICE='"deep"', note=None),
}
os.makedirs("kaggle", exist_ok=True)
for name, v in VARIANTS.items():
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
    for kind, cname, src in kaggle_cells.CELLS:
        if kind == "code" and cname == "config":
            for key in ("RUN_BASELINE", "RUN_DEEP", "FINAL_CHOICE"):
                src, n = re.subn(rf"^{key} = [^\s#]+", f"{key} = {v[key]}", src, flags=re.M); assert n == 1, key
        if kind == "markdown" and cname == "title" and v["note"]:
            src = src.replace("\n\n**What it does, in order**", f"\n\n{v['note']}\n\n**What it does, in order**", 1)
        c = nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src)
        c.metadata["name"] = cname; nb.cells.append(c)
    nbf.validate(nb); path = f"kaggle/casmi_submission_{name}.ipynb"; nbf.write(nb, path)
    print("wrote", path, "|", re.findall(r"^(RUN_BASELINE = \w+|RUN_DEEP = \w+|FINAL_CHOICE = \"\w+\")", next(c.source for c in nb.cells if c.metadata.get("name") == "config"), flags=re.M))
