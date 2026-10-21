"""What does a realistic candidate pool cost the current best model? (Step 1 of the external-pool plan; no training.)

Pool A = the 229,964-molecule benchmark pool (every earlier number).  Pool B = A + COCONUT-not-in-A (~432k NP decoys).
The true molecule is in both. Same predictions, same mass windows: only the number/kind of decoys changes.
Run from the repo root (RDKit env):  python work/p7a_pool_cost.py   -> artifacts/p7a_pool_cost_by_source.csv
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import benchmark as bm
import v1_pipeline as v1

C = pd.read_parquet("artifacts/coconut_v1.parquet"); cfp = np.load("artifacts/coconut_fp_v1.npy").astype(np.float32)
extra = (C.key.values, v1.unit(cfp), C.mass.values); del cfp
D = v1.load_data()
preds = {"deep_D": np.load("artifacts/scale/pred_deep_D.npy")}
rows = []
for pool_name, ex in [("A: 230k", None), ("B: 230k + COCONUT", extra)]:
    B = bm.Bench(D, all_val=True, extra=ex)
    src_ = B.va.ingest_lib.values                       # per-source slices (v0.1 window, all val spectra of that source) + all non-enveda-180
    B.slices["non_enveda180"] = (np.flatnonzero(src_ != "enveda-180"), 10.0, None)
    for sname in ["gnps", "riken", "pluskal_ms2", "spectraverse", "massbank", "mona", "msdial"]:
        B.slices["src_" + sname] = (np.flatnonzero(src_ == sname), 10.0, None)
    for name, P in preds.items():
        s = B.score(name, P).assign(pool=pool_name); rows.append(s)
        print(s[["pool", "slice", "n", "model", "random", "model_minus_random", "cands_median"]].round(4).to_string(index=False), flush=True)
    del B
out = pd.concat(rows); out.to_csv("artifacts/p7a_pool_cost_by_source.csv", index=False)
print(out.pivot_table(index=["version", "slice"], columns="pool", values=["model", "cands_median"]).round(4))
