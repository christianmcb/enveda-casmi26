"""Equivalence test: the notebook's ranking/windowing/pooling code vs the project benchmark, on identical inputs.
Uses the cells from work/kaggle_cells.py (the exact code that ends up in the Kaggle notebook) with the benchmark's 229,964-molecule pool.
Targets (deep model, tier D, enveda-180 val): per-spectrum MRR 0.6706 (benchmark), pooled-by-molecule MRR 0.7314 (p6pre_pooling).
Run from the repo root:  python work/test_kaggle_rank.py"""
import re
import sys

import numpy as np
import pandas as pd
from rdkit import Chem

sys.path.insert(0, "src"); sys.path.insert(0, "work")
import benchmark as bm
import kaggle_cells
import mass_window as mw
import v1_pipeline as v1

g = {"np": np, "pd": pd, "Chem": Chem, "tick": lambda m: print(m, flush=True)}
for name in ("helpers_mass", "ranking"):                                   # exactly the notebook's code
    exec(next(src for k, n, src in kaggle_cells.CELLS if n == name), g)

D = v1.load_data(); B = bm.Bench(D); va = D["va"]
lib = g["Library"](list(B.keys), B.keys, B.mass, B.fp)                     # benchmark pool: unit-length fps, exact masses
e = np.flatnonzero((va.ingest_lib == "enveda-180").values); q = va.iloc[e].reset_index(drop=True)
P = np.load("artifacts/scale/pred_deep_D.npy")[e]
lo, hi = g["mass_window"](q.neutral_mass.values, q.adduct.values, B.HW, B.BIAS)
lo2, hi2 = mw.window(q.neutral_mass.values, q.adduct.values, B.HW, B.BIAS)
print("window code identical to src/mass_window.py:", bool(np.allclose(lo, lo2) and np.allclose(hi, hi2)))

def rr_from(ranked, true_keys):
    return np.array([1 / (list(lib.keys[r]).index(k) + 1) if k in set(lib.keys[r]) else 0.0 for r, k in zip(ranked, true_keys)])
ranked, ncand = g["rank_queries"](g["unit"](P), lo, hi, lib)
rr = rr_from(ranked, q.key.values)
bench = B.score("t", np.load("artifacts/scale/pred_deep_D.npy")); bench = bench[bench.slice == "enveda180"].iloc[0]
print(f"per-spectrum MRR: notebook code {rr.mean():.4f} | benchmark {bench.model:.4f} | median candidates {np.median(ncand):.0f} vs {bench.cands_median:.0f}")

keys, Pm, L, H, nbad = g["pool_molecules"](P, q.key.values, lo, hi)
ranked_m, _ = g["rank_queries"](Pm, L, H, lib)
rr_m = rr_from(ranked_m, keys)
print(f"pooled-by-molecule MRR: notebook code {rr_m.mean():.4f} | p6pre_pooling 0.7314 | molecules {len(keys)} | non-overlapping window sets {nbad}")
assert abs(rr.mean() - bench.model) < 0.003 and abs(rr_m.mean() - 0.7314) < 0.004, "notebook ranking code does not reproduce the benchmark"
print("EQUIVALENCE TEST PASSED")
# sanity of the submission validator
sub = pd.DataFrame({"molecule_id": ["a", "b"], "smiles": ["CCO;CCN", "CCC"]}); n = g["validate_submission"](sub, ["a", "b"]); print("validator ok, guesses:", n.tolist())
for bad, msg in [(pd.DataFrame({"molecule_id": ["a", "a"], "smiles": ["C", "C"]}), "repeat"), (pd.DataFrame({"molecule_id": ["a", "b"], "smiles": ["C", ""]}), "empty"),
                 (pd.DataFrame({"molecule_id": ["a", "b"], "smiles": ["C", ";".join(["C"] * 26)]}), ">25")]:
    try: g["validate_submission"](bad, ["a", "b"]); print("VALIDATOR MISSED:", msg)
    except AssertionError: print("validator rejects:", msg)
