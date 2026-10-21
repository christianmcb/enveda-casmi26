"""Pre-P6 check: does pooling a molecule's spectra beat scoring them one at a time? (the submission needs ONE ranking per molecule_id)
Pooled query = mean of the predicted fingerprints of the molecule's spectra + intersection of their (P3-policy) mass windows
(fallback to the union if the windows do not overlap). Compared with the mean of the per-spectrum reciprocal ranks.
Run from the repo root:  python work/p6pre_pooling.py  -> artifacts/pooling_check.csv
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import benchmark as bm
import mass_window as mw
import v1_pipeline as v1

D = v1.load_data(); B = bm.Bench(D); va = D["va"]
e = np.flatnonzero((va.ingest_lib == "enveda-180").values)
q = va.iloc[e].reset_index(drop=True)
lo_all, hi_all = mw.window(q.neutral_mass.values, q.adduct.values, B.HW, B.BIAS)
groups = q.groupby("key").indices
print(f"enveda-180 val: {len(q):,} spectra, {len(groups):,} molecules; spectra/molecule {len(q) / len(groups):.2f}; "
      f"molecules with >1 adduct {sum(q.adduct.values[i].tolist().__len__() and len(set(q.adduct.values[i])) > 1 for i in groups.values()):,}", flush=True)
rows = []
for name, path in [("ridge_A (v0.2c)", "artifacts/scale/pred_ridge_A.npy"), ("ridge_D", "artifacts/scale/pred_ridge_D.npy"), ("deep_D", "artifacts/scale/pred_deep_D.npy")]:
    P = np.load(path)[e]
    keys = np.array(list(groups.keys()))
    Pm = np.stack([P[groups[k]].mean(0) for k in keys]); Pm = v1.unit(Pm)
    lo = np.array([lo_all[groups[k]].max() for k in keys]); hi = np.array([hi_all[groups[k]].min() for k in keys])
    bad = lo > hi
    lo[bad] = np.array([lo_all[groups[k]].min() for k in keys])[bad]; hi[bad] = np.array([hi_all[groups[k]].max() for k in keys])[bad]
    pooled = B._evaluate(Pm, keys, lo, hi)
    single = B._evaluate(v1.unit(P), q.key.values, lo_all, hi_all)
    single_mol = single.groupby(q.key.values).rr_model.mean().reindex(keys)
    multi = np.array([len(groups[k]) > 1 for k in keys])
    rows.append(dict(model=name, molecules=len(keys), pooled_mrr=pooled.rr_model.mean(), per_spectrum_mrr=single_mol.mean(),
                     pooled_minus_single=pooled.rr_model.mean() - single_mol.mean(),
                     pooled_mrr_multi=pooled.rr_model[multi].mean(), single_mrr_multi=single_mol[multi].mean(),
                     window_miss_pooled_pct=(1 - pooled.inside.mean()) * 100, empty_intersection_pct=bad.mean() * 100,
                     cands_median_pooled=pooled.cands.median(), cands_median_single=single.cands.median()))
    print(pd.DataFrame(rows[-1:]).round(4).T.to_string(header=False), flush=True)
pd.DataFrame(rows).to_csv("artifacts/pooling_check.csv", index=False)
