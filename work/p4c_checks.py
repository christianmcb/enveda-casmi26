"""P4c checks: (1) Ridge-alpha sensitivity of the official config, (2) per-source gains, (3) does the gain depend on how close a
val molecule is to the training molecules (memorisation of library fragments vs real generalisation)?

Run from the repo root AFTER the fits:  python work/p4c_checks.py   (scores saved predictions; ~5 min, light on memory)
Writes artifacts/p4c_checks_scores.csv, p4c_by_source.csv, p4c_by_novelty.csv, p4c_pred_*.npy (predicted val fingerprints).
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import benchmark as bm
import v1_pipeline as v1

D = v1.load_data()
B = bm.Bench(D, all_val=True)
va, tr = D["va"], D["tr"]
scores = []

# predictions come from separate processes (work/p4c_fit.py), one fit each; v0.2b = uniform 0.1 Da both blocks (work/p4c_defect_bins.py run)
for name, alpha in [("v0.2b", 1.0), ("v0.2c", 1.0), ("v0.2c_alpha3", 3.0), ("v0.2c_alpha10", 10.0)]:
    P = np.load(f"artifacts/p4c_pred_{name}.npy")
    scores.append(B.score(name, P).assign(alpha=alpha))
    print(scores[-1][["version", "slice", "model", "model_minus_random"]].round(4).to_string(index=False), flush=True)
    pd.concat(scores).to_csv("artifacts/p4c_checks_scores.csv", index=False)

# ---- per-source gains (all val spectra, v0.1 10 ppm window) -------------------------------------------------------
def gain(a, b, mask, sl="all_val", n_boot=1000):
    ra, rb = B.per_query[(a, sl)], B.per_query[(b, sl)]
    d = (rb.rr_model - ra.rr_model)[mask]
    g = d.groupby(ra.key.values[mask]).agg(["sum", "size"])
    idx = np.random.default_rng(0).integers(len(g), size=(n_boot, len(g)))
    boots = g["sum"].values[idx].sum(1) / g["size"].values[idx].sum(1)
    return dict(n=int(mask.sum()), v0_2b=ra.rr_model[mask].mean(), v0_2c=rb.rr_model[mask].mean(), diff=d.mean(),
                ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5))
src = va.ingest_lib.values
rows = [dict(source=s, **gain("v0.2b", "v0.2c", src == s)) for s in pd.Series(src).value_counts().index if (src == s).sum() >= 30]
rows.append(dict(source="ALL", **gain("v0.2b", "v0.2c", np.ones(len(va), bool))))
by_src = pd.DataFrame(rows); by_src.to_csv("artifacts/p4c_by_source.csv", index=False)

# ---- novelty: max Tanimoto similarity of each val molecule to ANY training molecule -----------------------------------
FPb = D["FP"].astype(np.float32); row_of = pd.Series(np.arange(len(D["mols_u"])), index=D["mols_u"].key)
tr_keys, va_keys = tr.key.unique(), va.key.unique()
A, T = FPb[row_of[va_keys].values], FPb[row_of[tr_keys].values]
na, nt = A.sum(1), T.sum(1)
maxsim = np.zeros(len(va_keys), np.float32)
for i in range(0, len(va_keys), 1000):
    inter = A[i:i + 1000] @ T.T
    maxsim[i:i + 1000] = (inter / (na[i:i + 1000, None] + nt[None, :] - inter + 1e-9)).max(1)
ms_q = pd.Series(maxsim, index=va_keys)[va.key.values].values
bins = [0, .3, .4, .5, .6, 1.01]; lab = ["<0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", ">=0.6"]
cat = pd.cut(ms_q, bins, labels=lab, right=False)
rows = []
for s_name, smask in [("enveda-180", src == "enveda-180"), ("all non-enveda-180", src != "enveda-180"), ("all", np.ones(len(va), bool))]:
    for l in lab:
        m = smask & (cat == l)
        if m.sum() >= 30:
            rows.append(dict(group=s_name, max_tanimoto_to_train=l, **gain("v0.2b", "v0.2c", m)))
by_nov = pd.DataFrame(rows); by_nov.to_csv("artifacts/p4c_by_novelty.csv", index=False)
pd.set_option("display.width", 250)
print(pd.concat(scores)[lambda d: d.slice.isin(["enveda180", "mixed_2000"])].pivot_table(index="version", columns="slice", values="model", sort=False).round(4))
print(by_src.round(4).to_string(index=False)); print(by_nov.round(4).to_string(index=False))
