"""P5 scoring: v0.2c Ridge vs the MLP configurations, on both benchmark slices, plus per-bit AUC, per-source and novelty breakdowns.

Run from the repo root AFTER the fits:  python work/p5_score.py
The OFFICIAL MLP is the configuration with the lowest inner-holdout BCE (chosen from the training-split holdout only; validation
results are never used to pick it).
Writes artifacts/p5_scores.csv, p5_bootstrap.csv, p5_auc.csv, p5_by_source.csv, p5_by_novelty.csv, p5_selection.csv
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, "src")
import benchmark as bm
import v1_pipeline as v1

names = sorted(os.path.basename(f)[len("p5_pred_"):-4] for f in glob.glob("artifacts/p5_pred_*.npy") if "smoke" not in f)
sel = pd.DataFrame([dict(config=n, **{k: v for k, v in json.load(open(f"artifacts/p5_hist_{n}.json")).items() if k != "history"}) for n in names])
sel = sel.sort_values("best_holdout_bce"); sel.to_csv("artifacts/p5_selection.csv", index=False)
official = sel.config.iloc[0]
print("configs ranked by INNER-holdout BCE (selection rule):\n", sel.round(5).to_string(index=False), "\nOFFICIAL:", official, flush=True)

D = v1.load_data()
B = bm.Bench(D, all_val=True)
va = D["va"]
preds = {"v0.2c_ridge": np.load("artifacts/p4c_pred_v0.2c.npy"), **{n: np.load(f"artifacts/p5_pred_{n}.npy") for n in names}}
scores = pd.concat([B.score(n, P) for n, P in preds.items()])
scores.to_csv("artifacts/p5_scores.csv", index=False)
pd.set_option("display.width", 250)
print(scores[scores.slice.isin(["enveda180", "mixed_2000"])][["version", "slice", "model", "random", "model_minus_random", "cands_median", "fp_cosine"]].round(4).to_string(index=False), flush=True)

boot = []
for n in names:
    for sl in ("enveda180", "mixed_2000", "all_val"):
        d, (lo, hi) = B.paired_diff("v0.2c_ridge", n, sl)
        boot.append(dict(slice=sl, baseline="v0.2c_ridge", variant=n, diff=d, ci_lo=lo, ci_hi=hi))
pd.DataFrame(boot).to_csv("artifacts/p5_bootstrap.csv", index=False)

# ---- per-bit AUC: bit-level quality, independent of pool size and mass window ----------------------------------------
row_of = pd.Series(np.arange(len(D["mols_u"])), index=D["mols_u"].key)
Yva = D["FP"][row_of[va.key].values]
def mean_auc(P, mask, min_n=30):
    Y, S = Yva[mask], P[mask]
    pos = Y.sum(0); ok = (pos >= min_n) & (Y.shape[0] - pos >= min_n)
    r = rankdata(S[:, ok], axis=0); n1 = pos[ok]; n0 = Y.shape[0] - n1
    auc = (r * Y[:, ok]).sum(0) - n1 * (n1 + 1) / 2
    auc = auc / (n1 * n0)
    return float(auc.mean()), int(ok.sum())
e180 = (va.ingest_lib == "enveda-180").values
auc_rows = []
for n, P in preds.items():
    for grp, m in [("all val", np.ones(len(va), bool)), ("enveda-180 val", e180)]:
        a, nb = mean_auc(P, m); auc_rows.append(dict(version=n, group=grp, mean_bit_auc=a, bits_evaluated=nb))
auc = pd.DataFrame(auc_rows); auc.to_csv("artifacts/p5_auc.csv", index=False); print(auc.round(4).to_string(index=False), flush=True)

# ---- per-source and novelty breakdown for the OFFICIAL config vs v0.2c Ridge -----------------------------------------
def gain(a, b, mask, sl="all_val", n_boot=1000):
    ra, rb = B.per_query[(a, sl)], B.per_query[(b, sl)]
    d = (rb.rr_model - ra.rr_model)[mask]
    g = d.groupby(ra.key.values[mask]).agg(["sum", "size"])
    idx = np.random.default_rng(0).integers(len(g), size=(n_boot, len(g)))
    boots = g["sum"].values[idx].sum(1) / g["size"].values[idx].sum(1)
    return dict(n=int(mask.sum()), ridge=ra.rr_model[mask].mean(), mlp=rb.rr_model[mask].mean(), diff=d.mean(),
                ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5))
src = va.ingest_lib.values
rows = [dict(source=s, **gain("v0.2c_ridge", official, src == s)) for s in pd.Series(src).value_counts().index if (src == s).sum() >= 30]
rows.append(dict(source="ALL", **gain("v0.2c_ridge", official, np.ones(len(va), bool))))
pd.DataFrame(rows).to_csv("artifacts/p5_by_source.csv", index=False)
FPb = D["FP"].astype(np.float32); tr_keys, va_keys = D["tr"].key.unique(), va.key.unique()
A, T = FPb[row_of[va_keys].values], FPb[row_of[tr_keys].values]; na, nt = A.sum(1), T.sum(1); maxsim = np.zeros(len(va_keys), np.float32)
for i in range(0, len(va_keys), 1000):
    inter = A[i:i + 1000] @ T.T; maxsim[i:i + 1000] = (inter / (na[i:i + 1000, None] + nt[None, :] - inter + 1e-9)).max(1)
cat = pd.cut(pd.Series(maxsim, index=va_keys)[va.key.values].values, [0, .3, .4, .5, .6, 1.01], labels=["<0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", ">=0.6"], right=False)
rows = []
for gname, smask in [("enveda-180", src == "enveda-180"), ("all non-enveda-180", src != "enveda-180")]:
    for l in ["<0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", ">=0.6"]:
        m = smask & (cat == l)
        if m.sum() >= 30: rows.append(dict(group=gname, max_tanimoto_to_train=l, **gain("v0.2c_ridge", official, m)))
pd.DataFrame(rows).to_csv("artifacts/p5_by_novelty.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False)); print(pd.read_csv("artifacts/p5_bootstrap.csv").round(4).to_string(index=False))
