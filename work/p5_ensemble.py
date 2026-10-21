"""P5 diagnostic (not a version): average the v0.2c Ridge and the official MLP. NO training: both prediction files already exist.
Declared test BEFORE looking at results: weight 0.5 on each. 0.25 / 0.75 are sensitivity checks only (not selected on validation).

Run from the repo root:  python work/p5_ensemble.py   -> artifacts/p5_ensemble_scores.csv, p5_ensemble_bootstrap.csv, p5_ensemble_novelty.csv
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import benchmark as bm
import v1_pipeline as v1

official = pd.read_csv("artifacts/p5_selection.csv").config.iloc[0]
D = v1.load_data(); B = bm.Bench(D, all_val=True); va = D["va"]
Pr = np.clip(np.load("artifacts/p4c_pred_v0.2c.npy"), 0, 1)          # Ridge outputs are unbounded; clip to a probability scale
Pm = np.load(f"artifacts/p5_pred_{official}.npy")
preds = {"ridge": Pr, "mlp": Pm, **{f"ens_w{w}": w * Pr + (1 - w) * Pm for w in (0.25, 0.5, 0.75)}}
sc = pd.concat([B.score(n, P) for n, P in preds.items()]); sc.to_csv("artifacts/p5_ensemble_scores.csv", index=False)
pd.set_option("display.width", 250)
print(sc[sc.slice.isin(["enveda180", "mixed_2000"])][["version", "slice", "model", "model_minus_random", "fp_cosine"]].round(4).to_string(index=False), flush=True)
boot = []
for n in ("mlp", "ens_w0.25", "ens_w0.5", "ens_w0.75"):
    for sl in ("enveda180", "mixed_2000", "all_val"):
        d, (lo, hi) = B.paired_diff("ridge", n, sl); boot.append(dict(slice=sl, baseline="ridge", variant=n, diff=d, ci_lo=lo, ci_hi=hi))
pd.DataFrame(boot).to_csv("artifacts/p5_ensemble_bootstrap.csv", index=False); print(pd.DataFrame(boot).round(4).to_string(index=False), flush=True)

# novelty bins for the declared ensemble (w=0.5)
row_of = pd.Series(np.arange(len(D["mols_u"])), index=D["mols_u"].key); src = va.ingest_lib.values
FPb = D["FP"].astype(np.float32); tr_keys, va_keys = D["tr"].key.unique(), va.key.unique()
A, T = FPb[row_of[va_keys].values], FPb[row_of[tr_keys].values]; na, nt = A.sum(1), T.sum(1); maxsim = np.zeros(len(va_keys), np.float32)
for i in range(0, len(va_keys), 1000):
    inter = A[i:i + 1000] @ T.T; maxsim[i:i + 1000] = (inter / (na[i:i + 1000, None] + nt[None, :] - inter + 1e-9)).max(1)
labs = ["<0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", ">=0.6"]
cat = pd.cut(pd.Series(maxsim, index=va_keys)[va.key.values].values, [0, .3, .4, .5, .6, 1.01], labels=labs, right=False)
def gain(a, b, mask, sl="all_val", n_boot=1000):
    ra, rb = B.per_query[(a, sl)], B.per_query[(b, sl)]; d = (rb.rr_model - ra.rr_model)[mask]
    g = d.groupby(ra.key.values[mask]).agg(["sum", "size"]); idx = np.random.default_rng(0).integers(len(g), size=(n_boot, len(g)))
    boots = g["sum"].values[idx].sum(1) / g["size"].values[idx].sum(1)
    return dict(n=int(mask.sum()), ridge=ra.rr_model[mask].mean(), other=rb.rr_model[mask].mean(), diff=d.mean(), ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5))
rows = []
for variant in ("mlp", "ens_w0.5"):
    for gname, smask in [("enveda-180", src == "enveda-180"), ("all non-enveda-180", src != "enveda-180")]:
        for l in labs:
            m = smask & (cat == l)
            if m.sum() >= 30: rows.append(dict(variant=variant, group=gname, max_tanimoto_to_train=l, **gain("ridge", variant, m)))
nov = pd.DataFrame(rows); nov.to_csv("artifacts/p5_ensemble_novelty.csv", index=False); print(nov.round(4).to_string(index=False))
