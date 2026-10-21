"""P5b follow-up: does the DEEP residual MLP's gain survive when the validation molecule has no close analogue in training?
Scores ridge_A/D and deep_A/D, then reports (i) deep minus Ridge at tier D and (ii) deep D minus deep A, by max Tanimoto of the val
molecule to the tier-D training molecules, split enveda-180 vs the other sources; plus a per-source table.
Run from the repo root:  python work/p5b_deep_novelty.py   -> artifacts/scale/deep_novelty.csv, deep_by_source.csv
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import benchmark as bm
import v1_pipeline as v1

OUT = "artifacts/scale"
R = pd.read_parquet(f"{OUT}/rows_train.parquet"); fp = np.load(f"{OUT}/mol_fp.npy"); val_idx = np.load(f"{OUT}/val_mol_idx.npy")
D = v1.load_data(); B = bm.Bench(D, all_val=True); va = D["va"]; src = va.ingest_lib.values
for r in ["ridge_A", "ridge_D", "mlp_D", "deep_A", "deep_D"]:
    B.score(r, np.load(f"{OUT}/pred_{r}.npy")); print("scored", r, flush=True)

labs = ["<0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", ">=0.6"]
uv, inv = np.unique(val_idx, return_inverse=True); A_ = fp[uv].astype(np.float32); na = A_.sum(1)
tm = np.unique(R.mol_idx.values[R["in_D"].values]); T = fp[tm].astype(np.float32); nt = T.sum(1); mx = np.zeros(len(uv), np.float32)
for i in range(0, len(uv), 500):
    inter = A_[i:i + 500] @ T.T; mx[i:i + 500] = (inter / (na[i:i + 500, None] + nt[None, :] - inter + 1e-9)).max(1)
cat = pd.cut(mx[inv], [0, .3, .4, .5, .6, 1.01], labels=labs, right=False)

def gain(a, b, mask, sl="all_val", n_boot=500):
    ra, rb = B.per_query[(a, sl)], B.per_query[(b, sl)]; d = (rb.rr_model - ra.rr_model)[mask]
    g = d.groupby(ra.key.values[mask]).agg(["sum", "size"]); idx = np.random.default_rng(0).integers(len(g), size=(n_boot, len(g)))
    boots = g["sum"].values[idx].sum(1) / g["size"].values[idx].sum(1)
    return dict(n=int(mask.sum()), base=ra.rr_model[mask].mean(), other=rb.rr_model[mask].mean(), diff=d.mean(), ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5))
rows = []
for a, b, lab in [("ridge_D", "deep_D", "deep minus ridge @ D"), ("mlp_D", "deep_D", "deep minus plain MLP @ D"), ("deep_A", "deep_D", "deep: D minus A"), ("ridge_A", "ridge_D", "ridge: D minus A")]:
    for gname, smask in [("enveda-180", src == "enveda-180"), ("all non-enveda-180", src != "enveda-180")]:
        for l in labs:
            m = smask & (cat == l)
            if m.sum() >= 30: rows.append(dict(comparison=lab, group=gname, max_tanimoto=l, **gain(a, b, m)))
pd.DataFrame(rows).to_csv(f"{OUT}/deep_novelty.csv", index=False)
rows = []
for a, b, lab in [("ridge_D", "deep_D", "deep minus ridge @ D"), ("ridge_A", "deep_D", "deep D minus ridge A (the old baseline)")]:
    for s in pd.Series(src).value_counts().index:
        if (src == s).sum() >= 30: rows.append(dict(comparison=lab, source=s, **gain(a, b, src == s)))
    rows.append(dict(comparison=lab, source="ALL", **gain(a, b, np.ones(len(va), bool))))
pd.DataFrame(rows).to_csv(f"{OUT}/deep_by_source.csv", index=False)
pd.set_option("display.width", 250)
print(pd.read_csv(f"{OUT}/deep_novelty.csv").round(4).to_string(index=False)); print(pd.read_csv(f"{OUT}/deep_by_source.csv").round(4).to_string(index=False))
