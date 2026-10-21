"""Score every saved (model, tier) prediction on both benchmark slices + per-bit AUC, paired bootstraps, and a novelty breakdown.

Run from the repo root AFTER the fits:  python work/p5b_score.py
-> artifacts/scale/scale_scores.csv, scale_bootstrap.csv, scale_auc.csv, scale_novelty.csv
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

OUT = "artifacts/scale"
R = pd.read_parquet(f"{OUT}/rows_train.parquet"); fp = np.load(f"{OUT}/mol_fp.npy"); val_idx = np.load(f"{OUT}/val_mol_idx.npy")
runs = sorted(os.path.basename(f)[5:-4] for f in glob.glob(f"{OUT}/pred_*.npy"))            # e.g. ridge_A
D = v1.load_data(); B = bm.Bench(D, all_val=True); va = D["va"]
preds = {r: np.load(f"{OUT}/pred_{r}.npy") for r in runs}
rows = []
for r, P in preds.items():
    model, tier = r.split("_")
    h = json.load(open(f"{OUT}/hist_{r}.json"))
    s = B.score(r, P).assign(arch=model, tier=tier, n_spectra=h["n_spectra"], n_molecules=h["n_molecules"], minutes=h["seconds"] / 60,
                             best_epoch=h.get("best_epoch"), holdout_bce=h.get("best_holdout_bce"))
    rows.append(s); print(s[s.slice.isin(["enveda180", "mixed_2000"])][["version", "slice", "model", "model_minus_random", "n_spectra"]].round(4).to_string(index=False), flush=True)
sc = pd.concat(rows); sc.to_csv(f"{OUT}/scale_scores.csv", index=False)

# ---- per-bit AUC ---------------------------------------------------------------------------------------------------------
Yva = fp[val_idx]; e180 = (va.ingest_lib == "enveda-180").values
def mean_auc(P, mask, min_n=30):
    Y, S = Yva[mask], P[mask]; pos = Y.sum(0); ok = (pos >= min_n) & (Y.shape[0] - pos >= min_n)
    r_ = rankdata(S[:, ok], axis=0); n1 = pos[ok]; n0 = Y.shape[0] - n1
    return float((((r_ * Y[:, ok]).sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)).mean())
auc = pd.DataFrame([dict(run=r, group=g, mean_bit_auc=mean_auc(P, m)) for r, P in preds.items() for g, m in [("all val", np.ones(len(va), bool)), ("enveda-180 val", e180)]])
auc.to_csv(f"{OUT}/scale_auc.csv", index=False)

# ---- paired bootstraps ----------------------------------------------------------------------------------------------------
pairs = []
for t in ["A25", "A50", "A", "B", "D"]:
    for m_ in ("mlp", "deep"):
        if f"{m_}_{t}" in preds and f"ridge_{t}" in preds: pairs.append((f"ridge_{t}", f"{m_}_{t}", f"{m_} vs ridge @ {t}"))
for m_ in ("ridge", "mlp", "deep"):
    if f"{m_}_A" in preds and f"{m_}_D" in preds: pairs.append((f"{m_}_A", f"{m_}_D", f"{m_}: tier D vs A"))
    if f"{m_}_A25" in preds and f"{m_}_A" in preds: pairs.append((f"{m_}_A25", f"{m_}_A", f"{m_}: tier A vs A25"))
boot = []
for a, b, lab in pairs:
    for sl in ("enveda180", "mixed_2000", "all_val"):
        d, (lo, hi) = B.paired_diff(a, b, sl); boot.append(dict(comparison=lab, slice=sl, diff=d, ci_lo=lo, ci_hi=hi))
pd.DataFrame(boot).to_csv(f"{OUT}/scale_bootstrap.csv", index=False)

# ---- novelty: gain by max Tanimoto of the val molecule to THAT tier's training molecules ----------------------------------
labs = ["<0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", ">=0.6"]; vk = va.key.values
uv, inv = np.unique(val_idx, return_inverse=True); A_ = fp[uv].astype(np.float32); na = A_.sum(1)
def novelty(tier):
    tm = np.unique(R.mol_idx.values[R[f"in_{tier}"].values]); T = fp[tm].astype(np.float32); nt = T.sum(1); out = np.zeros(len(uv), np.float32)
    for i in range(0, len(uv), 500):
        inter = A_[i:i + 500] @ T.T; out[i:i + 500] = (inter / (na[i:i + 500, None] + nt[None, :] - inter + 1e-9)).max(1)
    return pd.cut(out[inv], [0, .3, .4, .5, .6, 1.01], labels=labs, right=False)
def gain(a, b, mask, sl="all_val", n_boot=500):
    ra, rb = B.per_query[(a, sl)], B.per_query[(b, sl)]; d = (rb.rr_model - ra.rr_model)[mask]
    g = d.groupby(ra.key.values[mask]).agg(["sum", "size"]); idx = np.random.default_rng(0).integers(len(g), size=(n_boot, len(g)))
    boots = g["sum"].values[idx].sum(1) / g["size"].values[idx].sum(1)
    return dict(n=int(mask.sum()), base=ra.rr_model[mask].mean(), other=rb.rr_model[mask].mean(), diff=d.mean(), ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5))
src = va.ingest_lib.values; nov = []
for t in ["A", "D"]:
    if f"ridge_{t}" not in preds: continue
    cat = novelty(t)
    for gname, smask in [("enveda-180", src == "enveda-180"), ("all non-enveda-180", src != "enveda-180")]:
        for l in labs:
            m = smask & (cat == l)
            if m.sum() < 30: continue
            if f"mlp_{t}" in preds: nov.append(dict(tier=t, comparison="mlp minus ridge", group=gname, max_tanimoto=l, **gain(f"ridge_{t}", f"mlp_{t}", m)))
for m_ in ("ridge", "mlp"):
    if f"{m_}_A" in preds and f"{m_}_D" in preds:
        catD = novelty("D")           # similarity to the LARGE training set
        for gname, smask in [("enveda-180", src == "enveda-180"), ("all non-enveda-180", src != "enveda-180")]:
            for l in labs:
                m = smask & (catD == l)
                if m.sum() >= 30: nov.append(dict(tier="D", comparison=f"{m_}: D minus A", group=gname, max_tanimoto=l, **gain(f"{m_}_A", f"{m_}_D", m)))
pd.DataFrame(nov).to_csv(f"{OUT}/scale_novelty.csv", index=False)
pd.set_option("display.width", 250)
print(sc[sc.slice.isin(["enveda180", "mixed_2000"])].pivot_table(index=["arch", "tier"], columns="slice", values="model", sort=False).round(4))
print(pd.DataFrame(boot).round(4).to_string(index=False)); print(pd.DataFrame(nov).round(4).to_string(index=False))
