"""P3: tolerance sweep and mass-window policy on the 230k-pool benchmark (v0.1 Ridge, nothing else changes).

Run from the repo root:  python work/p3_benchmark.py
Writes artifacts/mass_bias_enveda180_v1.csv, artifacts/p3_benchmark.csv, artifacts/p3_by_adduct.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "src")
import mass_window as mw
import v1_pipeline as v1

# ---- 1. per-adduct calibration offset of the test-like instrument, from ALL enveda-180 rows -------------------
b_path = "artifacts/mass_bias_enveda180_v1.csv"
if not os.path.exists(b_path):
    m = pq.read_table(v1.TRAIN_PATH, columns=["ingest_lib", "adduct", "precursor_mz", "molecular_formula"]).to_pandas()
    m = m[(m.ingest_lib == "enveda-180") & m.adduct.isin(mw.SHIFT)].copy()
    m["M"] = m.molecular_formula.map(mw.formula_mass)
    m = m.dropna(subset=["M"])
    err = mw.signed_error_ppm(mw.neutral_mass(m.precursor_mz, m.adduct), m.M)
    mw.fit_bias(err, m.adduct.values).to_csv(b_path)
bias_all = pd.read_csv(b_path, index_col=0)          # fitted on ALL enveda-180 rows (used by the notebook cells)
print(bias_all.round(2))

# ---- 2. rebuild v0.1 (split, features, Ridge) and the 229,964-molecule pool -------------------------------
S = v1.build_v1()
tr, va, reg = S["tr"], S["va"], S["reg"]
pool_keys, pool_fp, pool_mass = v1.load_pool(S["mols_u"], S["FP"], S["mass"])
print("pool:", len(pool_keys))

# leak-free offsets for the benchmark: fitted ONLY on the training-split enveda-180 rows (true mass from the SMILES)
tr180 = tr[tr.ingest_lib == "enveda-180"]
true_m = pd.Series(S["mass"], index=S["mols_u"].key)[tr180.key].values
bias_tab = mw.fit_bias(mw.signed_error_ppm(tr180.neutral_mass.values, true_m), tr180.adduct.values)
BIAS = bias_tab.bias_ppm.to_dict()
HW = mw.halfwidth_by_adduct(bias_tab)                 # 3.5 robust sigmas, >= 3 ppm
print("train-only offsets vs all-rows offsets (ppm):")
print(pd.DataFrame({"train_only": bias_tab.bias_ppm, "all_rows": bias_all.bias_ppm, "halfwidth": pd.Series(HW)}).round(2))
bias_tab.to_csv("artifacts/mass_bias_enveda180_trainsplit_v1.csv")
order = np.argsort(pool_mass)
ms = pool_mass[order]
inv = np.empty(len(order), int)
inv[order] = np.arange(len(order))
key_pos = pd.Series(np.arange(len(pool_keys)), index=pool_keys)
H = np.concatenate([[0.0], np.cumsum(1 / np.arange(1, 26))])          # H[k] = 1 + 1/2 + ... + 1/k


def evaluate(Pq, true_keys, lo, hi):
    """Per query: candidates in [lo, hi], is the truth inside, model reciprocal rank, EXPECTED random-order rr."""
    tp = inv[key_pos[true_keys].values]
    out = np.zeros((len(lo), 4))
    for i in range(len(lo)):
        a, b = np.searchsorted(ms, [lo[i], hi[i]])
        n, inside = b - a, a <= tp[i] < b
        rr_m = rr_r = 0.0
        if inside:
            idx = order[a:b]
            s = pool_fp[idx] @ Pq[i]
            rank = 1 + int((s > s[np.flatnonzero(idx == order[tp[i]])[0]]).sum())
            rr_m = 1 / rank if rank <= 25 else 0.0
            rr_r = H[min(n, 25)] / n                                   # expected 1/rank under random order
        out[i] = (n, inside, rr_m, rr_r)
    return pd.DataFrame(out, columns=["cands", "inside", "rr_model", "rr_random"])


def summarise(r):
    return dict(n_queries=len(r), model=r.rr_model.mean(), random=r.rr_random.mean(),
                model_minus_random=r.rr_model.mean() - r.rr_random.mean(),
                cands_mean=r.cands.mean(), cands_median=r.cands.median(), miss_pct=(1 - r.inside.mean()) * 100)


# ---- 3. query slices ---------------------------------------------------------------------------------------
rng = np.random.default_rng(42)
vi = rng.choice(len(va), size=min(2000, len(va)), replace=False)       # the v0.1 eval subset (now seeded)
slices = {"all_val_2000": vi,
          "enveda180_val": np.flatnonzero((va.ingest_lib == "enveda-180").values)}
rows, by_adduct = [], []
for sname, ix in slices.items():
    q = va.iloc[ix]
    Pq = v1.unit(reg.predict(S["Xva"][ix]))
    pol = [("v01_uncentered", t, None) for t in (2, 3, 4, 5, 7.5, 10, 15, 20, 25)]
    if sname == "enveda180_val":                       # offsets were fitted on enveda-180, so only valid for this slice
        pol += [("centered", t, BIAS) for t in (1, 2, 3, 4, 5, 7.5, 10)]
        pol += [("centered_adduct_sigma", HW, BIAS)]
    for pname, tol, bias in pol:
        lo, hi = mw.window(q.neutral_mass.values, q.adduct.values, tol, bias)
        r = evaluate(Pq, q.key.values, lo, hi)
        tol_lab = tol if np.isscalar(tol) else -1                      # -1 = per-adduct widths (3.5 sigma, >=3 ppm)
        rows.append(dict(slice=sname, policy=pname, tol_ppm=tol_lab, **summarise(r)))
        if tol_lab in (3, 5, 10, -1):
            for a, g in r.groupby(q.adduct.values):
                by_adduct.append(dict(slice=sname, policy=pname, tol_ppm=tol_lab, adduct=a, **summarise(g)))
        print(rows[-1]["slice"], pname, tol, {k: round(v, 4) for k, v in rows[-1].items() if k not in ("slice", "policy", "tol_ppm")}, flush=True)
res = pd.DataFrame(rows)
res.to_csv("artifacts/p3_benchmark.csv", index=False)
pd.DataFrame(by_adduct).to_csv("artifacts/p3_by_adduct.csv", index=False)
print(res.round(4).to_string(index=False))
